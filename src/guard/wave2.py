"""Wave-2 GUARD studies: multilingual validity, multi-domain transfer, paraphrase
attack, and pre-deployment certificate diagnostics (W1-W4).

Scientific purpose
------------------
Wave 1 (guard.experiments, E1-E8) mapped the split-conformal false-accusation
certificate on English hotel reviews. Wave 2 stresses the same certificate along the
axes a real deployer actually moves on, and evaluates the pre-deployment diagnostics
(guard.conformal.predicted_fpr_bounds / pilot_fpr_ucb) that are supposed to warn them:

  W1 multilingual    - per-language in-language validity (expected exact: humans of one
                       language resampled into calibration/test are exchangeable by
                       construction) and power, plus the CROSS-LANGUAGE transfer map:
                       calibrate on English humans, test other languages' humans (the
                       deployment mistake of reusing an English calibration set), and
                       the full language x language FPR matrix for cheap scorers.
  W2 domains         - the same transfer map across human DOMAINS: hotel reviews (en),
                       movie reviews, paper abstracts, books, news, poetry (RAID), with
                       per-domain in-domain validity and power as controls.
  W3 paraphrase      - a paraphrase-model editor (guard.paraphrase.T5Paraphraser)
                       replacing wave 1's lexical proxies: paraphrased humans as the
                       certificate attack, paraphrased AI as the evasion attack, and the
                       E8-style editor-aware robust calibration (max over k modeled
                       paraphrases per calibration text) as the candidate fix.
  W4 diagnostics     - for EVERY shift condition above (each non-English language, each
                       non-hotel domain, paraphrased humans) plus an in-domain control:
                       does a small pilot of m known-human deployment texts predict the
                       realized FPR? Reports realized per-repeat FPR, the exact-binomial
                       alarm rate of pilot_fpr_ucb at m in {25,50,100,200}, and the
                       plug-in TV / Gamma population-shift bounds of predicted_fpr_bounds.

Design principle (inherited from guard.experiments): SCORE ONCE, RESAMPLE MANY. Every
unique text is scored exactly once -- GPT-2 statistics through the parquet cache owned by
guard.scores.compute_scores, TF-IDF recomputed from a fit on the English TRAIN-half of
the entity-grouped master split -- and all R repeats are pure index resampling over the
precomputed score arrays. Optional supervised scorers ("supervised" = TransformerScorer
fit on the same English train-half, "offshelf" = an off-the-shelf detector) are imported
from guard.supervised lazily and merged as extra score columns; their absence degrades
to a warning, never a crash. Likewise guard.paraphrase: if the paraphraser is
unavailable, W3 (and its W4 condition) is skipped with an explicit reason.

Leakage protocol: the English train/eval halves come from guard.data.grouped_split by
hotel with the same seed as the wave-1 engine, so supervised scorers never see
evaluation entities. Non-English languages and non-hotel domains contribute evaluation
text only; nothing is ever fit on them.

Everything is deterministic given cfg["seed"]; all randomness flows through
numpy.random.default_rng seeded with (seed, stream_id[, item]) tuples; paraphrase seeds
live in disjoint ranges (test humans 10000+i, calibration variants 20000+k*i+j, AI
30000+j) so calibration edits are never test edits. No experimental result is
hard-coded; run_all returns one JSON-serialisable dict.
"""
from __future__ import annotations

import inspect
import math
import platform
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from guard.conformal import (
    conformal_pvalues,
    pilot_fpr_ucb,
    predicted_fpr_bounds,
    robust_calibration_scores,
    violation_pvalue,
)
from guard.experiments import ScoreBook, _jsonable  # serializer shared with wave 1
from guard.scores import GPT2_STATS, GPT2Scores, TfidfScorer, compute_scores

_ROOT = Path(__file__).resolve().parents[2]

#: Index of the auto-converted parquet shards of RAID on the HF datasets server.
RAID_PARQUET_INDEX = "https://datasets-server.huggingface.co/parquet?dataset=liamdugan/raid"

DEFAULT_CFG: dict[str, Any] = {
    "data_csv": str(_ROOT / "data" / "all_data.csv"),
    "reviews_pool": str(_ROOT / "results" / "cache" / "raid_reviews_pool.parquet"),
    "abstracts_parquet": str(_ROOT / "results" / "cache" / "raid_abstracts.parquet"),
    "cache_path": str(_ROOT / "results" / "cache" / "scores.parquet"),
    "cache_dir": str(_ROOT / "results" / "cache"),
    "scorers": ["loglik", "tfidf"],
    "with_supervised": False,   # adds "supervised" + "offshelf" via guard.supervised
    "alphas": [0.01, 0.05, 0.10],
    "R": 200,
    "seed": 0,
    "device": "cpu",            # GPT-2 / paraphraser / supervised inference device
    "master_split_seed": 0,     # same entity split as the wave-1 engine
    "lang_cap_human": 500,      # per-language human cap (seed-fixed sample)
    "lang_cap_ai": 300,         # per-language AI cap
    "fetch_domains": ["books", "news", "poetry"],
    "domain_fetch_cap": 300,    # per-class LIMIT of the duckdb fetch (cache-stable)
    "fetch_if_missing": True,   # False: use only already-cached domain pools
    "max_para": 500,            # paraphrased humans (W3 certificate attack)
    "para_ai_cap": 300,         # paraphrased AI texts (W3 evasion attack)
    "para_cache": str(_ROOT / "results" / "cache" / "paraphrases.parquet"),
    "k_cal_variants": 2,        # modeled-editor paraphrases per calibration human
    "pilot_sizes": [25, 50, 100, 200],
    "alarm_level": 0.10,        # pilot_fpr_ucb alarm level (calibrated failure detector)
}

_MATRIX_SCORERS = ("loglik", "tfidf")  # full LxL / DxD matrices kept to the cheap pair


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #

def _akey(alpha: float) -> str:
    """Stable string key for an alpha level ('0.01', '0.05', '0.1')."""
    return f"{float(alpha):g}"


def _agg(values: Iterable[float]) -> dict[str, float | None]:
    """Mean and 95% percentile interval over finite entries of a per-repeat array."""
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None}
    return {
        "mean": float(finite.mean()),
        "ci_lo": float(np.percentile(finite, 2.5)),
        "ci_hi": float(np.percentile(finite, 97.5)),
    }


def _finite_mean(values: Iterable[float]) -> float | None:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else None


def _norm_ws(s: str) -> str:
    """Collapse whitespace runs to single spaces (guard.data convention)."""
    return " ".join(str(s).split())


def _cap_list(items: Sequence[str], cap: int | None, rng: np.random.Generator) -> list[str]:
    """Seed-fixed subsample of `items` to at most `cap`, preserving original order."""
    items = list(items)
    if cap is None or len(items) <= int(cap):
        return items
    keep = np.sort(rng.choice(len(items), size=int(cap), replace=False))
    return [items[i] for i in keep]


def _split_masks(n: int, R: int, rng: np.random.Generator) -> list[np.ndarray]:
    """R boolean masks over range(n): True = calibration half (n // 2 members)."""
    n_cal = n // 2
    masks: list[np.ndarray] = []
    for _ in range(R):
        m = np.zeros(n, dtype=bool)
        m[rng.permutation(n)[:n_cal]] = True
        masks.append(m)
    return masks


def _finite(cal: np.ndarray) -> np.ndarray:
    """Calibration scores with non-finite entries dropped (a NaN cal score would
    corrupt the sorted-search inside conformal_pvalues)."""
    cal = np.asarray(cal, dtype=float)
    return cal[np.isfinite(cal)]


def _params_of(fn: Callable) -> dict | None:
    try:
        return dict(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return None


def _human_record(alpha: float, fpr: list[float], violation_p: list[float],
                  n_cal: int, n_test: int, **extra: Any) -> dict:
    return {
        "alpha": float(alpha), "n_cal": int(n_cal), "n_test_human": int(n_test),
        "fpr": fpr, "fpr_summary": _agg(fpr),
        "violation_p": violation_p, "violation_p_mean": _finite_mean(violation_p),
        **extra,
    }


def _ai_record(alpha: float, tpr: list[float], n_cal: int, n_ai: int, **extra: Any) -> dict:
    return {
        "alpha": float(alpha), "n_cal": int(n_cal), "n_test_ai": int(n_ai),
        "tpr": tpr, "tpr_summary": _agg(tpr),
        **extra,
    }


# --------------------------------------------------------------------------- #
# Data: multilingual MAiDE-up and the RAID domain pools
# --------------------------------------------------------------------------- #

def load_multilingual(csv_path: str | Path) -> pd.DataFrame:
    """Load the FULL multilingual MAiDE-up CSV as [text, label, language, hotel].

    text = Upside_Review + ' ' + Downside_Review (missing halves empty,
    whitespace-normalised); rows with empty combined text are dropped.
    label = int(source): 0 human, 1 AI. All 10 Review_Language values are kept.
    """
    df = pd.read_csv(csv_path)
    up = df["Upside_Review"].fillna("").astype(str)
    down = df["Downside_Review"].fillna("").astype(str)
    out = pd.DataFrame(
        {
            "text": (up + " " + down).map(_norm_ws),
            "label": pd.to_numeric(df["source"]).astype(int),
            "language": df["Review_Language"].astype(str),
            "hotel": df["Hotel Name"].astype(str),
        }
    )
    return out[out["text"].str.len() > 0].reset_index(drop=True)


def _raid_parquet_urls(config: str = "raid", split: str = "train",
                       timeout: float = 60.0) -> list[str]:
    """Parquet-shard URLs of the RAID conversion from the HF datasets server."""
    import json

    with urllib.request.urlopen(RAID_PARQUET_INDEX, timeout=timeout) as resp:
        payload = json.load(resp)
    urls = [
        f["url"]
        for f in payload.get("parquet_files", [])
        if f.get("config") == config and f.get("split") == split
    ]
    if not urls:
        raise RuntimeError(
            f"no parquet files for config={config!r} split={split!r} at {RAID_PARQUET_INDEX}"
        )
    return urls


def fetch_domain_pools(
    domains: Sequence[str],
    cap_per_class: int = 300,
    cache_dir: str | Path = "results/cache",
    allow_fetch: bool = True,
) -> dict[str, pd.DataFrame]:
    """Human + AI pools for RAID domains, via duckdb httpfs over the converted shards.

    For each domain, selects unattacked rows (attack = 'none') and takes the first
    `cap_per_class` rows per class in md5(generation) order -- a deterministic,
    order-independent pseudo-random sample that is identical across machines and runs.
    Results are cached as parquet (text, label, model) per domain under `cache_dir`;
    a cached pool is reused regardless of cap (delete the file to refetch). Domains
    that cannot be fetched (no cache and allow_fetch=False, or network/duckdb failure)
    are silently absent from the returned dict; callers decide how loudly to complain.

    NOTE: the auto-converted shards contain only a subset of RAID's domains
    (abstracts/books/news/poetry); the reviews-domain pool is cached locally by the
    wave-1 pipeline and is NOT fetchable through this helper.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    urls: list[str] | None = None
    for domain in domains:
        cache_file = cache_dir / f"raid_{domain}_pool.parquet"
        if cache_file.exists():
            out[domain] = pd.read_parquet(cache_file)
            continue
        if not allow_fetch:
            print(f"[guard.wave2] domain {domain!r}: no cache and fetching disabled; skipped",
                  flush=True)
            continue
        try:
            import duckdb

            if urls is None:
                urls = _raid_parquet_urls()
            url_list = ", ".join("'" + u.replace("'", "''") + "'" for u in urls)
            query = f"""
                WITH pool AS (
                    SELECT generation AS text, model,
                           CASE WHEN model = 'human' THEN 0 ELSE 1 END AS label
                    FROM read_parquet([{url_list}])
                    WHERE domain = ? AND attack = 'none'
                      AND generation IS NOT NULL AND length(generation) > 0
                ), ranked AS (
                    SELECT text, label, model,
                           row_number() OVER (PARTITION BY label ORDER BY md5(text)) AS rk
                    FROM pool
                )
                SELECT text, label, model FROM ranked WHERE rk <= ? ORDER BY label, rk
            """
            print(f"[guard.wave2] fetching RAID domain {domain!r} "
                  f"(cap {cap_per_class}/class) via duckdb httpfs ...", flush=True)
            t0 = time.time()
            con = duckdb.connect()
            try:
                con.execute("INSTALL httpfs; LOAD httpfs;")
                df = con.execute(query, [domain, int(cap_per_class)]).df()
            finally:
                con.close()
            df["text"] = df["text"].astype(str).map(_norm_ws)
            df["label"] = df["label"].astype(int)
            df = df[df["text"].str.len() > 0].reset_index(drop=True)
            df.to_parquet(cache_file)
            print(f"[guard.wave2] domain {domain!r}: {int((df['label'] == 0).sum())} human / "
                  f"{int((df['label'] == 1).sum())} AI in {time.time() - t0:.1f}s "
                  f"-> {cache_file.name}", flush=True)
            out[domain] = df
        except Exception as exc:
            print(f"[guard.wave2] domain {domain!r}: fetch failed ({exc!r}); skipped",
                  flush=True)
    return out


def _local_pool(path: str | Path, warnings_list: list[str], tag: str) -> pd.DataFrame | None:
    """Load a locally cached RAID pool (text/label/model[/attack]), unattacked rows only."""
    p = Path(path)
    if not p.exists():
        warnings_list.append(f"{tag}: cache {p} absent; domain skipped")
        return None
    df = pd.read_parquet(p)
    if "attack" in df.columns:
        df = df[df["attack"].fillna("none").astype(str).str.lower().isin({"none", ""})]
    df = df[["text", "label", "model"]].copy()
    df["text"] = df["text"].astype(str).map(_norm_ws)
    df["label"] = pd.to_numeric(df["label"]).astype(int)
    return df[df["text"].str.len() > 0].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Sibling-module adapters (written in parallel; tolerate naming differences)
# --------------------------------------------------------------------------- #

def _make_paraphraser(device: str, warnings_list: list[str]) -> Any | None:
    """Instantiate guard.paraphrase.T5Paraphraser (lazy; None + warning if unavailable)."""
    try:
        from guard import paraphrase as guard_paraphrase
    except Exception as exc:  # ImportError or any load-time failure of the sibling
        warnings_list.append(f"guard.paraphrase unavailable ({exc!r}); W3 skipped")
        return None
    cls = None
    for name in ("T5Paraphraser", "Paraphraser"):
        cls = getattr(guard_paraphrase, name, None)
        if cls is not None:
            break
    if cls is None:
        warnings_list.append("guard.paraphrase exposes no T5Paraphraser; W3 skipped")
        return None
    try:
        params = _params_of(cls)
        kwargs = {"device": device} if params and "device" in params else {}
        return cls(**kwargs)
    except Exception as exc:
        warnings_list.append(f"T5Paraphraser construction failed ({exc!r}); W3 skipped")
        return None


def _paraphrase_one(pp: Any, text: str, seed: int) -> str:
    """One paraphrase of one text, tolerating small API differences in the sibling."""
    for name in ("paraphrase", "paraphrase_text", "rewrite", "generate", "__call__"):
        fn = getattr(pp, name, None)
        if not callable(fn):
            continue
        params = _params_of(fn)
        kwargs: dict[str, Any] = {}
        if params is not None:
            if "seed" in params:
                kwargs["seed"] = int(seed)
            elif "rng" in params:
                kwargs["rng"] = np.random.default_rng(int(seed))
        out = fn(text, **kwargs)
        if isinstance(out, (list, tuple)):
            out = out[0] if out else text
        return _norm_ws(str(out))
    raise AttributeError("paraphraser exposes no recognised paraphrase method")


def _paraphrase_texts(pp: Any, texts: Sequence[str], seeds: Sequence[int],
                      warnings_list: list[str], tag: str,
                      cache_path: str | Path | None = None) -> list[str]:
    """Paraphrase each text with its own seed; failures keep the original (conservative).

    Prefers the paraphraser's batched ``paraphrase_many(texts, seeds, cache_path)``
    (persistent per-(text, seed) cache: each pair is generated once ever) and falls
    back to per-text calls when that method or the cache path is unavailable.
    """
    many = getattr(pp, "paraphrase_many", None)
    if callable(many) and cache_path is not None:
        try:
            print(f"[guard.wave2] paraphrase {tag}: {len(texts)} texts via cache "
                  f"{Path(cache_path).name}", flush=True)
            return [_norm_ws(t) for t in many(list(texts), [int(s) for s in seeds], cache_path)]
        except Exception as exc:
            warnings_list.append(
                f"paraphrase_many failed for {tag} ({exc!r}); per-text fallback"
            )
    out: list[str] = []
    failures = 0
    for i, (t, s) in enumerate(zip(texts, seeds)):
        if i % 50 == 0:
            print(f"[guard.wave2] paraphrase {tag}: {i}/{len(texts)}", flush=True)
        try:
            out.append(_paraphrase_one(pp, t, s))
        except Exception:
            failures += 1
            out.append(str(t))
    if failures:
        warnings_list.append(f"paraphrase {tag}: {failures}/{len(texts)} texts left unedited")
    print(f"[guard.wave2] paraphrase {tag}: {len(texts)}/{len(texts)} done", flush=True)
    return out


_SUPERVISED_CLASSES: dict[str, tuple[str, ...]] = {
    "supervised": ("TransformerScorer",),
    "offshelf": (
        "OffShelfScorer", "OffTheShelfScorer", "OffShelfDetector",
        "PretrainedDetector", "RobertaDetectorScorer", "OpenAIDetectorScorer",
    ),
}


def _score_with(obj: Any, texts: Sequence[str]) -> np.ndarray:
    for name in ("score", "scores", "score_texts", "predict_scores", "decision_function"):
        fn = getattr(obj, name, None)
        if callable(fn):
            return np.asarray(fn(list(texts)), dtype=float)
    if callable(obj):
        return np.asarray(obj(list(texts)), dtype=float)
    raise AttributeError(f"{type(obj).__name__} exposes no scoring method")


def _supervised_columns(
    names: Sequence[str],
    texts: Sequence[str],
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    device: str,
    warnings_list: list[str],
) -> dict[str, np.ndarray]:
    """Extra score columns from guard.supervised (lazy import; drop-with-warning on failure).

    "supervised" is fit on the English TRAIN-half only (leakage contract identical to
    TF-IDF); "offshelf" is used as shipped (fit, if any, happened elsewhere).
    """
    out: dict[str, np.ndarray] = {}
    try:
        from guard import supervised as guard_supervised
    except Exception as exc:
        warnings_list.append(f"guard.supervised unavailable ({exc!r}); supervised scorers dropped")
        return out
    for name in names:
        obj = None
        for cls_name in _SUPERVISED_CLASSES.get(name, ()):
            cls = getattr(guard_supervised, cls_name, None)
            if cls is None:
                continue
            try:
                params = _params_of(cls)
                kwargs = {"device": device} if params and "device" in params else {}
                obj = cls(**kwargs)
                break
            except Exception as exc:
                warnings_list.append(f"guard.supervised.{cls_name}() failed ({exc!r})")
        if obj is None:
            warnings_list.append(f"scorer {name!r}: no usable class in guard.supervised; dropped")
            continue
        try:
            if name == "supervised":
                fit = getattr(obj, "fit", None)
                if callable(fit):
                    fitted = fit(list(train_texts), np.asarray(train_labels, dtype=int))
                    obj = fitted if fitted is not None else obj
            arr = _score_with(obj, texts)
            if arr.shape == (len(texts),):
                out[name] = arr.astype(float)
            else:
                warnings_list.append(f"scorer {name!r}: score shape {arr.shape} unusable; dropped")
        except Exception as exc:
            warnings_list.append(f"scorer {name!r} failed ({exc!r}); dropped")
    return out


# --------------------------------------------------------------------------- #
# Shared evaluation kernels (W1 and W2 are the same study over different pools)
# --------------------------------------------------------------------------- #

def _validity_and_power(
    book: ScoreBook,
    scorers: Sequence[str],
    h_idx: np.ndarray,
    ai_idx: np.ndarray,
    masks: list[np.ndarray],
    alphas: Sequence[float],
) -> tuple[dict, dict]:
    """In-pool validity (cal/test halves of the pool's humans) and power on its AI."""
    n_cal = int(h_idx.size) // 2
    n_test = int(h_idx.size) - n_cal
    validity: dict[str, dict] = {}
    power: dict[str, dict] = {}
    for scorer in scorers:
        sarr = book.get(scorer, h_idx)
        aarr = book.get(scorer, ai_idx) if ai_idx.size else None
        cells = {a: {"fpr": [], "violation_p": [], "tpr": []} for a in alphas}
        for mask in masks:
            cal = _finite(sarr[mask])
            pv = conformal_pvalues(cal, sarr[~mask])
            pv_ai = conformal_pvalues(cal, aarr) if aarr is not None else None
            for a in alphas:
                fl = pv <= a
                cells[a]["fpr"].append(float(fl.mean()) if fl.size else math.nan)
                cells[a]["violation_p"].append(
                    violation_pvalue(int(fl.sum()), int(fl.size), int(cal.size), a)
                )
                if pv_ai is not None:
                    cells[a]["tpr"].append(float((pv_ai <= a).mean()))
        validity[scorer] = {
            _akey(a): _human_record(a, cells[a]["fpr"], cells[a]["violation_p"],
                                    n_cal, n_test, scorer=scorer)
            for a in alphas
        }
        if aarr is not None:
            power[scorer] = {
                _akey(a): _ai_record(a, cells[a]["tpr"], n_cal, int(ai_idx.size), scorer=scorer)
                for a in alphas
            }
    return validity, power


def _transfer_row(
    book: ScoreBook,
    scorers: Sequence[str],
    src_name: str,
    src_idx: np.ndarray,
    src_masks: list[np.ndarray],
    pools_idx: dict[str, np.ndarray],
    pools_masks: dict[str, list[np.ndarray]],
    alphas: Sequence[float],
) -> dict:
    """Calibration-transfer row: calibrate on the source pool's cal half, test each
    target pool's TEST half (disjointness mirrors the in-pool splits). Per-repeat FPR
    lists are kept -- this row doubles as the realized-FPR reference for W4. Records
    carry explicit ``source`` / ``target`` labels for guard.viz_wave2.f8_transfer_matrix."""
    R = len(src_masks)
    out: dict[str, dict] = {}
    for scorer in scorers:
        src = book.get(scorer, src_idx)
        per_alpha: dict[str, dict] = {_akey(a): {} for a in alphas}
        for name, t_idx in pools_idx.items():
            tarr = book.get(scorer, t_idx)
            t_masks = pools_masks[name]
            cells = {a: {"fpr": [], "violation_p": []} for a in alphas}
            n_cal = int(src_idx.size) // 2
            for r in range(R):
                cal = _finite(src[src_masks[r]])
                pv = conformal_pvalues(cal, tarr[~t_masks[r]])
                for a in alphas:
                    fl = pv <= a
                    cells[a]["fpr"].append(float(fl.mean()) if fl.size else math.nan)
                    cells[a]["violation_p"].append(
                        violation_pvalue(int(fl.sum()), int(fl.size), int(cal.size), a)
                    )
            for a in alphas:
                per_alpha[_akey(a)][name] = _human_record(
                    a, cells[a]["fpr"], cells[a]["violation_p"],
                    n_cal, int(t_idx.size) - int(t_idx.size) // 2,
                    scorer=scorer, source=src_name, target=name,
                )
        out[scorer] = per_alpha
    return out


def _transfer_matrix(
    book: ScoreBook,
    matrix_scorers: Sequence[str],
    names: Sequence[str],
    pools_idx: dict[str, np.ndarray],
    pools_masks: dict[str, list[np.ndarray]],
    alphas: Sequence[float],
) -> dict:
    """Full source x target FPR matrix (cheap once scored). Each cell is a
    self-describing record (alpha / scorer / source / target + per-repeat ``fpr``
    list + ``fpr_summary``) so guard.viz_wave2.f8_transfer_matrix can read it; the
    exact violation test is omitted here (the in-pool and source-row records carry
    it), so figure-side violation calls on pure-matrix cells fall back to the CI."""
    R = len(next(iter(pools_masks.values())))
    out: dict[str, dict] = {}
    for scorer in matrix_scorers:
        arrs = {n: book.get(scorer, pools_idx[n]) for n in names}
        per_alpha: dict[str, dict] = {_akey(a): {s: {} for s in names} for a in alphas}
        for src in names:
            for tgt in names:
                cells = {a: [] for a in alphas}
                for r in range(R):
                    cal = _finite(arrs[src][pools_masks[src][r]])
                    pv = conformal_pvalues(cal, arrs[tgt][~pools_masks[tgt][r]])
                    for a in alphas:
                        cells[a].append(float((pv <= a).mean()) if pv.size else math.nan)
                for a in alphas:
                    per_alpha[_akey(a)][src][tgt] = {
                        "alpha": float(a), "scorer": scorer,
                        "source": src, "target": tgt,
                        "fpr": cells[a], "fpr_summary": _agg(cells[a]),
                    }
        out[scorer] = per_alpha
    return out


# --------------------------------------------------------------------------- #
# W3: paraphrase attack and editor-aware robust calibration
# --------------------------------------------------------------------------- #

def _run_w3(
    book: ScoreBook,
    scorers: Sequence[str],
    clean_idx: np.ndarray,
    para_idx: np.ndarray,
    variants_idx: np.ndarray,  # (n, k) book indices of calibration paraphrases
    ai_idx: np.ndarray,
    ai_para_idx: np.ndarray,
    masks: list[np.ndarray],
    alphas: Sequence[float],
) -> dict:
    n = int(clean_idx.size)
    k = int(variants_idx.shape[1])
    n_cal = n // 2
    out: dict[str, Any] = {
        "skipped": False,
        "n_human": n,
        "n_ai": int(ai_idx.size),
        "k_cal_variants": k,
        "n_cal": n_cal,
        "n_test_human": n - n_cal,
        "note": (
            "naive = clean-score calibration; robust = editor-aware calibration "
            "(max over clean + k modeled paraphrases per calibration human, "
            "guard.conformal.robust_calibration_scores). Calibration-variant seeds are "
            "disjoint from test-paraphrase seeds, so robustness is to the modeled "
            "editor, not to memorised edits."
        ),
        "per_scorer": {},
    }
    for scorer in scorers:
        clean = book.get(scorer, clean_idx)
        para = book.get(scorer, para_idx)
        var = book.get(scorer, variants_idx.ravel()).reshape(n, k)
        robust_all = robust_calibration_scores(clean, var)
        conds: dict[str, tuple[str, np.ndarray]] = {
            "human_clean": ("human", clean),
            "human_para": ("human", para),
            "ai_clean": ("ai", book.get(scorer, ai_idx)),
            "ai_para": ("ai", book.get(scorer, ai_para_idx)),
        }
        cells = {
            mode: {c: {a: {"rate": [], "violation_p": []} for a in alphas} for c in conds}
            for mode in ("naive", "robust")
        }
        for mask in masks:
            for mode, cal_full in (("naive", clean), ("robust", robust_all)):
                cal = _finite(cal_full[mask])
                for cond, (typ, arr) in conds.items():
                    test = arr[~mask] if typ == "human" else arr
                    pv = conformal_pvalues(cal, test)
                    for a in alphas:
                        fl = pv <= a
                        cell = cells[mode][cond][a]
                        cell["rate"].append(float(fl.mean()) if fl.size else math.nan)
                        if typ == "human":
                            cell["violation_p"].append(
                                violation_pvalue(int(fl.sum()), int(fl.size), int(cal.size), a)
                            )
        out["per_scorer"][scorer] = {
            mode: {
                cond: {
                    _akey(a): (
                        _human_record(
                            a, cells[mode][cond][a]["rate"], cells[mode][cond][a]["violation_p"],
                            n_cal, n - n_cal, scorer=scorer, mode=mode, condition=cond,
                        )
                        if conds[cond][0] == "human"
                        else _ai_record(
                            a, cells[mode][cond][a]["rate"], n_cal, int(conds[cond][1].size),
                            scorer=scorer, mode=mode, condition=cond,
                        )
                    )
                    for a in alphas
                }
                for cond in conds
            }
            for mode in ("naive", "robust")
        }
    return out


# --------------------------------------------------------------------------- #
# W4: pre-deployment diagnostics under every shift condition
# --------------------------------------------------------------------------- #

def _run_w4(
    book: ScoreBook,
    scorers: Sequence[str],
    english_h_idx: np.ndarray,
    english_masks: list[np.ndarray],
    conditions: list[dict],
    alphas: Sequence[float],
    pilot_sizes: Sequence[int],
    alarm_level: float,
    seed: int,
) -> dict:
    """Realized FPR vs what a pilot of m known humans would have predicted.

    Condition modes:
      external - calibration = English cal half (the deployed certificate); the
                 condition pool is split per repeat into a TEST half and a PILOT
                 remainder (disjoint), so the pilot never contains test texts.
      threeway - the condition pool shares its underlying texts with the calibration
                 population (in-domain control; paraphrased humans), so each repeat
                 3-way-splits one permutation: cal half / test quarter / pilot quarter,
                 keeping calibration, test, and pilot entity-disjoint at text level.

    Per condition x scorer x alpha: per-repeat realized FPR + exact Beta-Binomial
    violation p-values; alarm rate of pilot_fpr_ucb at each feasible m; mean plug-in
    TV / Gamma bounds of predicted_fpr_bounds from the largest feasible pilot. The
    'violated' flag is median(violation_p) < 0.05 across repeats.
    """
    R = len(english_masks)
    pilot_sizes = [int(m) for m in pilot_sizes]
    out: dict[str, Any] = {
        "pilot_sizes": pilot_sizes,
        "alarm_level": float(alarm_level),
        "note": (
            "alarm = exact one-sided binomial rejection of FPR <= alpha on the pilot "
            "(calibrated failure detector: fires with prob <= alarm_level under a valid "
            "certificate); tv/gamma bounds are plug-in score-space diagnostics, not "
            "certificates. gamma_hat may be infinite (pilot mass in empty calibration "
            "bins); means are over finite repeats. 'violated' = median exact "
            "Beta-Binomial tail probability < 0.05 across repeats. 'pilot_alarms' and "
            "'bounds' replicate the alarm / bound numbers as flat per-m and per-repeat "
            "records for guard.viz_wave2.f10_diagnostics."
        ),
        "conditions": {},
        "pilot_alarms": {},
        "bounds": {},
    }
    for ci, cond in enumerate(conditions):
        name, kind, mode = cond["name"], cond["kind"], cond["mode"]
        pool_idx: np.ndarray = cond["pool_idx"]
        n_pool = int(pool_idx.size)
        rng = np.random.default_rng((seed, 70, ci))
        perms = [rng.permutation(n_pool) for _ in range(R)]
        if mode == "external":
            n_test = n_pool // 2
            pilot_cap = n_pool - n_test
            n_cal = int(english_h_idx.size) // 2
        else:  # threeway
            n_cal = n_pool // 2
            n_test = (n_pool - n_cal) // 2
            pilot_cap = n_pool - n_cal - n_test
        m_feasible = [m for m in pilot_sizes if m <= pilot_cap]
        m_max = max(m_feasible, default=0)
        cond_out: dict[str, Any] = {}
        alarm_out: dict[str, Any] = {}
        bounds_out: dict[str, Any] = {}
        for scorer in scorers:
            pool_arr = book.get(scorer, pool_idx)
            if mode == "external":
                cal_arr = book.get(scorer, english_h_idx)
            else:
                cal_arr = book.get(scorer, cond["cal_idx"])
            cells = {
                a: {
                    "fpr": [], "violation_p": [],
                    "alarm": {m: [] for m in m_feasible},
                    "tv_hat": [], "tv_bound": [], "gamma_hat": [], "gamma_bound": [],
                }
                for a in alphas
            }
            for r in range(R):
                perm = perms[r]
                if mode == "external":
                    cal = _finite(cal_arr[english_masks[r]])
                    test = pool_arr[perm[:n_test]]
                    pilot_pool = perm[n_test:]
                else:
                    cal = _finite(cal_arr[perm[:n_cal]])
                    test = pool_arr[perm[n_cal:n_cal + n_test]]
                    pilot_pool = perm[n_cal + n_test:]
                pv = conformal_pvalues(cal, test)
                pilot_max = pool_arr[pilot_pool[:m_max]] if m_max else None
                for a in alphas:
                    fl = pv <= a
                    cells[a]["fpr"].append(float(fl.mean()) if fl.size else math.nan)
                    cells[a]["violation_p"].append(
                        violation_pvalue(int(fl.sum()), int(fl.size), int(cal.size), a)
                    )
                    for m in m_feasible:
                        res = pilot_fpr_ucb(cal, pool_arr[pilot_pool[:m]], a,
                                            alarm_level=alarm_level)
                        cells[a]["alarm"][m].append(bool(res["alarm"]))
                    if pilot_max is not None:
                        try:
                            b = predicted_fpr_bounds(cal, pilot_max, a)
                        except Exception:
                            b = None
                        # NaN on failure keeps the per-repeat arrays parallel to "fpr".
                        for key in ("tv_hat", "tv_bound", "gamma_hat", "gamma_bound"):
                            cells[a][key].append(
                                float(b[key]) if b is not None else math.nan
                            )
            cond_out[scorer] = {
                _akey(a): {
                    "condition": name, "kind": kind, "scorer": scorer, "alpha": float(a),
                    "n_cal": int(n_cal), "n_test": int(n_test), "n_pool": n_pool,
                    "pilot_m_max": int(m_max),
                    "fpr": cells[a]["fpr"],
                    "fpr_summary": _agg(cells[a]["fpr"]),
                    "violation_p": cells[a]["violation_p"],
                    "violated": bool(
                        np.median(np.asarray(cells[a]["violation_p"], dtype=float)) < 0.05
                    ) if cells[a]["violation_p"] else None,
                    "alarm_rate": {
                        str(m): (
                            float(np.mean(cells[a]["alarm"][m])) if m in cells[a]["alarm"]
                            and cells[a]["alarm"][m] else None
                        )
                        for m in pilot_sizes
                    },
                    "tv_hat_mean": _finite_mean(cells[a]["tv_hat"]),
                    "tv_bound_mean": _finite_mean(cells[a]["tv_bound"]),
                    "gamma_hat_mean": _finite_mean(cells[a]["gamma_hat"]),
                    "gamma_bound_mean": _finite_mean(cells[a]["gamma_bound"]),
                }
                for a in alphas
            }
            # Flat record trees for f10_diagnostics: one alarm record per condition x
            # alpha with parallel (m, alarm_rate) arrays; one bounds record with
            # parallel per-repeat (tv_bound, realized_fpr, ...) arrays.
            alarm_out[scorer] = {
                _akey(a): {
                    "condition": name, "kind": kind, "scorer": scorer, "alpha": float(a),
                    "alarm_level": float(alarm_level),
                    "m": list(m_feasible),
                    "alarm_rate": [
                        float(np.mean(cells[a]["alarm"][m])) if cells[a]["alarm"][m] else None
                        for m in m_feasible
                    ],
                }
                for a in alphas
                if m_feasible
            }
            bounds_out[scorer] = {
                _akey(a): {
                    "condition": name, "kind": kind, "scorer": scorer, "alpha": float(a),
                    "m": int(m_max),
                    "tv_bound": cells[a]["tv_bound"],
                    "tv_hat": cells[a]["tv_hat"],
                    "gamma_bound": cells[a]["gamma_bound"],
                    "gamma_hat": cells[a]["gamma_hat"],
                    "realized_fpr": list(cells[a]["fpr"]),
                }
                for a in alphas
                if cells[a]["tv_bound"]
            }
        out["conditions"][name] = cond_out
        if any(alarm_out.values()):
            out["pilot_alarms"][name] = alarm_out
        if any(bounds_out.values()):
            out["bounds"][name] = bounds_out
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_all(cfg: dict | None = None) -> dict:
    """Run the wave-2 study (W1-W4) and return one JSON-serialisable results dict.

    Keys of the returned dict: meta, w1_multilingual, w2_domains, w3_paraphrase,
    w4_diagnostics. Every block stores per-repeat arrays plus mean / 95%
    percentile-interval summaries. Deterministic given cfg['seed'].
    """
    t_start = time.time()
    c = dict(DEFAULT_CFG)
    c.update(cfg or {})
    seed = int(c["seed"])
    R = int(c["R"])
    alphas = [float(a) for a in c["alphas"]]
    device = str(c["device"])
    warnings_list: list[str] = []

    # ---- Stage 1: multilingual MAiDE-up; English entity-grouped halves ------ #
    print(f"[guard.wave2] loading multilingual MAiDE-up from {c['data_csv']}", flush=True)
    full = load_multilingual(c["data_csv"])
    langs = sorted(full["language"].unique())
    english = "English"
    if english not in langs:
        raise ValueError(f"no English rows in {c['data_csv']} (languages: {langs})")

    from guard.data import grouped_split

    df_en = full[full["language"] == english].reset_index(drop=True)
    train_half, eval_half = grouped_split(df_en, test_frac=0.5, seed=int(c["master_split_seed"]))
    print(
        f"[guard.wave2] {len(full)} rows / {len(langs)} languages; English train-half "
        f"{len(train_half)} rows / eval-half {len(eval_half)} rows", flush=True,
    )

    lang_h_texts: dict[str, list[str]] = {}
    lang_ai_texts: dict[str, list[str]] = {}
    for li, lang in enumerate(langs):
        src = eval_half if lang == english else full[full["language"] == lang]
        rng_l = np.random.default_rng((seed, 20, li))
        lang_h_texts[lang] = _cap_list(
            src.loc[src["label"] == 0, "text"].tolist(), c["lang_cap_human"], rng_l
        )
        lang_ai_texts[lang] = _cap_list(
            src.loc[src["label"] == 1, "text"].tolist(), c["lang_cap_ai"], rng_l
        )

    # ---- Stage 2: domain pools (hotel/local caches/duckdb fetch) ------------ #
    dom_h_texts: dict[str, list[str]] = {"hotel": lang_h_texts[english]}
    dom_ai_texts: dict[str, list[str]] = {"hotel": lang_ai_texts[english]}
    local_pools = (
        ("reviews", c["reviews_pool"]),
        ("abstracts", c["abstracts_parquet"]),
    )
    for di, (name, path) in enumerate(local_pools):
        pool = _local_pool(path, warnings_list, name)
        if pool is None:
            continue
        rng_d = np.random.default_rng((seed, 21, di))
        dom_h_texts[name] = _cap_list(
            pool.loc[pool["label"] == 0, "text"].tolist(), c["lang_cap_human"], rng_d
        )
        dom_ai_texts[name] = _cap_list(
            pool.loc[pool["label"] == 1, "text"].tolist(), c["lang_cap_ai"], rng_d
        )
    fetched = fetch_domain_pools(
        list(c["fetch_domains"]),
        cap_per_class=int(c["domain_fetch_cap"]),
        cache_dir=c["cache_dir"],
        allow_fetch=bool(c["fetch_if_missing"]),
    )
    for name in c["fetch_domains"]:
        if name not in fetched:
            warnings_list.append(f"domain {name!r} unavailable (fetch failed/disabled); skipped")
            continue
        pool = fetched[name]
        rng_d = np.random.default_rng((seed, 22, list(c["fetch_domains"]).index(name)))
        dom_h_texts[name] = _cap_list(
            pool.loc[pool["label"] == 0, "text"].tolist(), c["lang_cap_human"], rng_d
        )
        dom_ai_texts[name] = _cap_list(
            pool.loc[pool["label"] == 1, "text"].tolist(), c["lang_cap_ai"], rng_d
        )
    domains = [d for d in ("hotel", "reviews", "abstracts", "books", "news", "poetry")
               if d in dom_h_texts and len(dom_h_texts[d]) >= 20]
    print(f"[guard.wave2] domains in play: {domains}", flush=True)

    # ---- Stage 3: paraphrase texts (W3), seed ranges disjoint by role ------- #
    max_para = int(c["max_para"])
    k_var = int(c["k_cal_variants"])
    w3_skip_reason: str | None = None
    para_h: list[str] | None = None
    para_ai: list[str] | None = None
    cal_variants: list[list[str]] | None = None
    para_src_h: list[str] = []
    para_src_ai: list[str] = []
    if max_para < 4:
        w3_skip_reason = f"max_para={max_para} too small"
    else:
        pp = _make_paraphraser(device, warnings_list)
        if pp is None:
            w3_skip_reason = warnings_list[-1] if warnings_list else "paraphraser unavailable"
        else:
            para_src_h = lang_h_texts[english][:max_para]
            para_src_ai = lang_ai_texts[english][: int(c["para_ai_cap"])]
            n_p = len(para_src_h)
            pcache = c["para_cache"]
            para_h = _paraphrase_texts(
                pp, para_src_h, [10_000 + i for i in range(n_p)],
                warnings_list, "humans (test)", cache_path=pcache,
            )
            para_ai = _paraphrase_texts(
                pp, para_src_ai, [30_000 + j for j in range(len(para_src_ai))],
                warnings_list, "AI (evasion)", cache_path=pcache,
            )
            cal_variants = [
                _paraphrase_texts(
                    pp, para_src_h, [20_000 + k_var * i + j for i in range(n_p)],
                    warnings_list, f"cal variants {j + 1}/{k_var}", cache_path=pcache,
                )
                for j in range(k_var)
            ]
            n_fail = int(getattr(pp, "n_failures", 0) or 0)
            if n_fail:
                warnings_list.append(f"paraphraser reported {n_fail} generation failures "
                                     f"(originals kept)")

    # ---- Stage 4: score every unique text exactly once ---------------------- #
    requested = list(dict.fromkeys(c["scorers"]))
    if c["with_supervised"]:
        for extra in ("supervised", "offshelf"):
            if extra not in requested:
                requested.append(extra)
    base_names = [s for s in requested if s in GPT2_STATS or s == "tfidf"]
    extra_names = [s for s in requested if s not in base_names]

    ordered: list[Sequence[str]] = []
    ordered.extend(lang_h_texts[lang] for lang in langs)
    ordered.extend(lang_ai_texts[lang] for lang in langs)
    ordered.extend(dom_h_texts[d] for d in domains)
    ordered.extend(dom_ai_texts[d] for d in domains)
    if para_h is not None and para_ai is not None and cal_variants is not None:
        ordered.append(para_h)
        ordered.append(para_ai)
        ordered.extend(cal_variants)
    unique_texts = list(dict.fromkeys(t for block in ordered for t in block))
    print(
        f"[guard.wave2] scoring {len(unique_texts)} unique texts with {requested} "
        f"(cache: {c['cache_path']}, device: {device})", flush=True,
    )
    tfidf = None
    if "tfidf" in base_names:
        tfidf = TfidfScorer().fit(train_half["text"].tolist(), train_half["label"].tolist())
    frame = compute_scores(
        unique_texts, base_names, c["cache_path"],
        gpt2=GPT2Scores(device=device), tfidf=tfidf,
    )
    score_arrays = {s: frame[s].to_numpy(dtype=float) for s in base_names}
    if extra_names:
        score_arrays.update(
            _supervised_columns(
                extra_names, unique_texts,
                train_half["text"].tolist(), train_half["label"].to_numpy(dtype=int),
                device, warnings_list,
            )
        )
    book = ScoreBook(unique_texts, score_arrays)
    scorers_used = [s for s in requested if s in book.scores]
    matrix_scorers = [s for s in _MATRIX_SCORERS if s in book.scores]
    print(f"[guard.wave2] scorers in play: {scorers_used}", flush=True)

    lang_h_idx = {lang: book.idx(lang_h_texts[lang]) for lang in langs}
    lang_ai_idx = {lang: book.idx(lang_ai_texts[lang]) for lang in langs}
    dom_h_idx = {d: book.idx(dom_h_texts[d]) for d in domains}
    dom_ai_idx = {d: book.idx(dom_ai_texts[d]) for d in domains}

    # ---- Stage 5: resampling plans ------------------------------------------ #
    lang_masks = {
        lang: _split_masks(int(lang_h_idx[lang].size), R, np.random.default_rng((seed, 30, li)))
        for li, lang in enumerate(langs)
    }
    dom_masks: dict[str, list[np.ndarray]] = {}
    for di, d in enumerate(domains):
        if d == "hotel":
            dom_masks[d] = lang_masks[english]  # identical pool -> identical splits
        else:
            dom_masks[d] = _split_masks(
                int(dom_h_idx[d].size), R, np.random.default_rng((seed, 40, di))
            )

    results: dict[str, Any] = {}

    def _stage(name: str, fn: Callable[[], dict]) -> dict:
        t0 = time.time()
        print(f"[guard.wave2] {name} ...", flush=True)
        res = fn()
        print(f"[guard.wave2] {name} done in {time.time() - t0:.1f}s", flush=True)
        return res

    # ---- W1: multilingual ---------------------------------------------------- #
    def _w1() -> dict:
        per_language: dict[str, Any] = {}
        for lang in langs:
            validity, power = _validity_and_power(
                book, scorers_used, lang_h_idx[lang], lang_ai_idx[lang],
                lang_masks[lang], alphas,
            )
            per_language[lang] = {
                "n_human": int(lang_h_idx[lang].size),
                "n_ai": int(lang_ai_idx[lang].size),
                "n_cal": int(lang_h_idx[lang].size) // 2,
                "in_language": validity,
                "power": power,
            }
        others = [lang for lang in langs if lang != english]
        cross_from_english = _transfer_row(
            book, scorers_used, english, lang_h_idx[english], lang_masks[english],
            {lang: lang_h_idx[lang] for lang in others},
            {lang: lang_masks[lang] for lang in others}, alphas,
        )
        cross_matrix = _transfer_matrix(
            book, matrix_scorers, langs, lang_h_idx, lang_masks, alphas
        )
        return {
            "languages": langs,
            "english": english,
            "caps": {"human": c["lang_cap_human"], "ai": c["lang_cap_ai"]},
            "per_language": per_language,
            "cross_from_english": cross_from_english,
            "cross_matrix": cross_matrix,
        }

    results["w1_multilingual"] = _stage("W1 multilingual", _w1)

    # ---- W2: domains ---------------------------------------------------------- #
    def _w2() -> dict:
        per_domain: dict[str, Any] = {}
        for d in domains:
            validity, power = _validity_and_power(
                book, scorers_used, dom_h_idx[d], dom_ai_idx[d], dom_masks[d], alphas
            )
            per_domain[d] = {
                "n_human": int(dom_h_idx[d].size),
                "n_ai": int(dom_ai_idx[d].size),
                "n_cal": int(dom_h_idx[d].size) // 2,
                "in_domain": validity,
                "power": power,
            }
        others = [d for d in domains if d != "hotel"]
        cross_from_hotel = _transfer_row(
            book, scorers_used, "hotel", dom_h_idx["hotel"], dom_masks["hotel"],
            {d: dom_h_idx[d] for d in others},
            {d: dom_masks[d] for d in others}, alphas,
        )
        cross_matrix = _transfer_matrix(
            book, matrix_scorers, domains, dom_h_idx, dom_masks, alphas
        )
        return {
            "domains": domains,
            "per_domain": per_domain,
            "cross_from_hotel": cross_from_hotel,
            "cross_matrix": cross_matrix,
        }

    results["w2_domains"] = _stage("W2 domains", _w2)

    # ---- W3: paraphrase ------------------------------------------------------- #
    para_clean_idx = para_test_idx = variants_idx = None
    if para_h is not None and para_ai is not None and cal_variants is not None:
        n_p = len(para_src_h)
        para_clean_idx = book.idx(para_src_h)
        para_test_idx = book.idx(para_h)
        variants_idx = np.stack([book.idx(col) for col in cal_variants], axis=1)  # (n_p, k)
        w3_masks = _split_masks(n_p, R, np.random.default_rng((seed, 50)))
        results["w3_paraphrase"] = _stage(
            "W3 paraphrase attack",
            lambda: _run_w3(
                book, scorers_used, para_clean_idx, para_test_idx, variants_idx,
                book.idx(para_src_ai), book.idx(para_ai), w3_masks, alphas,
            ),
        )
    else:
        results["w3_paraphrase"] = {"skipped": True, "reason": w3_skip_reason}
        print(f"[guard.wave2] W3 skipped: {w3_skip_reason}", flush=True)

    # ---- W4: pre-deployment diagnostics --------------------------------------- #
    conditions: list[dict] = [
        {
            "name": "control_english", "kind": "control", "mode": "threeway",
            "pool_idx": lang_h_idx[english], "cal_idx": lang_h_idx[english],
        }
    ]
    conditions.extend(
        {
            "name": f"lang_{lang}", "kind": "language", "mode": "external",
            "pool_idx": lang_h_idx[lang],
        }
        for lang in langs if lang != english
    )
    conditions.extend(
        {
            "name": f"domain_{d}", "kind": "domain", "mode": "external",
            "pool_idx": dom_h_idx[d],
        }
        for d in domains if d != "hotel"
    )
    if para_test_idx is not None and para_clean_idx is not None:
        conditions.append(
            {
                "name": "paraphrased_humans", "kind": "edit", "mode": "threeway",
                "pool_idx": para_test_idx, "cal_idx": para_clean_idx,
            }
        )
    results["w4_diagnostics"] = _stage(
        "W4 diagnostics",
        lambda: _run_w4(
            book, scorers_used, lang_h_idx[english], lang_masks[english],
            conditions, alphas, list(c["pilot_sizes"]), float(c["alarm_level"]), seed,
        ),
    )

    # ---- meta ----------------------------------------------------------------- #
    results["meta"] = {
        "cfg": {k: (str(v) if isinstance(v, Path) else v) for k, v in c.items()},
        "scorers_used": scorers_used,
        "matrix_scorers": matrix_scorers,
        "counts": {
            "languages": {lang: {"human": len(lang_h_texts[lang]), "ai": len(lang_ai_texts[lang])}
                          for lang in langs},
            "domains": {d: {"human": len(dom_h_texts[d]), "ai": len(dom_ai_texts[d])}
                        for d in domains},
            "paraphrased_humans": len(para_h) if para_h is not None else 0,
            "paraphrased_ai": len(para_ai) if para_ai is not None else 0,
            "cal_variant_texts": (
                int(variants_idx.size) if variants_idx is not None else 0
            ),
            "unique_texts_scored": len(unique_texts),
            "w4_conditions": [cond["name"] for cond in conditions],
        },
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "warnings": warnings_list,
        "started_utc": datetime.fromtimestamp(t_start, tz=timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - t_start, 2),
    }
    print(f"[guard.wave2] study complete in {results['meta']['runtime_seconds']}s", flush=True)
    ordered_out = {
        k: results[k]
        for k in ("meta", "w1_multilingual", "w2_domains", "w3_paraphrase", "w4_diagnostics")
    }
    return _jsonable(ordered_out)
