"""GUARD study engine: experiments E1-E7 from METHODOLOGY.md.

Scientific purpose
------------------
Split-conformal calibration on human text gives a distribution-free certificate
P(flag | human) <= alpha whose validity depends ONLY on exchangeability of human
calibration and human test text (the certificate asymmetry). This module maps,
empirically and with exact finite-sample reference distributions, where that
certificate survives deployment and where it breaks:

  E1 validity in-distribution   - empirical FPR distribution over R repeated
                                  calibration/test partitions vs the exact Beta
                                  (infinite-test) and Beta-Binomial (finite-test)
                                  envelopes; random-row and grouped-by-hotel variants.
  E2 generator shift            - calibrate on MAiDE hotel humans, measure power per
                                  RAID-reviews generator and FPR on RAID-reviews humans
                                  (mild human shift). Skipped cleanly if the pool cache
                                  is absent.
  E3 human domain shift         - same calibration, FPR on RAID abstracts humans.
  E4 human edit attack          - HEADLINE: calibrate on clean EVAL humans, test on
                                  attacked versions of the SAME humans (function-word /
                                  synonym edits at increasing rates). Because the test
                                  texts are edited copies of the same human population,
                                  any certificate violation isolates the edit effect
                                  from population shift.
  E5 leakage                    - random vs entity-grouped calibration partitions
                                  (both FPR distributions stored; delta computed at
                                  analysis time).
  E6 Mondrian by length         - per-length-bin FPR under marginal vs Mondrian
                                  calibration at alpha in {0.05, 0.01}.
  E7 power frontier             - TPR per scorer per certified alpha on clean and
                                  synonym-attacked (evasion) AI text.

Design principle: SCORE ONCE, RESAMPLE MANY. Every unique text is scored exactly once
(GPT-2 statistics go through the parquet cache owned by guard.scores); the R repeated
calibration/test draws are pure index resampling over precomputed score arrays, so
R=200 repeats add no scoring cost.

Leakage protocol: an entity-grouped master split by hotel (seed fixed, default 0)
divides MAiDE English into a TRAIN-half and an EVAL-half. The supervised TF-IDF scorer
is fit on the TRAIN-half only; ALL conformal calibration and test draws use EVAL-half
texts, so the supervised scorer never sees evaluation entities.

Expected sibling-module API (written in parallel; thin signature-adaptive shims below
tolerate small naming differences):
  guard.scores.compute_scores(texts, scorers=[...], cache_path=...) ->
      dict[str, array-like] or DataFrame of per-text scores, higher = more AI-like.
  guard.scores.TfidfScorer().fit(texts, labels) / .score(texts) -> array-like.
  guard.data.load_maide_english(csv_path) -> DataFrame with text/label/hotel columns
      (a local loader replicates this if absent).
  guard.attacks: per-text attack functions for kinds "fw" (function-word perturbation)
      and "synonym" (synonym substitution), signature fn(text, rate, rng) -> str, or a
      dispatcher apply_attack(text, kind=..., rate=..., rng=...).

Everything is deterministic given cfg["seed"]; all randomness flows through
numpy.random.default_rng seeded with (seed, stream_id) tuples. No experimental result
is hard-coded; run_all returns one JSON-serialisable dict that downstream analysis and
figures regenerate from.
"""
from __future__ import annotations

import inspect
import math
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy import stats as scipy_stats

from guard.conformal import (
    beta_envelope,
    evaluate_certificate,  # wraps violation_pvalue: exact Beta-Binomial tail per draw
    flag,
    length_bins,
    mondrian_flag,
)

_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CFG: dict[str, Any] = {
    "data_csv": str(_ROOT / "data" / "all_data.csv"),
    "reviews_pool": str(_ROOT / "results" / "cache" / "raid_reviews_pool.parquet"),
    "abstracts_parquet": str(_ROOT / "results" / "cache" / "raid_abstracts.parquet"),
    "cache_path": str(_ROOT / "results" / "cache" / "scores.parquet"),
    "scorers": ["loglik", "logrank", "entropy", "fast_detectgpt", "tfidf"],
    "alphas": [0.005, 0.01, 0.02, 0.05, 0.10],
    "R": 200,
    "cal_frac": 0.5,
    "seed": 0,
    "edit_rates": [0.1, 0.3, 0.5],
    "evasion_kind": "synonym",
    "evasion_rate": 0.3,
    "master_split_seed": 0,  # entity split is a constant of the study (METHODOLOGY 4)
    "max_gpt2_texts": None,  # cap per base text set, smoke runs only
}

_ATTACK_KINDS = ("fw", "synonym", "cleanup")
_E6_ALPHAS = (0.05, 0.01)
_LENGTH_BIN_EDGES = (0, 30, 60, 120, 10_000)  # mirrors guard.conformal.length_bins default


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #

def _akey(alpha: float) -> str:
    """Stable string key for an alpha level ('0.005', '0.05', '0.1', ...)."""
    return f"{float(alpha):g}"


def _rkey(rate: float) -> str:
    return f"{float(rate):g}"


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


def _mode(values: Sequence[int]) -> int:
    return int(Counter(int(v) for v in values).most_common(1)[0][0])


def _betabinom_envelope(
    n_cal: int, n_test: int, alpha: float, q: tuple[float, float] = (0.025, 0.975)
) -> dict[str, float | int]:
    """Exact finite-test envelope of the realised FPR.

    With n_cal calibration humans and m = n_test test humans, the false-flag count under
    exchangeability is Beta-Binomial(m, k, n_cal + 1 - k), k = floor(alpha (n_cal + 1)).
    The realised FPR is that count divided by m; beta_envelope alone is the m -> inf
    limit, so both are reported (METHODOLOGY: verify the whole distribution).
    """
    n_cal, n_test = int(n_cal), int(n_test)
    k = int(math.floor(alpha * (n_cal + 1)))
    if k < 1 or n_test < 1:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "k": k, "n_cal": n_cal, "n_test": n_test}
    rv = scipy_stats.betabinom(n_test, k, n_cal + 1 - k)
    return {
        "mean": float(rv.mean()) / n_test,
        "lo": float(rv.ppf(q[0])) / n_test,
        "hi": float(rv.ppf(q[1])) / n_test,
        "k": k,
        "n_cal": n_cal,
        "n_test": n_test,
    }


def _envelopes(n_cal: int, n_test: int, alphas: Sequence[float]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for a in alphas:
        mean, lo, hi = beta_envelope(int(n_cal), float(a), q=(0.025, 0.975))
        out[_akey(a)] = {
            "beta_infinite_test": {"mean": mean, "lo": lo, "hi": hi, "n_cal": int(n_cal)},
            "betabinom_finite_test": _betabinom_envelope(n_cal, n_test, a),
        }
    return out


def _jsonable(obj: Any) -> Any:
    """Recursively convert to plain JSON types; non-finite floats become None."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _params_of(fn: Callable) -> dict | None:
    try:
        return dict(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return None


class ScoreBook:
    """Immutable registry of unique texts and their precomputed score arrays.

    Texts are scored exactly once; experiments address scores via integer index
    arrays, making repeated resampling free (the design principle of this module).
    """

    def __init__(self, texts: Sequence[str], scores: dict[str, np.ndarray]):
        self.texts: list[str] = list(texts)
        self.scores: dict[str, np.ndarray] = {}
        for name, arr in scores.items():
            a = np.asarray(arr, dtype=float)
            if a.shape != (len(self.texts),):
                raise ValueError(
                    f"score array for {name!r} has shape {a.shape}, "
                    f"expected ({len(self.texts)},)"
                )
            self.scores[name] = a
        self._pos = {t: i for i, t in enumerate(self.texts)}

    @property
    def scorers(self) -> list[str]:
        return list(self.scores)

    def idx(self, texts: Sequence[str]) -> np.ndarray:
        """Index array of the given texts inside the book (texts must be registered)."""
        return np.fromiter((self._pos[t] for t in texts), dtype=np.int64, count=len(texts))

    def get(self, scorer: str, idx: np.ndarray) -> np.ndarray:
        return self.scores[scorer][idx]


# --------------------------------------------------------------------------- #
# Data loading (guard.data preferred; local fallbacks replicate the documented facts)
# --------------------------------------------------------------------------- #

def _normalize_maide(df: pd.DataFrame) -> pd.DataFrame | None:
    """Map a sibling loader's output onto the canonical text/label/hotel frame."""
    lower = {str(c).strip().lower(): c for c in df.columns}

    def pick(*names: str) -> str | None:
        for n in names:
            if n in lower:
                return lower[n]
        return None

    text_c = pick("text", "review", "review_text", "full_text")
    label_c = pick("label", "source", "y", "is_ai")
    hotel_c = pick("hotel", "hotel name", "hotel_name", "entity", "group")
    if text_c is None or label_c is None or hotel_c is None:
        return None
    out = pd.DataFrame(
        {
            "text": df[text_c].astype(str),
            "label": pd.to_numeric(df[label_c]).astype(int),
            "hotel": df[hotel_c].astype(str),
        }
    )
    out = out[out["text"].str.strip() != ""].reset_index(drop=True)
    return out


def _load_maide_local(csv_path: str | Path) -> pd.DataFrame:
    """Local MAiDE-up English loader: text = upside + downside, label = source (1 = AI)."""
    df = pd.read_csv(csv_path)
    lang = df["Review_Language"].astype(str).str.strip().str.lower()
    df = df[lang.isin({"english", "en"}) | lang.str.startswith("english")]
    up = df["Upside_Review"].fillna("").astype(str).str.strip()
    down = df["Downside_Review"].fillna("").astype(str).str.strip()
    out = pd.DataFrame(
        {
            "text": (up + " " + down).str.strip(),
            "label": pd.to_numeric(df["source"]).astype(int),
            "hotel": df["Hotel Name"].astype(str),
        }
    )
    out = out[out["text"] != ""].reset_index(drop=True)
    return out


def _load_maide(csv_path: str | Path, warnings_list: list[str]) -> pd.DataFrame:
    try:
        from guard import data as guard_data
    except ImportError:
        guard_data = None
    if guard_data is not None:
        for name in ("load_maide_english", "load_maide", "load_english_reviews"):
            fn = getattr(guard_data, name, None)
            if not callable(fn):
                continue
            try:
                raw = fn(csv_path)
            except Exception as exc:  # fall back rather than abort the study
                warnings_list.append(f"guard.data.{name} failed ({exc!r}); using local loader")
                continue
            norm = _normalize_maide(raw) if isinstance(raw, pd.DataFrame) else None
            if norm is not None:
                return norm
            warnings_list.append(f"guard.data.{name} output not interpretable; using local loader")
    return _load_maide_local(csv_path)


def _grouped_halves_local(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    hotels = np.sort(df["hotel"].unique())
    perm = np.random.default_rng(seed).permutation(hotels.size)
    train_hotels = set(hotels[perm[: hotels.size // 2]])
    mask = df["hotel"].isin(train_hotels).to_numpy()
    return df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)


def _master_split(
    df: pd.DataFrame, seed: int, warnings_list: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entity-grouped TRAIN-half / EVAL-half split by hotel (grouped_split, fixed seed)."""
    try:
        from guard import data as guard_data
    except ImportError:
        guard_data = None
    fn = getattr(guard_data, "grouped_split", None) if guard_data is not None else None
    if callable(fn):
        params = _params_of(fn)
        kwargs: dict[str, Any] = {}
        if params is not None:
            for nm, val in (
                ("group_col", "hotel"),
                ("groups", df["hotel"].to_numpy()),
                ("frac", 0.5),
                ("train_frac", 0.5),
                ("seed", seed),
                ("random_state", seed),
            ):
                if nm in params and nm not in kwargs:
                    kwargs[nm] = val
        try:
            out = fn(df, **kwargs)
            a, b = out
            if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
                return a.reset_index(drop=True), b.reset_index(drop=True)
            a = np.asarray(a)
            b = np.asarray(b)
            if a.dtype == bool:
                return df[a].reset_index(drop=True), df[b].reset_index(drop=True)
            return df.iloc[a].reset_index(drop=True), df.iloc[b].reset_index(drop=True)
        except Exception as exc:
            warnings_list.append(
                f"guard.data.grouped_split unusable ({exc!r}); using local grouped split"
            )
    return _grouped_halves_local(df, seed)


def _human_mask(df: pd.DataFrame) -> np.ndarray:
    """Human rows of a RAID-style frame (model == 'human', else label == 0/'human')."""
    if "model" in df.columns:
        m = df["model"].astype(str).str.strip().str.lower().eq("human")
        if m.any():
            return m.to_numpy()
    lab = df["label"]
    if lab.dtype == object:
        return lab.astype(str).str.strip().str.lower().isin({"human", "0"}).to_numpy()
    return (pd.to_numeric(lab) == 0).to_numpy()


def _filter_clean_attack(df: pd.DataFrame) -> pd.DataFrame:
    """Keep unattacked rows when the cache distinguishes them; otherwise keep all."""
    if "attack" not in df.columns:
        return df
    vals = df["attack"].fillna("none").astype(str).str.strip().str.lower()
    clean = vals.isin({"none", "", "nan", "no_attack"})
    return df[clean] if clean.any() else df


def _cap_indices(n: int, cap: int | None, rng: np.random.Generator) -> np.ndarray:
    """Deterministic subsample of range(n) (sorted), used by max_gpt2_texts smoke caps."""
    if cap is None or n <= int(cap):
        return np.arange(n)
    return np.sort(rng.choice(n, size=int(cap), replace=False))


def _load_raid_frame(path: str | Path | None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_parquet(p)


# --------------------------------------------------------------------------- #
# Attacks (guard.attacks; per-text, deterministic via the supplied generator)
# --------------------------------------------------------------------------- #

_ATTACK_FN_NAMES: dict[str, tuple[str, ...]] = {
    "fw": (
        "fw_perturb",  # guard.attacks' actual API
        "function_word_attack",
        "fw_attack",
        "perturb_function_words",
        "function_word_perturbation",
    ),
    "synonym": (
        "synonym_perturb",  # guard.attacks' actual API
        "synonym_attack",
        "synonym_substitution",
        "substitute_synonyms",
        "synonym_perturbation",
    ),
    "cleanup": (
        "cleanup_perturb",  # fluency-INCREASING editor (spell/punct normalisation)
    ),
}
_ATTACK_DISPATCH_NAMES = ("apply_attack", "attack", "perturb_text", "perturb")


def _flex_attack_call(
    fn: Callable, text: str, kind: str | None, rate: float, rng: np.random.Generator
) -> str:
    """Call an attack function tolerating small signature differences."""
    params = _params_of(fn)
    if params is None:
        return fn(text, rate)
    kwargs: dict[str, Any] = {}

    def put(value: Any, *names: str) -> bool:
        for nm in names:
            if nm in params:
                kwargs[nm] = value
                return True
        return False

    if kind is not None:
        put(kind, "kind", "attack", "method", "name")
    has_rate = put(rate, "rate", "edit_rate", "p", "frac", "fraction", "strength")
    if not put(rng, "rng", "random_state", "generator"):
        put(int(rng.integers(0, 2**31 - 1)), "seed")
    if not has_rate:
        try:
            return fn(text, rate, **kwargs)
        except TypeError:
            pass
    return fn(text, **kwargs)


def _resolve_attack(kind: str) -> Callable[[str, float, np.random.Generator], str]:
    """Resolve a per-text attack callable for kind in {'fw', 'synonym'}."""
    from guard import attacks as guard_attacks

    for name in _ATTACK_FN_NAMES[kind]:
        fn = getattr(guard_attacks, name, None)
        if callable(fn):
            return lambda text, rate, rng, _fn=fn: _flex_attack_call(_fn, text, None, rate, rng)
    for name in _ATTACK_DISPATCH_NAMES:
        fn = getattr(guard_attacks, name, None)
        if callable(fn):
            return lambda text, rate, rng, _fn=fn, _k=kind: _flex_attack_call(
                _fn, text, _k, rate, rng
            )
    raise AttributeError(
        f"guard.attacks exposes no attack for kind={kind!r}; "
        f"tried {_ATTACK_FN_NAMES[kind] + _ATTACK_DISPATCH_NAMES}"
    )


def _attack_texts(
    fn: Callable[[str, float, np.random.Generator], str],
    texts: Sequence[str],
    rate: float,
    rng: np.random.Generator,
    warnings_list: list[str],
    tag: str,
) -> list[str]:
    out: list[str] = []
    failures = 0
    for t in texts:
        try:
            res = fn(t, float(rate), rng)
            out.append(res if isinstance(res, str) else str(res))
        except Exception:
            failures += 1
            out.append(t)  # conservative: an un-attackable text stays clean
    if failures:
        warnings_list.append(f"attack {tag}: {failures}/{len(texts)} texts left unedited (errors)")
    return out


# --------------------------------------------------------------------------- #
# Scoring (guard.scores; one pass over the union of all unique texts)
# --------------------------------------------------------------------------- #

def _call_compute_scores(
    fn: Callable, texts: Sequence[str], scorer_names: Sequence[str], cache_path: str | None
) -> Any:
    params = _params_of(fn)
    if params is None:
        return fn(list(texts), list(scorer_names))
    has_var = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs: dict[str, Any] = {}

    def put(value: Any, *names: str) -> None:
        for nm in names:
            if nm in params:
                kwargs[nm] = value
                return
        if has_var:
            kwargs[names[0]] = value

    put(list(scorer_names), "scorers", "scorer_names", "names", "methods")
    if cache_path:
        put(str(cache_path), "cache_path", "cache", "cache_file")
    return fn(list(texts), **kwargs)


def _normalize_score_output(res: Any, n: int) -> dict[str, np.ndarray]:
    if isinstance(res, pd.DataFrame):
        cand = {str(c): res[c].to_numpy() for c in res.columns}
    elif isinstance(res, dict):
        cand = {str(k): np.asarray(v) for k, v in res.items()}
    else:
        raise TypeError(f"compute_scores returned {type(res).__name__}; expected dict or DataFrame")
    out: dict[str, np.ndarray] = {}
    for name, arr in cand.items():
        a = np.asarray(arr)
        if a.ndim == 1 and a.size == n and np.issubdtype(a.dtype, np.number):
            out[name] = a.astype(float)
    return out


def _fit_tfidf(train_texts: Sequence[str], train_labels: Sequence[int]) -> Any:
    from guard import scores as guard_scores

    cls = None
    for name in ("TfidfScorer", "TfIdfScorer", "TFIDFScorer"):
        cls = getattr(guard_scores, name, None)
        if cls is not None:
            break
    if cls is None:
        raise AttributeError("guard.scores exposes no TfidfScorer class")
    model = cls()
    fitted = model.fit(list(train_texts), np.asarray(train_labels, dtype=int))
    return fitted if fitted is not None else model


def _tfidf_score(model: Any, texts: Sequence[str]) -> np.ndarray:
    for name in ("score", "scores", "score_texts", "predict_scores", "decision_function"):
        fn = getattr(model, name, None)
        if callable(fn):
            return np.asarray(fn(list(texts)), dtype=float)
    if callable(model):
        return np.asarray(model(list(texts)), dtype=float)
    raise AttributeError("fitted TfidfScorer exposes no scoring method")


def _score_all(
    texts: Sequence[str],
    requested: Sequence[str],
    cache_path: str | None,
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    warnings_list: list[str],
) -> dict[str, np.ndarray]:
    from guard import scores as guard_scores

    out: dict[str, np.ndarray] = {}
    unsup = [s for s in requested if s != "tfidf"]
    if unsup:
        res = _call_compute_scores(guard_scores.compute_scores, texts, unsup, cache_path)
        norm = _normalize_score_output(res, len(texts))
        for s in unsup:
            if s in norm:
                out[s] = norm[s]
            else:
                warnings_list.append(f"scorer {s!r} missing from compute_scores output; dropped")
    if "tfidf" in requested:
        try:
            model = _fit_tfidf(train_texts, train_labels)
            out["tfidf"] = _tfidf_score(model, texts)
        except Exception as exc:
            warnings_list.append(f"tfidf scorer unavailable ({exc!r}); dropped")
    if not out:
        raise RuntimeError("no usable scorers; cannot run the study")
    return out


# --------------------------------------------------------------------------- #
# Resampling plans (shared across experiments so comparisons are paired)
# --------------------------------------------------------------------------- #

def _partition_masks(n: int, n_cal: int, R: int, rng: np.random.Generator) -> list[np.ndarray]:
    """R boolean masks: True = calibration member of a cal/test partition of range(n)."""
    masks = []
    for _ in range(R):
        sel = np.zeros(n, dtype=bool)
        sel[rng.permutation(n)[:n_cal]] = True
        masks.append(sel)
    return masks


def _grouped_partition_masks(
    groups: np.ndarray, R: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """R boolean masks splitting by entity: half the hotels calibrate, the rest test."""
    uniq = np.unique(groups)
    masks = []
    for _ in range(R):
        cal_groups = set(uniq[rng.permutation(uniq.size)[: uniq.size // 2]])
        masks.append(np.fromiter((g in cal_groups for g in groups), dtype=bool, count=groups.size))
    return masks


def _calibration_draws(
    n: int, n_cal: int, R: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """R index draws (size n_cal, without replacement) from the EVAL-human score array."""
    return [np.sort(rng.choice(n, size=n_cal, replace=False)) for _ in range(R)]


# --------------------------------------------------------------------------- #
# E1 (+ partitions reused by E5/E6): validity in-distribution
# --------------------------------------------------------------------------- #

def _run_e1(
    book: ScoreBook,
    scorers: Sequence[str],
    h_idx: np.ndarray,
    partitions: dict[str, list[np.ndarray]],
    alphas: Sequence[float],
) -> dict:
    """Empirical FPR distribution per partition variant vs exact Beta / Beta-Binomial laws."""
    out: dict[str, Any] = {"n_eval_humans": int(h_idx.size)}
    for variant, masks in partitions.items():
        n_cal_list = [int(m.sum()) for m in masks]
        n_test_list = [int(m.size - m.sum()) for m in masks]
        nc, nt = _mode(n_cal_list), _mode(n_test_list)
        per_scorer: dict[str, dict] = {}
        for scorer in scorers:
            sarr = book.get(scorer, h_idx)
            cells = {a: {"fpr": [], "violation_p": []} for a in alphas}
            for mask in masks:
                cal, test = sarr[mask], sarr[~mask]
                for a in alphas:
                    res = evaluate_certificate(cal, test, None, a)
                    cells[a]["fpr"].append(res.fpr)
                    cells[a]["violation_p"].append(res.violation_p)
            # Self-describing leaf: alpha/scorer/split plus per-repeat arrays under the
            # canonical keys guard.viz.iter_records consumes ("fpr", "violation_p").
            per_scorer[scorer] = {
                _akey(a): {
                    "alpha": float(a),
                    "scorer": scorer,
                    "split": variant,
                    "n_cal": nc,
                    "n_test_human": nt,
                    "fpr": cells[a]["fpr"],
                    "fpr_summary": _agg(cells[a]["fpr"]),
                    "violation_p": cells[a]["violation_p"],
                    "violation_p_mean": _agg(cells[a]["violation_p"])["mean"],
                }
                for a in alphas
            }
        out[variant] = {
            "n_cal": nc,
            "n_test": nt,
            "n_cal_min": int(min(n_cal_list)),
            "n_cal_max": int(max(n_cal_list)),
            "envelope": _envelopes(nc, nt, alphas),
            "per_scorer": per_scorer,
        }
    return out


# --------------------------------------------------------------------------- #
# E2: generator shift (power per RAID generator; FPR on RAID-reviews humans)
# --------------------------------------------------------------------------- #

def _run_e2(
    book: ScoreBook,
    scorers: Sequence[str],
    h_idx: np.ndarray,
    cal_draws: list[np.ndarray],
    pool_h_idx: np.ndarray | None,
    pool_gen_idx: dict[str, np.ndarray] | None,
    alphas: Sequence[float],
) -> dict:
    if pool_h_idx is None or pool_gen_idx is None:
        return {"skipped": True, "reason": "reviews pool cache not found"}
    n_cal = int(cal_draws[0].size)
    out: dict[str, Any] = {
        "skipped": False,
        "n_cal": n_cal,
        "n_pool_humans": int(pool_h_idx.size),
        "generators": {g: int(v.size) for g, v in pool_gen_idx.items()},
        "envelope": _envelopes(n_cal, int(pool_h_idx.size), alphas),
        "fpr_pool_humans": {},
        "power_per_generator": {},
    }
    for scorer in scorers:
        sarr = book.get(scorer, h_idx)
        ph = book.get(scorer, pool_h_idx)
        gen_scores = {g: book.get(scorer, gi) for g, gi in pool_gen_idx.items()}
        f_cells = {a: {"fpr": [], "violation_p": []} for a in alphas}
        p_cells = {g: {a: [] for a in alphas} for g in gen_scores}
        for draw in cal_draws:
            cal = sarr[draw]
            for a in alphas:
                res = evaluate_certificate(cal, ph, None, a)
                f_cells[a]["fpr"].append(res.fpr)
                f_cells[a]["violation_p"].append(res.violation_p)
                for g, gs in gen_scores.items():
                    p_cells[g][a].append(float(flag(cal, gs, a).mean()))
        out["fpr_pool_humans"][scorer] = {
            _akey(a): {
                "alpha": float(a),
                "scorer": scorer,
                "population": "human",
                "n_cal": n_cal,
                "n_test_human": int(ph.size),
                "fpr": f_cells[a]["fpr"],
                "fpr_summary": _agg(f_cells[a]["fpr"]),
                "violation_p": f_cells[a]["violation_p"],
                "violation_p_mean": _agg(f_cells[a]["violation_p"])["mean"],
            }
            for a in alphas
        }
        out["power_per_generator"][scorer] = {
            g: {
                _akey(a): {
                    "alpha": float(a),
                    "scorer": scorer,
                    "generator": g,
                    "tpr": p_cells[g][a],
                    "tpr_summary": _agg(p_cells[g][a]),
                }
                for a in alphas
            }
            for g in gen_scores
        }
    return out


# --------------------------------------------------------------------------- #
# E3: human domain shift (hotel humans -> abstracts humans)
# --------------------------------------------------------------------------- #

def _run_e3(
    book: ScoreBook,
    scorers: Sequence[str],
    h_idx: np.ndarray,
    cal_draws: list[np.ndarray],
    abs_h_idx: np.ndarray | None,
    alphas: Sequence[float],
) -> dict:
    if abs_h_idx is None:
        return {"skipped": True, "reason": "abstracts cache not found"}
    n_cal = int(cal_draws[0].size)
    out: dict[str, Any] = {
        "skipped": False,
        "n_cal": n_cal,
        "n_abstract_humans": int(abs_h_idx.size),
        "envelope": _envelopes(n_cal, int(abs_h_idx.size), alphas),
        "per_scorer": {},
    }
    for scorer in scorers:
        sarr = book.get(scorer, h_idx)
        ah = book.get(scorer, abs_h_idx)
        cells = {a: {"fpr": [], "violation_p": []} for a in alphas}
        for draw in cal_draws:
            cal = sarr[draw]
            for a in alphas:
                res = evaluate_certificate(cal, ah, None, a)
                cells[a]["fpr"].append(res.fpr)
                cells[a]["violation_p"].append(res.violation_p)
        out["per_scorer"][scorer] = {
            _akey(a): {
                "alpha": float(a),
                "scorer": scorer,
                "n_cal": n_cal,
                "n_test_human": int(abs_h_idx.size),
                "fpr": cells[a]["fpr"],
                "fpr_summary": _agg(cells[a]["fpr"]),
                "violation_p": cells[a]["violation_p"],
                "violation_p_mean": _agg(cells[a]["violation_p"])["mean"],
            }
            for a in alphas
        }
    return out


# --------------------------------------------------------------------------- #
# E4 (HEADLINE): human edit attack — same humans, edited; isolates the edit
# effect from population shift because calibration and test share the population.
# --------------------------------------------------------------------------- #

def _run_e4(
    book: ScoreBook,
    scorers: Sequence[str],
    h_idx: np.ndarray,
    cal_draws: list[np.ndarray],
    attacked_idx: dict[tuple[str, float], np.ndarray],
    alphas: Sequence[float],
) -> dict:
    n_cal = int(cal_draws[0].size)
    n_test = int(h_idx.size)
    out: dict[str, Any] = {
        "note": (
            "calibration on clean EVAL humans; test on attacked copies of the same "
            "EVAL humans (fixed per kind/rate) — violations isolate the edit effect"
        ),
        "n_cal": n_cal,
        "n_test": n_test,
        "envelope": _envelopes(n_cal, n_test, alphas),
        "per_scorer": {},
    }
    for scorer in scorers:
        sarr = book.get(scorer, h_idx)
        sc: dict[str, dict] = {}
        for (kind, rate), t_idx in attacked_idx.items():
            tscores = book.get(scorer, t_idx)
            cells = {a: {"fpr": [], "violation_p": []} for a in alphas}
            for draw in cal_draws:
                cal = sarr[draw]
                for a in alphas:
                    res = evaluate_certificate(cal, tscores, None, a)
                    cells[a]["fpr"].append(res.fpr)
                    cells[a]["violation_p"].append(res.violation_p)
            sc.setdefault(kind, {})[_rkey(rate)] = {
                _akey(a): {
                    "alpha": float(a),
                    "scorer": scorer,
                    "attack": kind,
                    "rate": float(rate),
                    "n_cal": n_cal,
                    "n_test_human": n_test,
                    "fpr": cells[a]["fpr"],
                    "fpr_summary": _agg(cells[a]["fpr"]),
                    "violation_p": cells[a]["violation_p"],
                    "violation_p_mean": _agg(cells[a]["violation_p"])["mean"],
                }
                for a in alphas
            }
        out["per_scorer"][scorer] = sc
    return out


# --------------------------------------------------------------------------- #
# E5: leakage — random vs grouped calibration FPR distributions (from E1)
# --------------------------------------------------------------------------- #

def _run_e5(e1: dict, scorers: Sequence[str], alphas: Sequence[float]) -> dict:
    out: dict[str, Any] = {
        "note": (
            "FPR distributions copied from E1 random vs grouped partitions; "
            "the miscoverage delta is computed at analysis time"
        ),
        "per_scorer": {},
    }
    for scorer in scorers:
        out["per_scorer"][scorer] = {}
        for a in alphas:
            ak = _akey(a)
            rnd = e1["random"]["per_scorer"][scorer][ak]
            grp = e1["grouped"]["per_scorer"][scorer][ak]
            # One self-describing record per split so figures can tell them apart.
            out["per_scorer"][scorer][ak] = {
                split: {
                    "alpha": float(a),
                    "scorer": scorer,
                    "split": split,
                    "fpr": src["fpr"],
                    "fpr_summary": src["fpr_summary"],
                }
                for split, src in (("random", rnd), ("grouped", grp))
            }
    return out


# --------------------------------------------------------------------------- #
# E6: Mondrian by length — per-bin FPR, marginal vs Mondrian calibration
# --------------------------------------------------------------------------- #

def _run_e6(
    book: ScoreBook,
    scorers: Sequence[str],
    h_idx: np.ndarray,
    h_texts: Sequence[str],
    masks: list[np.ndarray],
) -> dict:
    bins = length_bins(h_texts)
    n_bins = len(_LENGTH_BIN_EDGES) - 1
    counts = {str(b): int((bins == b).sum()) for b in range(n_bins)}
    out: dict[str, Any] = {
        "bin_edges_words": list(_LENGTH_BIN_EDGES),
        "bin_counts_eval_humans": counts,
        "alphas": [float(a) for a in _E6_ALPHAS],
        "per_scorer": {},
    }
    for scorer in scorers:
        sarr = book.get(scorer, h_idx)
        cells = {
            a: {m: {b: [] for b in range(n_bins)} for m in ("marginal", "mondrian")}
            for a in _E6_ALPHAS
        }
        for mask in masks:
            cal, test = sarr[mask], sarr[~mask]
            cb, tb = bins[mask], bins[~mask]
            for a in _E6_ALPHAS:
                fl_marg = flag(cal, test, a)
                fl_mond = mondrian_flag(cal, cb, test, tb, a)
                for b in range(n_bins):
                    sel = tb == b
                    for method, fl in (("marginal", fl_marg), ("mondrian", fl_mond)):
                        cells[a][method][b].append(
                            float(fl[sel].mean()) if sel.any() else math.nan
                        )
        out["per_scorer"][scorer] = {
            _akey(a): {
                method: {
                    str(b): {
                        "alpha": float(a),
                        "scorer": scorer,
                        "method": method,
                        "bin": str(b),
                        "fpr": cells[a][method][b],
                        "fpr_summary": _agg(cells[a][method][b]),
                    }
                    for b in range(n_bins)
                }
                for method in ("marginal", "mondrian")
            }
            for a in _E6_ALPHAS
        }
    return out


# --------------------------------------------------------------------------- #
# E7: power frontier — TPR per certified alpha, clean vs evasion-attacked AI
# --------------------------------------------------------------------------- #

def _run_e7(
    book: ScoreBook,
    scorers: Sequence[str],
    h_idx: np.ndarray,
    cal_draws: list[np.ndarray],
    ai_idx: np.ndarray,
    ai_evasion_idx: np.ndarray,
    alphas: Sequence[float],
    evasion_kind: str,
    evasion_rate: float,
) -> dict:
    out: dict[str, Any] = {
        "n_cal": int(cal_draws[0].size),
        "n_test_ai": int(ai_idx.size),
        "evasion_kind": evasion_kind,
        "evasion_rate": float(evasion_rate),
        "clean": {},
        "evasion": {},
    }
    for scorer in scorers:
        sarr = book.get(scorer, h_idx)
        ai_clean = book.get(scorer, ai_idx)
        ai_atk = book.get(scorer, ai_evasion_idx)
        cells = {a: {"clean": [], "evasion": []} for a in alphas}
        for draw in cal_draws:
            cal = sarr[draw]
            for a in alphas:
                cells[a]["clean"].append(float(flag(cal, ai_clean, a).mean()))
                cells[a]["evasion"].append(float(flag(cal, ai_atk, a).mean()))
        for variant in ("clean", "evasion"):
            out[variant][scorer] = {
                _akey(a): {
                    "alpha": float(a),
                    "scorer": scorer,
                    "condition": variant,
                    "tpr": cells[a][variant],
                    "tpr_summary": _agg(cells[a][variant]),
                }
                for a in alphas
            }
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_all(cfg: dict | None = None) -> dict:
    """Run the full E1-E7 study and return one JSON-serialisable results dict.

    See the module docstring for the design. Keys of the returned dict:
    meta, e1, e2, e3, e4, e5, e6, e7. Every experiment stores per-repeat arrays
    (length R) plus mean / 95% percentile-interval summaries; nothing is hand-entered
    downstream. Deterministic given cfg['seed'].
    """
    t_start = time.time()
    c = dict(DEFAULT_CFG)
    c.update(cfg or {})
    seed = int(c["seed"])
    R = int(c["R"])
    alphas = [float(a) for a in c["alphas"]]
    edit_rates = [float(r) for r in c["edit_rates"]]
    cal_frac = float(c["cal_frac"])
    cap = c["max_gpt2_texts"]
    cap = int(cap) if cap is not None else None
    warnings_list: list[str] = []

    # ---- Stage 0a: MAiDE English, entity-grouped master split -------------- #
    print(f"[guard] loading MAiDE-up English from {c['data_csv']}", flush=True)
    df = _load_maide(c["data_csv"], warnings_list)
    train_df, eval_df = _master_split(df, int(c["master_split_seed"]), warnings_list)
    print(
        f"[guard] master split: train={len(train_df)} rows / "
        f"{train_df['hotel'].nunique()} hotels, eval={len(eval_df)} rows / "
        f"{eval_df['hotel'].nunique()} hotels",
        flush=True,
    )

    eh = eval_df[eval_df["label"] == 0]
    ea = eval_df[eval_df["label"] == 1]
    sel_h = _cap_indices(len(eh), cap, np.random.default_rng((seed, 7)))
    sel_a = _cap_indices(len(ea), cap, np.random.default_rng((seed, 8)))
    eval_h_texts = eh["text"].to_numpy()[sel_h].tolist()
    eval_h_hotels = eh["hotel"].to_numpy()[sel_h]
    eval_ai_texts = ea["text"].to_numpy()[sel_a].tolist()

    # ---- Stage 0b: attacked variants (humans: certificate attack; AI: evasion) #
    print(
        f"[guard] generating attacked variants: kinds={list(_ATTACK_KINDS)}, "
        f"rates={edit_rates}, evasion={c['evasion_kind']}@{c['evasion_rate']}",
        flush=True,
    )
    attacked_h: dict[tuple[str, float], list[str]] = {}
    for ki, kind in enumerate(_ATTACK_KINDS):
        atk = _resolve_attack(kind)
        for ri, rate in enumerate(edit_rates):
            rng = np.random.default_rng((seed, 100 + 10 * ki + ri))
            attacked_h[(kind, rate)] = _attack_texts(
                atk, eval_h_texts, rate, rng, warnings_list, f"{kind}@{rate:g} (human)"
            )
    evasion_kind = str(c["evasion_kind"])
    evasion_rate = float(c["evasion_rate"])
    atk_ev = _resolve_attack(evasion_kind)
    attacked_ai = _attack_texts(
        atk_ev,
        eval_ai_texts,
        evasion_rate,
        np.random.default_rng((seed, 199)),
        warnings_list,
        f"{evasion_kind}@{evasion_rate:g} (AI evasion)",
    )

    # ---- Stage 0c: external pools ------------------------------------------ #
    pool_df = _load_raid_frame(c["reviews_pool"])
    pool_h_texts: list[str] | None = None
    pool_gens: dict[str, list[str]] | None = None
    if pool_df is None:
        print("[guard] reviews pool cache absent -> E2 will be skipped", flush=True)
    else:
        pool_df = _filter_clean_attack(pool_df)
        hm = _human_mask(pool_df)
        rng_p = np.random.default_rng((seed, 9))
        ph = pool_df.loc[hm, "text"].astype(str).tolist()
        pool_h_texts = [ph[i] for i in _cap_indices(len(ph), cap, rng_p)]
        pool_gens = {}
        ai_rows = pool_df.loc[~hm]
        for gen in sorted(ai_rows["model"].astype(str).unique()):
            gt = ai_rows.loc[ai_rows["model"].astype(str) == gen, "text"].astype(str).tolist()
            pool_gens[gen] = [gt[i] for i in _cap_indices(len(gt), cap, rng_p)]
        print(
            f"[guard] reviews pool: {len(pool_h_texts)} humans, "
            f"{len(pool_gens)} generators",
            flush=True,
        )

    abs_df = _load_raid_frame(c["abstracts_parquet"])
    abs_h_texts: list[str] | None = None
    if abs_df is None:
        print("[guard] abstracts cache absent -> E3 will be skipped", flush=True)
    else:
        ah = abs_df.loc[_human_mask(abs_df), "text"].astype(str).tolist()
        abs_h_texts = [ah[i] for i in _cap_indices(len(ah), cap, np.random.default_rng((seed, 10)))]
        print(f"[guard] abstracts: {len(abs_h_texts)} human texts", flush=True)

    # ---- Stage 0d: score every unique text exactly once --------------------- #
    ordered: list[Sequence[str]] = [eval_h_texts, eval_ai_texts]
    ordered.extend(attacked_h[(k, r)] for k in _ATTACK_KINDS for r in edit_rates)
    ordered.append(attacked_ai)
    if pool_h_texts is not None and pool_gens is not None:
        ordered.append(pool_h_texts)
        ordered.extend(pool_gens[g] for g in pool_gens)
    if abs_h_texts is not None:
        ordered.append(abs_h_texts)
    unique_texts = list(dict.fromkeys(t for block in ordered for t in block))
    print(
        f"[guard] scoring {len(unique_texts)} unique texts with scorers={c['scorers']} "
        f"(cache: {c['cache_path']})",
        flush=True,
    )
    scores = _score_all(
        unique_texts,
        list(c["scorers"]),
        c["cache_path"],
        train_df["text"].tolist(),
        train_df["label"].to_numpy(dtype=int),
        warnings_list,
    )
    book = ScoreBook(unique_texts, scores)
    scorers_used = [s for s in c["scorers"] if s in book.scores]
    print(f"[guard] scorers in play: {scorers_used}", flush=True)

    h_idx = book.idx(eval_h_texts)
    ai_idx = book.idx(eval_ai_texts)
    attacked_h_idx = {key: book.idx(txts) for key, txts in attacked_h.items()}
    attacked_ai_idx = book.idx(attacked_ai)
    pool_h_idx = book.idx(pool_h_texts) if pool_h_texts is not None else None
    pool_gen_idx = (
        {g: book.idx(t) for g, t in pool_gens.items()} if pool_gens is not None else None
    )
    abs_h_idx = book.idx(abs_h_texts) if abs_h_texts is not None else None

    # ---- Stage 0e: resampling plans (paired across experiments) ------------- #
    n_h = int(h_idx.size)
    n_cal_part = min(max(1, int(round(cal_frac * n_h))), n_h - 1)
    random_parts = _partition_masks(n_h, n_cal_part, R, np.random.default_rng((seed, 1)))
    grouped_parts = _grouped_partition_masks(eval_h_hotels, R, np.random.default_rng((seed, 11)))
    n_cal_draw = min(max(1, int(round(cal_frac * n_h))), n_h)
    cal_draws = _calibration_draws(n_h, n_cal_draw, R, np.random.default_rng((seed, 2)))

    results: dict[str, Any] = {}

    def _stage(name: str, fn: Callable[[], dict]) -> dict:
        t0 = time.time()
        print(f"[guard] {name} ...", flush=True)
        res = fn()
        print(f"[guard] {name} done in {time.time() - t0:.1f}s", flush=True)
        return res

    results["e1"] = _stage(
        "E1 validity in-distribution",
        lambda: _run_e1(
            book, scorers_used, h_idx,
            {"random": random_parts, "grouped": grouped_parts}, alphas,
        ),
    )
    results["e2"] = _stage(
        "E2 generator shift",
        lambda: _run_e2(book, scorers_used, h_idx, cal_draws, pool_h_idx, pool_gen_idx, alphas),
    )
    results["e3"] = _stage(
        "E3 human domain shift",
        lambda: _run_e3(book, scorers_used, h_idx, cal_draws, abs_h_idx, alphas),
    )
    results["e4"] = _stage(
        "E4 human edit attack (headline)",
        lambda: _run_e4(book, scorers_used, h_idx, cal_draws, attacked_h_idx, alphas),
    )
    results["e5"] = _stage(
        "E5 leakage (random vs grouped)",
        lambda: _run_e5(results["e1"], scorers_used, alphas),
    )
    results["e6"] = _stage(
        "E6 Mondrian by length",
        lambda: _run_e6(book, scorers_used, h_idx, eval_h_texts, random_parts),
    )
    results["e7"] = _stage(
        "E7 power frontier",
        lambda: _run_e7(
            book, scorers_used, h_idx, cal_draws, ai_idx, attacked_ai_idx,
            alphas, evasion_kind, evasion_rate,
        ),
    )

    results["meta"] = {
        "cfg": {k: (str(v) if isinstance(v, Path) else v) for k, v in c.items()},
        "scorers_used": scorers_used,
        "counts": {
            "maide_english_rows": int(len(df)),
            "train_rows": int(len(train_df)),
            "eval_rows": int(len(eval_df)),
            "train_hotels": int(train_df["hotel"].nunique()),
            "eval_hotels": int(eval_df["hotel"].nunique()),
            "eval_humans": int(len(eval_h_texts)),
            "eval_ai": int(len(eval_ai_texts)),
            "pool_humans": int(len(pool_h_texts)) if pool_h_texts is not None else None,
            "pool_generators": (
                {g: len(t) for g, t in pool_gens.items()} if pool_gens is not None else None
            ),
            "abstract_humans": int(len(abs_h_texts)) if abs_h_texts is not None else None,
            "unique_texts_scored": int(len(unique_texts)),
            "n_cal_partition": int(n_cal_part),
            "n_cal_draw": int(n_cal_draw),
        },
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
        "warnings": warnings_list,
        "started_utc": datetime.fromtimestamp(t_start, tz=timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - t_start, 2),
    }
    print(f"[guard] study complete in {results['meta']['runtime_seconds']}s", flush=True)
    ordered_out = {k: results[k] for k in ("meta", "e1", "e2", "e3", "e4", "e5", "e6", "e7")}
    return _jsonable(ordered_out)
