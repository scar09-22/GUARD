"""Base detector scores for GUARD's scorer-agnostic conformal certificate.

GUARD (see METHODOLOGY.md, Section 3) wraps ANY base detector score s(x) oriented so that
HIGHER = more AI-like; `guard.conformal` then turns the score into a distribution-free
false-accusation certificate. This module supplies the five deliberately modest base scores
used throughout E1-E7:

Training-free GPT-2 statistics, all derived from ONE forward pass per text (single
tokenization, single model call), so the per-text cost is one causal-LM evaluation:

- ``loglik``          : +mean token log-likelihood (AI text is more probable under the LM).
- ``logrank``         : -mean log(1-based rank of the observed token) (AI tokens rank higher).
- ``entropy``         : -mean predictive entropy (negated so higher = more AI-like).
- ``fast_detectgpt``  : Fast-DetectGPT analytic sampling discrepancy (Bao et al., 2024) with
  the scoring model as its own sampling model:
  (sum_t logp(obs_t) - sum_t E_p[logp]) / sqrt(sum_t Var_p[logp]),
  E_p[logp] = sum_v p_v logp_v and Var_p[logp] = sum_v p_v logp_v^2 - E^2 per position.

One supervised score:

- ``tfidf`` : TF-IDF (1-2 grams) + logistic regression decision_function, higher = AI.
  It MUST be fit on the training split only (grouped/entity-disjoint per METHODOLOGY.md E5).

Compute discipline: GPT-2 statistics are deterministic per (model, text) and dominate the
compute budget, so `compute_scores` caches them on disk in a parquet keyed by sha256(text)
and only evaluates texts not already cached. TF-IDF scores depend on the particular training
fit and are therefore NEVER cached; they are recomputed on every call.

No experimental results are hard-coded anywhere in this module.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# Names of the GPT-2 statistics, in canonical order. All four come from one forward pass.
GPT2_STATS: tuple[str, ...] = ("loglik", "logrank", "entropy", "fast_detectgpt")

# Name of the supervised score in `compute_scores` scorer_names.
TFIDF_STAT: str = "tfidf"


class GPT2Scores:
    """Four training-free detector statistics from a single GPT-2 forward pass per text.

    The HuggingFace model and tokenizer are loaded lazily on first use, so constructing
    this object is free (important for scripts that may only hit the score cache).
    All returned statistics are oriented HIGHER = more AI-like.
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cpu", max_tokens: int = 512) -> None:
        self.model_name = model_name
        self.device = device
        self.max_tokens = int(max_tokens)
        self._torch = None
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        """Load tokenizer + model on first use (lazy; heavy imports stay out of module load)."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    @staticmethod
    def _nan_result() -> dict[str, float]:
        return {name: float("nan") for name in GPT2_STATS}

    def score_text(self, text: str) -> dict[str, float]:
        """Compute all four GPT-2 statistics for one text from ONE forward pass.

        Returns a dict with keys ``loglik``, ``logrank``, ``entropy``, ``fast_detectgpt``.
        Texts that are empty or tokenize to fewer than 2 tokens (no next-token prediction
        is possible) return NaN for every statistic.
        """
        if not isinstance(text, str) or not text.strip():
            return self._nan_result()
        self._ensure_loaded()
        torch = self._torch

        enc = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_tokens
        )
        input_ids = enc["input_ids"].to(self.device)
        if input_ids.shape[1] < 2:
            return self._nan_result()

        with torch.no_grad():
            logits = self._model(input_ids).logits  # (1, T, V)
            # Predict token t+1 from position t; float32 log_softmax for numerical stability.
            logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)  # (T-1, V)
            targets = input_ids[0, 1:]  # (T-1,)
            tok_logp = logp.gather(1, targets.unsqueeze(1)).squeeze(1)  # logp(obs_t)
            # 1-based rank of the observed token: 1 + #{v : logp_v > logp_obs}.
            ranks = (logp > tok_logp.unsqueeze(1)).sum(dim=1) + 1
            p = logp.exp()
            e_logp = (p * logp).sum(dim=1)  # E_p[logp] = -predictive entropy, per position
            var_logp = (p * logp.square()).sum(dim=1) - e_logp.square()  # Var_p[logp]

            loglik = float(tok_logp.mean())
            logrank = float(-torch.log(ranks.float()).mean())
            entropy = float(e_logp.mean())  # = -(mean predictive entropy)
            var_sum = float(var_logp.sum())
            if var_sum > 0.0:
                fast_detectgpt = float((tok_logp.sum() - e_logp.sum())) / math.sqrt(var_sum)
            else:
                fast_detectgpt = float("nan")

        return {
            "loglik": loglik,
            "logrank": logrank,
            "entropy": entropy,
            "fast_detectgpt": fast_detectgpt,
        }


class TfidfScorer:
    """Supervised TF-IDF + logistic-regression detector score.

    LEAKAGE CONTRACT: this scorer MUST be fit on the TRAINING split only (entity-disjoint
    from calibration and test texts, per METHODOLOGY.md E5). Fitting on any text that later
    appears in calibration or test invalidates both the certificate experiments and the
    power comparison.

    Labels must be 0 = human, 1 = AI so that `score` (sklearn decision_function) is
    oriented higher = more AI-like, matching the orientation `guard.conformal` assumes.
    """

    def __init__(self) -> None:
        self._pipeline = None

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "TfidfScorer":
        """Fit the vectorizer + classifier on the training split (and nothing else)."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        y = np.asarray(labels, dtype=int)
        if set(np.unique(y)) - {0, 1}:
            raise ValueError("TfidfScorer labels must be 0 (human) / 1 (AI).")
        self._pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=50_000,
                        sublinear_tf=True,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=2000)),
            ]
        )
        self._pipeline.fit([str(t) for t in texts], y)
        return self

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Decision-function scores for `texts`; higher = more AI-like."""
        if self._pipeline is None:
            raise RuntimeError("TfidfScorer.score called before fit().")
        return np.asarray(
            self._pipeline.decision_function([str(t) for t in texts]), dtype=float
        )


def _sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _load_cache(cache_path: Path) -> pd.DataFrame:
    """Load the score cache (index = sha256 of text); empty frame if absent."""
    if cache_path.exists():
        cache = pd.read_parquet(cache_path)
        if cache.index.name != "sha256" and "sha256" in cache.columns:
            cache = cache.set_index("sha256")
        return cache
    return pd.DataFrame(index=pd.Index([], name="sha256"))


def _save_cache(cache: pd.DataFrame, cache_path: Path) -> None:
    """Atomically replace the cache file (write temp, then rename)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    cache.to_parquet(tmp_path)
    os.replace(tmp_path, cache_path)


def compute_scores(
    texts: Sequence[str],
    scorer_names: Sequence[str],
    cache_path: str | Path,
    gpt2: GPT2Scores | None = None,
    tfidf: TfidfScorer | None = None,
) -> pd.DataFrame:
    """Compute base detector scores for `texts`, caching the GPT-2 statistics on disk.

    Returns a DataFrame with one row per input text (in input order) and one column per
    entry of `scorer_names` (drawn from GPT2_STATS and/or ``tfidf``).

    Caching: GPT-2 statistics are deterministic per text, so they live in a parquet at
    `cache_path` keyed by sha256(text). Existing entries are loaded; only texts whose
    requested statistics are absent are evaluated (all four statistics are stored per
    evaluated text, since they share one forward pass); new rows are appended and the
    cache is rewritten atomically. TF-IDF scores depend on the particular training fit
    and are NOT cached; they are recomputed on every call from the provided `tfidf`,
    which must already be fit (on the training split only).
    """
    texts = [str(t) for t in texts]
    scorer_names = list(scorer_names)
    unknown = [s for s in scorer_names if s not in GPT2_STATS and s != TFIDF_STAT]
    if unknown:
        raise ValueError(
            f"Unknown scorer names {unknown}; valid names: {list(GPT2_STATS) + [TFIDF_STAT]}"
        )

    out = pd.DataFrame(index=pd.RangeIndex(len(texts)))
    gpt2_requested = [s for s in scorer_names if s in GPT2_STATS]

    if gpt2_requested:
        cache_path = Path(cache_path)
        cache = _load_cache(cache_path)
        hashes = [_sha256(t) for t in texts]

        # A cached row is usable only if every requested statistic exists as a column.
        have_cols = all(c in cache.columns for c in gpt2_requested)
        cached_keys = set(cache.index) if have_cols else set()

        # Unique missing texts, preserving first-occurrence order (duplicates scored once).
        missing: dict[str, str] = {}
        for h, t in zip(hashes, texts):
            if h not in cached_keys and h not in missing:
                missing[h] = t

        if missing:
            if gpt2 is None:
                gpt2 = GPT2Scores()
            new_rows: dict[str, dict[str, float]] = {}
            for i, (h, t) in enumerate(missing.items()):
                if i % 100 == 0:
                    print(
                        f"[guard.scores] gpt2 stats: {i}/{len(missing)} texts scored "
                        f"(cache: {cache_path.name})",
                        flush=True,
                    )
                new_rows[h] = gpt2.score_text(t)
            print(
                f"[guard.scores] gpt2 stats: {len(missing)}/{len(missing)} texts scored; "
                f"saving cache.",
                flush=True,
            )
            new_df = pd.DataFrame.from_dict(new_rows, orient="index", dtype=float)
            new_df.index.name = "sha256"
            cache = pd.concat([cache, new_df])
            cache = cache[~cache.index.duplicated(keep="last")]  # fresh rows win
            _save_cache(cache, cache_path)

        for stat in gpt2_requested:
            lookup = cache[stat].to_dict()
            out[stat] = [float(lookup.get(h, float("nan"))) for h in hashes]

    if TFIDF_STAT in scorer_names:
        if tfidf is None:
            raise ValueError(
                "scorer 'tfidf' requested but no fitted TfidfScorer was provided"
            )
        out[TFIDF_STAT] = tfidf.score(texts)

    return out[scorer_names]
