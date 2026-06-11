"""Split-conformal false-accusation certificates for machine-generated-text detection.

The deployment contract: given any base detector score s(x) oriented so that HIGHER = more
AI-like, calibrate on n human-written texts and flag a new text as AI only when its conformal
p-value against the human calibration class is <= alpha. If the new text is human and
exchangeable with the calibration humans, then P(flag) <= alpha — finite-sample,
distribution-free, for ANY score function (Vovk et al., 2005; Papadopoulos et al., 2002).

Only the HUMAN distribution matters for validity; the AI distribution affects power alone.
That asymmetry is the object of study in this project.

Conditional on the calibration set, the realised false-positive rate of the threshold rule is
itself random across calibration draws; for continuous scores the flagging threshold is the
k-th largest calibration score with k = floor(alpha * (n + 1)), and the conditional FPR
equals the probability mass above that order statistic, distributed Beta(k, n + 1 - k).
`beta_envelope` exposes that law so experiments can verify the whole distribution, not just
the mean.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats


def conformal_pvalues(cal_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """Conformal p-value of each test score against the human calibration class.

    p(x) = (1 + #{i: cal_i >= s(x)}) / (n + 1). Ties count toward the calibration side,
    which is the conservative direction (larger p, fewer flags).

    A non-finite test score (NaN: the scorer could not evaluate the text) carries no
    evidence of AI authorship, so it gets p = 1 and is never flagged. (Without this guard,
    searchsorted treats NaN as +inf and an unscoreable text would receive the MINIMAL
    p-value 1/(n+1), i.e. be accused with maximal confidence.)
    """
    cal = np.asarray(cal_scores, dtype=float)
    test = np.asarray(test_scores, dtype=float)
    n = cal.size
    # For each test score, count calibration scores >= it (vectorised via sorted search).
    cal_sorted = np.sort(cal)
    # number of cal < s  -> index from searchsorted(left); cal >= s = n - that
    n_less = np.searchsorted(cal_sorted, test, side="left")
    n_geq = n - n_less
    p = (1.0 + n_geq) / (n + 1.0)
    return np.where(np.isfinite(test), p, 1.0)


def flag(cal_scores: np.ndarray, test_scores: np.ndarray, alpha: float) -> np.ndarray:
    """Boolean mask: flag-as-AI decisions certified at level alpha."""
    return conformal_pvalues(cal_scores, test_scores) <= alpha


def flag_threshold(cal_scores: np.ndarray, alpha: float) -> float:
    """The equivalent score threshold: flag iff s(x) > t (continuous-score view).

    t is the k-th largest calibration score with k = floor(alpha * (n + 1)). If
    k < 1 the certificate cannot flag anything (alpha too small for n) and t = +inf.
    """
    cal = np.sort(np.asarray(cal_scores, dtype=float))[::-1]  # descending
    n = cal.size
    k = int(np.floor(alpha * (n + 1)))
    if k < 1:
        return float("inf")
    return float(cal[k - 1])


def beta_envelope(n: int, alpha: float, q: tuple = (0.05, 0.95)):
    """Distribution of the realised conditional FPR across calibration draws.

    For continuous scores, FPR | calibration ~ Beta(k, n + 1 - k), k = floor(alpha (n+1)).
    Returns (mean, lo, hi) at the requested quantiles; (0, 0, 0) when k < 1 (no flags
    possible). Used to draw the theoretical envelope that empirical FPR histograms must
    match when exchangeability holds.
    """
    k = int(np.floor(alpha * (n + 1)))
    if k < 1:
        return 0.0, 0.0, 0.0
    dist = scipy_stats.beta(k, n + 1 - k)
    return float(dist.mean()), float(dist.ppf(q[0])), float(dist.ppf(q[1]))


def violation_pvalue(n_false_flags: int, n_test_humans: int, n_cal: int, alpha: float) -> float:
    """Exact one-sided test of H0 'the certificate held' for one calibration/test draw.

    Under H0 (exchangeability), the count of false flags among m test humans given n
    calibration humans is Beta-Binomial(m, k, n + 1 - k) with k = floor(alpha (n+1)).
    Returns P(count >= observed) under H0; small values = certificate violated.
    """
    m = int(n_test_humans)
    k = int(np.floor(alpha * (n_cal + 1)))
    if k < 1:
        return 1.0 if n_false_flags == 0 else 0.0
    rv = scipy_stats.betabinom(m, k, n_cal + 1 - k)
    return float(rv.sf(n_false_flags - 1))  # P(X >= observed)


@dataclass
class CertifiedResult:
    """One calibration/test evaluation at one alpha."""

    alpha: float
    n_cal: int
    fpr: float          # realised false-accusation rate on test humans
    tpr: float          # power on test AI texts (nan if none provided)
    n_test_human: int
    n_test_ai: int
    violation_p: float  # exact Beta-Binomial tail probability of the observed FPR


def evaluate_certificate(
    cal_human: np.ndarray,
    test_human: np.ndarray,
    test_ai: np.ndarray | None,
    alpha: float,
) -> CertifiedResult:
    """Calibrate on human scores, decide on test scores, report validity + power."""
    cal = np.asarray(cal_human, dtype=float)
    th = np.asarray(test_human, dtype=float)
    fl_h = flag(cal, th, alpha)
    fpr = float(fl_h.mean()) if th.size else float("nan")
    if test_ai is not None and len(test_ai):
        ta = np.asarray(test_ai, dtype=float)
        tpr = float(flag(cal, ta, alpha).mean())
        n_ai = int(ta.size)
    else:
        tpr, n_ai = float("nan"), 0
    return CertifiedResult(
        alpha=alpha, n_cal=int(cal.size), fpr=fpr, tpr=tpr,
        n_test_human=int(th.size), n_test_ai=n_ai,
        violation_p=violation_pvalue(int(fl_h.sum()), int(th.size), int(cal.size), alpha),
    )


# ------------------------- Edit-robust (editor-aware) calibration ------------------------ #

def robust_calibration_scores(
    clean_scores: np.ndarray,
    edited_scores: np.ndarray,
) -> np.ndarray:
    """Editor-aware calibration scores: per calibration text, the max of its clean score and
    the scores of k sampled edits of that text.

    ``edited_scores`` has shape [n_cal, k] (NaN entries ignored). Returns shape [n_cal].

    Guarantee (conservative validity under a modeled editor): suppose a test human applies
    one edit drawn from the same edit process used to build the calibration variants, giving
    test score S = s(edit(X)). Each calibration text contributes
    M_i = max(s(X_i), s(edit_1(X_i)), ..., s(edit_k(X_i))) >= s(edit_j(X_i)) for every j, so
    (M_1, ..., M_n, S) is a sequence in which S is stochastically no larger than a fresh M
    would be; conformal p-values computed against {M_i} are therefore super-uniform and
    P(flag) <= alpha still holds. The price is power (thresholds move up). The guarantee is
    with respect to the MODELED editor; transfer to a mismatched editor is an empirical
    question (experiment E8 measures it).
    """
    clean = np.asarray(clean_scores, dtype=float).reshape(-1, 1)
    edited = np.asarray(edited_scores, dtype=float)
    if edited.ndim == 1:
        edited = edited.reshape(-1, 1)
    stacked = np.concatenate([clean, edited], axis=1)
    return np.nanmax(stacked, axis=1)


# ----------------------------- Mondrian (group-conditional) ----------------------------- #

def length_bins(texts, edges=(0, 30, 60, 120, 10_000)) -> np.ndarray:
    """Assign each text to a word-count bin. Default edges give short/medium/long/very-long."""
    counts = np.array([len(str(t).split()) for t in texts])
    return np.digitize(counts, bins=edges[1:-1])  # 0..len(edges)-2


def mondrian_flag(
    cal_scores: np.ndarray, cal_groups: np.ndarray,
    test_scores: np.ndarray, test_groups: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Per-group conformal flags: each group is calibrated only against its own calibration
    scores, giving group-conditional validity (Vovk's Mondrian construction). Groups with no
    calibration data never flag (conservative)."""
    cal_scores = np.asarray(cal_scores, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)
    out = np.zeros(test_scores.shape, dtype=bool)
    for g in np.unique(test_groups):
        cal_g = cal_scores[cal_groups == g]
        sel = test_groups == g
        if cal_g.size == 0:
            continue
        out[sel] = flag(cal_g, test_scores[sel], alpha)
    return out
