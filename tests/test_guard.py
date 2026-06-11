"""Self-checks for the GUARD codebase: certificate math, attacks, data, and scorers.

Scientific purpose: every experimental claim in METHODOLOGY.md rests on (a) the split
conformal certificate being implemented exactly (validity, Beta envelope, exact violation
test, Mondrian variant), (b) the attacks being deterministic and label-preserving so E4
violation curves are reproducible, (c) the data layer being leakage-aware (entity-disjoint
hotels), and (d) the base scorers producing sane, correctly oriented statistics. This file
verifies each of those preconditions on synthetic data and on the real MAiDE-up frame.

Runnable two ways:
    python tests/test_guard.py          # plain runner: PASS/SKIP/FAIL lines, exit code
    pytest tests/test_guard.py          # standard pytest collection

The GPT-2 scorer check downloads/loads a HuggingFace model and is skipped unless the
environment variable GUARD_TEST_GPT2=1 is set.

No experimental results are asserted here -- only mathematical and structural invariants.
"""
from __future__ import annotations

import os
import sys
import traceback
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from guard.conformal import (
    beta_envelope,
    flag,
    flag_threshold,
    mondrian_flag,
    violation_pvalue,
)

ROOT = Path(__file__).resolve().parents[1]
MAIDE_CSV = ROOT / "data" / "all_data.csv"


# ------------------------------- 1. conformal validity ------------------------------- #

def test_conformal_validity_matches_beta_envelope() -> None:
    """Mean FPR over 1000 exchangeable draws must match the Beta-envelope mean.

    Under exchangeability the conditional FPR is Beta(k, n+1-k); the empirical mean over
    many independent calibration/test draws must therefore agree with the analytic mean
    to within Monte-Carlo error (tolerance 0.005 is ~7 standard errors here).
    """
    alpha, n_cal, n_test, n_draws = 0.05, 200, 200, 1000
    rng = np.random.default_rng(12345)
    fprs = np.empty(n_draws)
    for r in range(n_draws):
        cal = rng.standard_normal(n_cal)
        test = rng.standard_normal(n_test)
        fprs[r] = flag(cal, test, alpha).mean()
    beta_mean, _, _ = beta_envelope(n_cal, alpha)
    assert abs(fprs.mean() - beta_mean) < 0.005, (
        f"empirical mean FPR {fprs.mean():.5f} vs Beta mean {beta_mean:.5f}"
    )


# ----------------------------- 2. violation test calibration ----------------------------- #

def test_violation_pvalue_calibrated_and_powerful() -> None:
    """The exact Beta-Binomial violation test must be valid under H0 and detect shift.

    Under H0 (test humans exchangeable with calibration humans) a valid p-value satisfies
    P(p <= 0.05) <= 0.05; we allow 0.075 for Monte-Carlo noise over 500 draws. Under a
    1-sigma upward shift of the test humans (the E4 failure mode) the median p-value must
    collapse below 1e-6.
    """
    alpha, n_cal, n_test, n_draws = 0.05, 200, 200, 500
    rng = np.random.default_rng(777)

    p_null = np.empty(n_draws)
    p_shift = np.empty(n_draws)
    for r in range(n_draws):
        cal = rng.standard_normal(n_cal)
        test_h0 = rng.standard_normal(n_test)
        test_shift = rng.standard_normal(n_test) + 1.0
        k0 = int(flag(cal, test_h0, alpha).sum())
        k1 = int(flag(cal, test_shift, alpha).sum())
        p_null[r] = violation_pvalue(k0, n_test, n_cal, alpha)
        p_shift[r] = violation_pvalue(k1, n_test, n_cal, alpha)

    frac_sig = float((p_null <= 0.05).mean())
    assert frac_sig <= 0.075, f"H0 rejection rate {frac_sig:.3f} > 0.075 (miscalibrated)"
    med = float(np.median(p_shift))
    assert med < 1e-6, f"median p under 1-sigma shift {med:.3g} not < 1e-6 (no power)"


# --------------------------------- 3. Mondrian validity --------------------------------- #

def test_mondrian_per_group_fpr() -> None:
    """Per-group FPR of mondrian_flag must respect alpha on two-scale synthetic groups.

    Group 0 scores ~ N(0,1), group 1 scores ~ N(0,5): a single pooled threshold would
    over-flag the wide group, but Mondrian calibration is group-conditionally valid, so
    each group's mean FPR (over 10 draws) must stay <= alpha + 0.05.
    """
    alpha, n_cal_g, n_test_g, n_draws = 0.10, 500, 2000, 10
    scales = {0: 1.0, 1: 5.0}
    rng = np.random.default_rng(2024)
    fpr_sums = {0: 0.0, 1: 0.0}
    for _ in range(n_draws):
        cal_scores = np.concatenate(
            [rng.standard_normal(n_cal_g) * scales[g] for g in (0, 1)]
        )
        cal_groups = np.repeat([0, 1], n_cal_g)
        test_scores = np.concatenate(
            [rng.standard_normal(n_test_g) * scales[g] for g in (0, 1)]
        )
        test_groups = np.repeat([0, 1], n_test_g)
        flags = mondrian_flag(cal_scores, cal_groups, test_scores, test_groups, alpha)
        for g in (0, 1):
            fpr_sums[g] += float(flags[test_groups == g].mean())
    for g in (0, 1):
        fpr_g = fpr_sums[g] / n_draws
        assert fpr_g <= alpha + 0.05, f"group {g} FPR {fpr_g:.4f} > {alpha + 0.05:.4f}"


# ----------------------------- 4. flag/threshold equivalence ----------------------------- #

def test_flag_threshold_equivalence() -> None:
    """For continuous scores, flag(cal, test, a) must equal test > flag_threshold(cal, a).

    Includes an alpha small enough that k < 1, where the threshold is +inf and the
    p-value rule likewise can never flag.
    """
    rng = np.random.default_rng(99)
    cal = rng.standard_normal(137)
    test = rng.standard_normal(500)
    for alpha in (0.001, 0.01, 0.05, 0.2, 0.5):
        via_pval = flag(cal, test, alpha)
        thr = flag_threshold(cal, alpha)
        via_thr = test > thr
        assert np.array_equal(via_pval, via_thr), (
            f"flag/threshold mismatch at alpha={alpha} (threshold {thr})"
        )


# -------------------------------------- 5. attacks -------------------------------------- #

_ATTACK_TEXT = (
    "The room was very clean and the staff were friendly because they had been told "
    "about our late arrival but the breakfast service was slow and we could not find "
    "a quiet table near the window in the morning."
)


def test_attacks_fw_perturb() -> None:
    """fw_perturb: deterministic given seed, and rate=0 is the identity."""
    from guard.attacks import fw_perturb

    a = fw_perturb(_ATTACK_TEXT, 0.3, seed=7)
    b = fw_perturb(_ATTACK_TEXT, 0.3, seed=7)
    assert a == b, "fw_perturb not deterministic for fixed (text, rate, seed)"
    assert fw_perturb(_ATTACK_TEXT, 0.0, seed=11) == _ATTACK_TEXT, "rate=0 not identity"
    assert a != _ATTACK_TEXT, "fw_perturb at rate=0.3 left a function-word-rich text intact"


def test_attacks_synonym_perturb() -> None:
    """synonym_perturb: deterministic given seed, and rate=0 is the identity.

    Skipped if the WordNet corpus cannot be loaded (e.g. no network for the first-use
    download); the attack module itself is responsible for fetching it.
    """
    from guard.attacks import synonym_perturb

    try:
        a = synonym_perturb(_ATTACK_TEXT, 0.5, seed=7)
    except (LookupError, RuntimeError, OSError) as exc:
        raise unittest.SkipTest(f"WordNet unavailable: {exc}") from exc
    b = synonym_perturb(_ATTACK_TEXT, 0.5, seed=7)
    assert a == b, "synonym_perturb not deterministic for fixed (text, rate, seed)"
    assert synonym_perturb(_ATTACK_TEXT, 0.0, seed=11) == _ATTACK_TEXT, "rate=0 not identity"


def test_perturb_frame_label_preserving() -> None:
    """perturb_frame must alter only the text column and be deterministic; rate=0 identity."""
    import pandas as pd

    from guard.attacks import perturb_frame

    df = pd.DataFrame(
        {
            "text": [
                "The pool was warm and the view from the balcony was lovely in the evening.",
                "We did not like the noise from the street but the bed was comfortable.",
                "The staff at the desk were helpful when we asked about the old town.",
                "Breakfast was cold and we had to wait for a table by the door.",
            ],
            "label": [0, 1, 0, 1],
            "hotel": ["A", "A", "B", "B"],
        }
    )
    before = df.copy(deep=True)

    out1 = perturb_frame(df, "fw", 0.5, seed=3)
    out2 = perturb_frame(df, "fw", 0.5, seed=3)
    assert out1["text"].tolist() == out2["text"].tolist(), "perturb_frame not deterministic"
    assert out1["label"].tolist() == df["label"].tolist(), "labels not preserved"
    assert out1["hotel"].tolist() == df["hotel"].tolist(), "metadata not preserved"
    assert len(out1) == len(df), "row count changed"
    assert df.equals(before), "perturb_frame mutated its input frame"
    assert out1["text"].tolist() != df["text"].tolist(), "rate=0.5 changed no text at all"

    out0 = perturb_frame(df, "fw", 0.0, seed=3)
    assert out0["text"].tolist() == df["text"].tolist(), "rate=0 not identity in frame"


# --------------------------------------- 6. data ---------------------------------------- #

def test_data_maide_and_grouped_split() -> None:
    """load_maide_english: 2000 balanced rows; grouped_split: zero shared hotels."""
    from guard.data import grouped_split, load_maide_english

    if not MAIDE_CSV.exists():
        raise unittest.SkipTest(f"MAiDE-up CSV not found at {MAIDE_CSV}")

    df = load_maide_english(MAIDE_CSV)
    assert len(df) == 2000, f"expected 2000 English rows, got {len(df)}"
    counts = df["label"].value_counts()
    assert int(counts.get(0, 0)) == 1000, f"human rows {counts.get(0, 0)} != 1000"
    assert int(counts.get(1, 0)) == 1000, f"AI rows {counts.get(1, 0)} != 1000"
    for col in ("text", "label", "hotel"):
        assert col in df.columns, f"missing column {col!r}"

    df_a, df_b = grouped_split(df, test_frac=0.5, seed=0)
    shared = set(df_a["hotel"]) & set(df_b["hotel"])
    assert not shared, f"grouped_split leaked {len(shared)} hotels across the split"
    assert len(df_a) > 0 and len(df_b) > 0, "grouped_split produced an empty side"
    assert len(df_a) + len(df_b) == len(df), "grouped_split dropped or duplicated rows"


# -------------------------------------- 7. scores --------------------------------------- #

def test_scores_gpt2() -> None:
    """GPT2Scores smoke check (env-gated): four finite stats; fluent beats gibberish.

    Heavyweight (loads GPT-2); enable with GUARD_TEST_GPT2=1.
    """
    if os.environ.get("GUARD_TEST_GPT2") != "1":
        raise unittest.SkipTest("set GUARD_TEST_GPT2=1 to run the GPT-2 scorer check")

    from guard.scores import GPT2_STATS, GPT2Scores

    fluent_a = (
        "The hotel room was clean and quiet, and the staff were friendly "
        "throughout our stay in the city."
    )
    fluent_b = (
        "We enjoyed the breakfast every morning and the location was convenient "
        "for visiting the old town on foot."
    )
    gibberish = "zxq plomf vrk trchx wupq jzmbl qrtv xkc dwpl ngh zzv blorp qpx"

    scorer = GPT2Scores()
    results = [scorer.score_text(t) for t in (fluent_a, fluent_b, gibberish)]
    for text, res in zip(("fluent_a", "fluent_b", "gibberish"), results):
        for stat in GPT2_STATS:
            assert np.isfinite(res[stat]), f"{stat} not finite for {text}: {res[stat]}"
    assert results[0]["loglik"] > results[2]["loglik"], "loglik(fluent_a) <= loglik(gibberish)"
    assert results[1]["loglik"] > results[2]["loglik"], "loglik(fluent_b) <= loglik(gibberish)"


def test_scores_tfidf() -> None:
    """TfidfScorer fit + score smoke check on 50 synthetic rows (training-set AUC > 0.7)."""
    from sklearn.metrics import roc_auc_score

    from guard.scores import TfidfScorer

    human = [
        f"honestly room {i} was kinda noisy at night but the staff were sweet and "
        f"breakfast ran late on day {i % 5} which suited us fine"
        for i in range(25)
    ]
    ai = [
        f"the accommodation provided an exceptional experience with impeccable service "
        f"and pristine facilities and visit {i} exceeded expectations in every regard"
        for i in range(25)
    ]
    texts = human + ai
    labels = [0] * 25 + [1] * 25

    scorer = TfidfScorer().fit(texts, labels)
    scores = scorer.score(texts)
    assert scores.shape == (50,), f"score shape {scores.shape} != (50,)"
    assert np.all(np.isfinite(scores)), "non-finite TF-IDF scores"
    auc = roc_auc_score(labels, scores)
    assert auc > 0.7, f"training-set AUC {auc:.3f} <= 0.7 (orientation or fit broken)"


# --------------------------------------- runner ----------------------------------------- #

TESTS = [
    test_conformal_validity_matches_beta_envelope,
    test_violation_pvalue_calibrated_and_powerful,
    test_mondrian_per_group_fpr,
    test_flag_threshold_equivalence,
    test_attacks_fw_perturb,
    test_attacks_synonym_perturb,
    test_perturb_frame_label_preserving,
    test_data_maide_and_grouped_split,
    test_scores_gpt2,
    test_scores_tfidf,
]


def main() -> int:
    """Run all tests; print one PASS/SKIP/FAIL line each; return nonzero on any failure."""
    n_fail = 0
    for test in TESTS:
        name = test.__name__
        try:
            test()
        except unittest.SkipTest as exc:
            print(f"SKIP {name} ({exc})")
        except Exception:
            n_fail += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    if n_fail:
        print(f"{n_fail} test(s) FAILED")
    else:
        print("all tests passed (or skipped)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
