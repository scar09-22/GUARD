"""E8 — the edit-robust certificate: repairing validity under tool-assisted human editing.

The diagnosis (E4) shows innocent edits to human text inflate the false-accusation rate past
the nominal certificate. This experiment evaluates the FIX: editor-aware calibration
(guard.conformal.robust_calibration_scores) — each calibration human contributes the MAX
score over its clean version and k sampled edits from a modeled editor, which provably
restores P(flag | edited human) <= alpha when the deployment editor matches the modeled one
(conservative under stochastic dominance), at a measurable cost in power.

Conditions evaluated for naive vs robust calibration, per scorer, per alpha:
  clean humans            (both certificates should hold; robust is conservative)
  synonym@r_model humans  (matched editor: naive violates, robust must hold — THE claim)
  synonym@r_high humans   (stronger-than-modeled editor: transfer measured)
  fw@r_model humans       (mismatched editor family: transfer measured)
  clean AI / synonym AI   (power and its cost)

Scores are cached by text hash, so calibration-edit variants are scored once ever.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import numpy as np

from guard.data import load_maide_english, grouped_split, humans, ais
from guard.attacks import synonym_perturb, fw_perturb
from guard.scores import GPT2Scores, TfidfScorer, compute_scores
from guard.conformal import (
    flag, robust_calibration_scores, violation_pvalue,
)
from guard.experiments import _jsonable  # same serializer as the main engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="data/all_data.csv")
    ap.add_argument("--cache", default="results/cache/scores.parquet")
    ap.add_argument("--scorers", nargs="+",
                    default=["loglik", "logrank", "entropy", "fast_detectgpt", "tfidf"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    ap.add_argument("--k_edits", type=int, default=8, help="modeled-editor samples per cal text")
    ap.add_argument("--r_model", type=float, default=0.3, help="modeled editor edit rate")
    ap.add_argument("--r_high", type=float, default=0.5, help="stronger-than-modeled rate")
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/guard_robust.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = load_maide_english(args.data_csv)
    train_half, eval_half = grouped_split(df, test_frac=0.5, seed=0)  # same split as engine
    eval_h = humans(eval_half).reset_index(drop=True)
    eval_a = ais(eval_half).reset_index(drop=True)
    print(f"EVAL humans={len(eval_h)} AI={len(eval_a)}; k_edits={args.k_edits}")

    # --- text sets -------------------------------------------------------------------- #
    H = eval_h["text"].tolist()
    A = eval_a["text"].tolist()
    cal_edit_variants = [[synonym_perturb(t, args.r_model, seed=1000 + 17 * i + j)
                          for j in range(args.k_edits)] for i, t in enumerate(H)]
    test_sets = {
        "human_clean": H,
        "human_syn_model": [synonym_perturb(t, args.r_model, seed=5000 + i) for i, t in enumerate(H)],
        "human_syn_high": [synonym_perturb(t, args.r_high, seed=6000 + i) for i, t in enumerate(H)],
        "human_fw_model": [fw_perturb(t, args.r_model, seed=7000 + i) for i, t in enumerate(H)],
        "ai_clean": A,
        "ai_syn_model": [synonym_perturb(t, args.r_model, seed=8000 + i) for i, t in enumerate(A)],
    }

    # --- score everything once (cached) ------------------------------------------------ #
    gpt2 = GPT2Scores(device=args.device)
    tfidf = TfidfScorer()
    tfidf.fit(train_half["text"].tolist(), train_half["label"].tolist())

    flat_variants = [v for row in cal_edit_variants for v in row]
    all_texts = {name: texts for name, texts in test_sets.items()}
    all_texts["cal_edit_variants"] = flat_variants

    S = {}
    for name, texts in all_texts.items():
        print(f"[scores] {name}: {len(texts)} texts")
        S[name] = compute_scores(texts, args.scorers, args.cache, gpt2=gpt2, tfidf=tfidf)

    n_h = len(H)
    results = {"meta": {**vars(args), "n_eval_human": n_h, "n_eval_ai": len(A)}, "records": []}

    for scorer in args.scorers:
        s_clean_h = S["human_clean"][scorer].to_numpy()
        s_variants = S["cal_edit_variants"][scorer].to_numpy().reshape(n_h, args.k_edits)
        s_robust_all = robust_calibration_scores(s_clean_h, s_variants)  # per human text

        for r in range(args.R):
            idx = rng.permutation(n_h)
            cal_idx, test_idx = idx[: n_h // 2], idx[n_h // 2:]
            for mode, cal_scores in (("naive", s_clean_h[cal_idx]),
                                     ("robust", s_robust_all[cal_idx])):
                for cond, frame in S.items():
                    if cond == "cal_edit_variants":
                        continue
                    s_test = frame[scorer].to_numpy()
                    s_test = s_test[test_idx] if cond.startswith("human") else s_test
                    is_human = cond.startswith("human")
                    for alpha in args.alphas:
                        fl = flag(cal_scores, s_test, alpha)
                        rec = {"scorer": scorer, "mode": mode, "condition": cond,
                               "alpha": alpha, "rate": float(fl.mean()), "repeat": r}
                        if is_human:
                            rec["violation_p"] = violation_pvalue(
                                int(fl.sum()), int(s_test.size), int(cal_scores.size), alpha)
                        results["records"].append(rec)

    # ---- aggregate ---------------------------------------------------------------- #
    import pandas as pd
    rec = pd.DataFrame(results["records"])
    agg = (rec.groupby(["scorer", "mode", "condition", "alpha"])["rate"]
              .agg(["mean", lambda x: float(np.percentile(x, 2.5)),
                    lambda x: float(np.percentile(x, 97.5))]))
    agg.columns = ["mean", "lo", "hi"]
    results["summary"] = {
        "|".join(map(str, k)): v for k, v in agg.to_dict("index").items()
    }
    results["records"] = results["records"][: 20000]  # cap stored raw records

    import json, os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(_jsonable(results), f, indent=1)
    print(f"\nSaved {args.out}")

    print(f"\n{'scorer':<16}{'condition':<18}{'naive FPR/TPR':>16}{'robust FPR/TPR':>16}  (alpha=0.05)")
    for scorer in args.scorers:
        for cond in test_sets:
            try:
                nv = agg.loc[(scorer, "naive", cond, 0.05), "mean"]
                rb = agg.loc[(scorer, "robust", cond, 0.05), "mean"]
                print(f"{scorer:<16}{cond:<18}{nv:>15.3f} {rb:>15.3f}")
            except KeyError:
                pass


if __name__ == "__main__":
    main()
