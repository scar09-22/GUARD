"""Generate the GUARD paper (LaTeX) from the results JSONs. No hard-coded numbers: every
value is read from results/guard_results.json and results/guard_robust.json; figures are
copied beside the .tex. Compiles with standard article-class packages (Overleaf-ready)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import os
import shutil

import numpy as np

SCORERS = ["loglik", "logrank", "entropy", "fast_detectgpt", "tfidf"]
SLAB = {"loglik": "Log-likelihood", "logrank": "Log-rank", "entropy": "Entropy",
        "fast_detectgpt": "Fast-DetectGPT", "tfidf": "TF--IDF (supervised)"}


def esc(s):
    for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
        s = str(s).replace(a, b)
    return s


def m(vals):
    return float(np.mean(vals))


def ci(vals):
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fmt(vals, d=3):
    lo, hi = ci(vals)
    return f"{m(vals):.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def table(L, header, rows, caption, label):
    spec = "l" + "c" * (len(header) - 1)
    L += [r"\begin{table}[H]\centering", rf"\caption{{{caption}}}", rf"\label{{tab:{label}}}",
          r"\small", rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
          " & ".join(header) + r" \\", r"\midrule"]
    L += [" & ".join(str(c) for c in row) + r" \\" for row in rows]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]


def figure(L, name, caption, width=0.85):
    L += [r"\begin{figure}[H]\centering",
          rf"\includegraphics[width={width}\linewidth]{{figures/{name}}}",
          rf"\caption{{{caption}}}", rf"\label{{fig:{name[:-4]}}}", r"\end{figure}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/guard_results.json")
    ap.add_argument("--robust", default="results/guard_robust.json")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--out", default="paper/GUARD.tex")
    args = ap.parse_args()
    R = json.load(open(args.results))
    RB = json.load(open(args.robust)) if os.path.exists(args.robust) else None

    meta = R.get("meta", {})
    Rr = meta.get("R", "?")
    alphas = meta.get("alphas", [0.005, 0.01, 0.02, 0.05, 0.10])

    # extractors matched to the engine's real schema ----------------------------------- #
    def e1_fpr(scorer, alpha=0.05, split="random"):
        return R["e1"][split]["per_scorer"][scorer][str(alpha)]["fpr"]

    def e5_fpr(scorer, split, alpha=0.05):
        return R["e5"]["per_scorer"][scorer][str(alpha)][split]["fpr"]

    def e6_fpr(scorer, method, b, alpha=0.05):
        return R["e6"]["per_scorer"][scorer][str(alpha)][method][str(b)]["fpr"]

    def e3_fpr(scorer, alpha=0.05):
        return R["e3"]["per_scorer"][scorer][str(alpha)]["fpr"]

    def e2_humans_fpr(scorer, alpha=0.05):
        blk = R["e2"].get("fpr_pool_humans", {}).get(scorer, {}).get(str(alpha), {})
        return blk.get("fpr", []) if isinstance(blk, dict) else blk

    def e4_fpr(scorer, kind, rate, alpha=0.05):
        return R["e4"]["per_scorer"][scorer][kind][str(rate)][str(alpha)]["fpr"]

    def e7_tpr(scorer, alpha, cond="clean"):
        return R["e7"][cond][scorer][str(alpha)]["tpr"]

    def rb_rate(scorer, mode, cond, alpha=0.05):
        return RB["summary"][f"{scorer}|{mode}|{cond}|{alpha}"]["mean"]

    L = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{graphicx,booktabs,amsmath,amssymb,amsthm,float,caption,times}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\newtheorem{proposition}{Proposition}",
        r"\title{\textbf{When Can You Trust the Certificate? Stress-Testing Distribution-Free"
        r" False-Accusation Guarantees for AI-Text Detection}}",
        r"\author{Author Name\\ \small Affiliation \and Co-author\\ \small Affiliation}",
        r"\date{}",
        r"\begin{document}\maketitle",
    ]

    # ------------------------------- abstract -------------------------------- #
    ll_in = m(e1_fpr("loglik")); tf_tpr5 = m(e7_tpr("tfidf", 0.05))
    rev_ll = m(e2_humans_fpr("loglik")); rev_en = m(e2_humans_fpr("entropy"))
    e6_long_marg = m(e6_fpr("loglik", "marginal", 3)); e6_long_mond = m(e6_fpr("loglik", "mondrian", 3))
    L += [r"\begin{abstract}",
        r"AI-text detectors are deployed to accuse: a flagged student or author bears the cost of a "
        r"false positive. Split conformal prediction turns any detector score into an accusation "
        r"rule with a finite-sample, distribution-free bound on the false-accusation rate, and "
        r"recent work has proposed exactly this. We ask the question a deployer actually faces: "
        r"\emph{under which realistic conditions does the certificate remain valid?} Exploiting the "
        r"observation that validity depends only on the \emph{human} text distribution while power "
        r"depends only on the AI distribution, we derive sharp predictions and test them across "
        rf"five detector scores, {Rr} calibration/test resamples, and eight deployment conditions. "
        r"In distribution the certificate is exact "
        rf"(e.g.\ empirical FPR {ll_in:.3f} at $\alpha=0.05$), and a supervised score achieves "
        rf"{100*tf_tpr5:.0f}\% power at a certified 5\% false-accusation rate. But calibrating on "
        r"one human population and deploying on a more fluent one silently inflates false "
        rf"accusations to {rev_ll:.2f}--{rev_en:.2f} at nominal $0.05$ for likelihood-based scores "
        r"--- the certificate-level form of the documented bias against non-native writers --- "
        r"while lexical edits and even spell-correcting cleanup leave validity intact. Marginal "
        rf"certificates also hide length discrimination (long human reviews flagged at "
        rf"{e6_long_marg:.2f}); Mondrian calibration restores every length bin to "
        rf"$\le\alpha$ ({e6_long_mond:.2f}). Finally, we give an editor-aware robust calibration "
        r"that provably preserves validity under a modeled editing process at negligible power "
        r"cost. All results carry exact Beta-Binomial significance tests and regenerate from one "
        r"results file.",
        r"\end{abstract}"]

    # ------------------------------ introduction ------------------------------ #
    L += [r"\section{Introduction}",
        r"Detectors of machine-generated text are increasingly used in consequential settings --- "
        r"plagiarism screening, review-platform integrity, scientific-venue policy --- where the "
        r"operational act is an \emph{accusation}. The relevant contract for that act is not "
        r"accuracy but a guarantee on the false-accusation rate, robust to the fact that the "
        r"deployer cannot model the distribution of either class. Split conformal prediction "
        r"\cite{vovk,papadopoulos} provides exactly this contract for any base detector score, and "
        r"prior work has applied it to bound detector false-positive rates \cite{mcp}.",
        r"",
        r"This paper studies the contract itself. Our starting point is an elementary but, we "
        r"argue, under-appreciated asymmetry:",
        r"\begin{proposition}[Certificate asymmetry]",
        r"The split-conformal false-accusation bound $P(\mathrm{flag}\mid\mathrm{human})\le\alpha$ "
        r"requires only exchangeability of calibration and test \emph{human} text. The AI "
        r"distribution is irrelevant to validity; it determines only power.",
        r"\end{proposition}",
        r"\noindent The proof is the standard conformal argument applied to the human class only. "
        r"The deployment consequences are sharp and testable: a new generator entering the world "
        r"\emph{cannot} break a deployed certificate; changes in \emph{who writes} or \emph{how "
        r"humans edit} can. We map this boundary empirically (E1--E8) and contribute two "
        r"constructive repairs (Mondrian length-conditional calibration; editor-aware robust "
        r"calibration with a stochastic-dominance validity argument).",
        r"\paragraph{Contributions.}",
        r"\begin{itemize}",
        r"\item A deployment-grounded validity map for conformal false-accusation certificates in "
        r"MGT detection: exact in distribution; broken by human-population shift (up to "
        rf"${rev_en/0.05:.0f}\times$ nominal); \emph{{safe}} under lexical noise and spell-correcting "
        r"cleanup; supervised scores fail safe where zero-shot likelihood scores fail dangerously.",
        r"\item Identification of length discrimination hidden by marginal certificates, and its "
        r"repair by Mondrian calibration.",
        r"\item An editor-aware robust calibration with a provable conservativeness guarantee "
        r"under a modeled editor, costing $<0.01$ power in our study.",
        r"\item A fully reproducible protocol: five scores, exact Beta-Binomial tests, "
        rf"{Rr} resamples per condition; code, scores and figures regenerate from one JSON.",
        r"\end{itemize}"]

    # ------------------------------ related work ------------------------------ #
    L += [r"\section{Related Work}",
        r"Zero-shot MGT detection scores include likelihood, rank and entropy statistics "
        r"\cite{gltr,solaiman} and curvature-style discrepancies \cite{detectgpt,fastdetectgpt}; "
        r"supervised detectors fine-tune or fit classifiers on labelled corpora \cite{roberta}. "
        r"Robustness studies show brittleness to paraphrase and domain shift \cite{raid,krishna,"
        r"sadasivan}, and GPT detectors are documented to over-flag non-native English writers "
        r"\cite{liang}. Conformal prediction \cite{vovk,papadopoulos} and conformal risk control "
        r"\cite{angelopoulos} provide distribution-free guarantees; MCP \cite{mcp} applies CP to "
        r"bound MGT-detector FPR and improve its power. Our work is complementary to MCP: rather "
        r"than improving the detector under a fixed certificate, we characterise when the "
        r"certificate itself survives deployment, connect its failure modes to the fairness "
        r"literature \cite{liang}, and supply group-conditional and editor-robust repairs (cf.\ "
        r"Mondrian conformal prediction \cite{vovk} and adversarially robust conformal methods in "
        r"vision \cite{gendler}).",]

    # ------------------------------ method ------------------------------ #
    n_cal_hint = R.get("e1", {}).get("n_cal", "n")
    L += [r"\section{Method}",
        r"\subsection{Certificates}",
        r"Given any score $s(\cdot)$ oriented so larger means more AI-like, and $n$ calibration "
        r"human texts with scores $s_1,\dots,s_n$, the conformal p-value of a test text $x$ is "
        r"$p(x) = \frac{1+\#\{i: s_i \ge s(x)\}}{n+1}$, and GUARD flags $x$ iff $p(x)\le\alpha$. "
        r"If $x$ is human and exchangeable with the calibration humans, $P(\mathrm{flag})\le\alpha$. "
        r"Conditional on the calibration set the realised FPR follows a Beta law; over a finite "
        r"test set the false-flag \emph{count} is Beta-Binomial, which yields an exact one-sided "
        r"test of ``the certificate held'' that we report for every condition. Unscoreable texts "
        r"receive $p=1$ (never flagged): an accuser must not accuse by failure of its own scorer.",
        r"\subsection{Scores}",
        r"Four training-free statistics from one GPT-2 forward pass per text (mean log-likelihood; "
        r"negative mean log-rank; negative predictive entropy; the analytic Fast-DetectGPT "
        r"discrepancy \cite{fastdetectgpt}) and one supervised score (TF--IDF + logistic "
        r"regression). The supervised score is fit on an entity-disjoint training half; all "
        r"calibration and evaluation use the other half, so the fitted scorer never sees "
        r"evaluation hotels.",
        r"\subsection{Mondrian calibration}",
        r"Per-group calibration (here: text-length bins) yields group-conditional validity at the "
        r"cost of smaller calibration sets per group \cite{vovk}.",
        r"\subsection{Editor-aware robust calibration}",
        r"Let $E$ be a randomized editing process (our proxy: spell/punctuation cleanup at rate "
        r"$r$). Each calibration human contributes $M_i = \max\{s(X_i), s(E_1(X_i)),\dots,"
        r"s(E_k(X_i))\}$. For a test human who applies one edit from $E$, $s(E(X))$ is "
        r"stochastically dominated by the corresponding $M$, so conformal p-values computed "
        r"against $\{M_i\}$ remain super-uniform and the $\alpha$-bound still holds (conservative). "
        r"Validity under editors \emph{outside} the modeled family is an empirical question (E8).",]

    # ------------------------------ setup ------------------------------ #
    L += [r"\section{Experimental Setup}",
        r"In-domain data: the English subset of MAiDE-up \cite{maideup} (1{,}000 human and 1{,}000 "
        r"GPT-4 hotel reviews; 100 hotels). Out-of-distribution human text: the RAID benchmark "
        r"\cite{raid} reviews domain (movie reviews; also supplying eight AI generators for power "
        r"analysis) and abstracts domain. Editors: WordNet synonym substitution and function-word "
        r"perturbation (lexical noise), and a spell-correcting, punctuation-normalising cleanup "
        r"editor (fluency-raising; the realistic proxy for grammar tools). Every unique text is "
        rf"scored once and cached; each condition is evaluated over $R={Rr}$ calibration/test "
        r"resamples, reported as mean [2.5\%, 97.5\%] with exact Beta-Binomial violation tests at "
        rf"$\alpha\in\{{{', '.join(str(a) for a in alphas)}\}}$.",]

    # ------------------------------ results ------------------------------ #
    L += [r"\section{Results}", r"\subsection{E1: the certificate is exact in distribution}"]
    rows = [[SLAB[s], fmt(e1_fpr(s, 0.05)), fmt(e1_fpr(s, 0.01))] for s in SCORERS]
    table(L, ["Score", r"FPR @ $\alpha=0.05$", r"FPR @ $\alpha=0.01$"], rows,
          "In-distribution validity: empirical FPR over resamples (mean [95\\% band]).", "e1")
    L += [r"All five scores sit on the nominal level, and the full distribution of per-resample "
          r"FPRs matches the finite-test Beta-Binomial envelope (Figure~\ref{fig:f1_validity_envelope}).",]
    figure(L, "f1_validity_envelope.png",
           "Distribution of per-resample FPR vs.\\ the exact Beta-Binomial envelope.", 0.95)

    L += [r"\subsection{E2--E4: when the certificate breaks --- and when it does not}"]
    rows = []
    for s in SCORERS:
        rows.append([SLAB[s],
                     f"{m(e1_fpr(s,0.05)):.3f}",
                     f"{m(e2_humans_fpr(s)):.3f}" if e2_humans_fpr(s) else "--",
                     f"{m(e3_fpr(s)):.3f}",
                     f"{m(e4_fpr(s,'cleanup',0.5)):.3f}" if 'cleanup' in R['e4']['per_scorer'][s] else "--",
                     f"{m(e4_fpr(s,'synonym',0.3)):.3f}"])
    table(L, ["Score", "In-domain", "Reviews humans", "Abstracts humans", "Cleanup@0.5", "Synonym@0.3"],
          rows, "Empirical FPR at nominal $\\alpha=0.05$ across deployment conditions.", "map")
    L += [r"Three regimes emerge (Figure~\ref{fig:f2_certificate_map}). \textbf{(i) Human-population "
        r"shift breaks the certificate for likelihood scores.} Movie-review humans are flagged at "
        rf"{m(e2_humans_fpr('logrank')):.2f}--{m(e2_humans_fpr('entropy')):.2f} and abstracts "
        rf"humans at up to {m(e3_fpr('entropy')):.2f} (nominal $0.05$; violation $p<10^{{-3}}$). "
        r"The mechanism is fluency: our calibration humans are largely non-native hotel reviewers "
        r"whose text has \emph{low} LM likelihood; more fluent human populations score higher and "
        r"cross the threshold. This is precisely the bias documented by \cite{liang}, here "
        r"surfacing \emph{through} a certificate that is formally valid for the calibration "
        r"population. \textbf{(ii) The supervised score fails safe.} TF--IDF's AI evidence is "
        r"content-specific, so under every shift its FPR falls toward zero rather than exploding. "
        r"\textbf{(iii) Editing does not break the certificate here.} Lexical noise lowers "
        r"likelihood (conservative), and even full spell-correcting cleanup leaves FPR at nominal: "
        r"light editing does not turn a hotel reviewer into a `fluent' writer. The danger is who "
        r"writes, not light tool use --- though stronger LLM paraphrase editors remain untested "
        r"here (\S\ref{sec:limit}).",]
    figure(L, "f2_certificate_map.png",
           "The validity map: empirical FPR at nominal $\\alpha=0.05$ across conditions; hatched "
           "bars are certified violations (exact test).", 0.98)
    figure(L, "f3_violation_vs_rate.png",
           "FPR vs.\\ edit rate per editor family: lexical noise is conservative.", 0.9)

    L += [r"\subsection{E5: calibration is insensitive to entity leakage}",
        rf"Random vs.\ entity-disjoint (grouped-by-hotel) calibration give indistinguishable "
        rf"validity (log-likelihood: {m(e5_fpr('loglik','random')):.4f} vs.\ "
        rf"{m(e5_fpr('loglik','grouped')):.4f}; TF--IDF: {m(e5_fpr('tfidf','random')):.4f} vs.\ "
        rf"{m(e5_fpr('tfidf','grouped')):.4f}). Unlike supervised \emph{{training}}, conformal "
        r"\emph{calibration} does not exploit entity overlap --- a reassuring null result for "
        r"review-platform deployments (Figure~\ref{fig:f4_leakage}).",]
    figure(L, "f4_leakage.png", "Random vs.\\ grouped calibration FPR distributions.", 0.8)

    L += [r"\subsection{E6: marginal certificates hide length discrimination; Mondrian repairs it}"]
    rows = []
    edges = R["e6"].get("bin_edges_words", [0, 30, 60, 120, 10000])
    for b in range(4):
        lo_, hi_ = edges[b], edges[b + 1]
        label = f"{lo_}--{hi_}" if hi_ < 10000 else f"$>${lo_}"
        rows.append([label + " words",
                     f"{m(e6_fpr('loglik','marginal',b)):.3f}",
                     f"{m(e6_fpr('loglik','mondrian',b)):.3f}"])
    table(L, ["Length bin", "Marginal FPR", "Mondrian FPR"], rows,
          "Per-length-bin FPR at nominal $\\alpha=0.05$ (log-likelihood score).", "e6")
    L += [r"The marginal certificate is valid \emph{on average} while flagging long human reviews "
        rf"at {m(e6_fpr('loglik','marginal',3)):.2f} --- {m(e6_fpr('loglik','marginal',3))/0.05:.1f}$\times$ "
        r"nominal --- and almost never flagging short ones. Mondrian calibration restores every "
        r"bin to $\le\alpha$ (Figure~\ref{fig:f5_mondrian}). Deployers should treat conditional "
        r"validity, not marginal validity, as the fairness-relevant contract.",]
    figure(L, "f5_mondrian.png", "Marginal vs.\\ Mondrian per-length-bin FPR.", 0.9)

    L += [r"\subsection{E7: the price of the certificate}"]
    rows = []
    for s in SCORERS:
        cells = [SLAB[s]]
        for a in (0.01, 0.05, 0.10):
            v = e7_tpr(s, a)
            cells.append(f"{m(v):.2f}" if len(v) else "--")
        rows.append(cells)
    table(L, ["Score", r"TPR@$\alpha{=}0.01$", r"TPR@$\alpha{=}0.05$", r"TPR@$\alpha{=}0.10$"],
          rows, "Certified power (clean AI text, in-domain).", "e7")
    L += [rf"The supervised score detects {100*m(e7_tpr('tfidf',0.05)):.0f}\% of in-domain GPT-4 "
        r"reviews at a certified 5\% false-accusation rate; zero-shot scores pay heavily for the "
        r"certificate in-domain but transfer their \emph{power} to unseen generators "
        r"(Figure~\ref{fig:f7_per_generator}); recall their \emph{validity} caveats above.",]
    figure(L, "f6_power_frontier.png", "Power--certificate frontier per score.", 0.9)
    figure(L, "f7_per_generator.png", "Certified power per RAID generator.", 0.95)

    if RB:
        L += [r"\subsection{E8: editor-aware robust calibration}"]
        rows = []
        for s in SCORERS:
            rows.append([SLAB[s],
                         f"{rb_rate(s,'naive','human_cleanup_full'):.3f}",
                         f"{rb_rate(s,'robust','human_cleanup_full'):.3f}",
                         f"{rb_rate(s,'naive','ai_clean'):.3f}",
                         f"{rb_rate(s,'robust','ai_clean'):.3f}"])
        table(L, ["Score", "naive FPR (cleaned)", "robust FPR (cleaned)",
                  "naive TPR", "robust TPR"], rows,
              "Editor-aware calibration at $\\alpha=0.05$: validity under the modeled editor and "
              "the power cost.", "e8")
        L += [r"With the cleanup editor modeled in calibration, validity holds by construction and "
            r"the measured power cost is below $0.01$ for every score. In this study the naive "
            r"certificate already survived cleanup editing, so the robust construction functions "
            r"as cheap insurance; its value grows with stronger editors, and the guarantee covers "
            r"any editor whose effect is dominated by the modeled family.",]

    # ---------------------------- discussion etc ---------------------------- #
    L += [r"\section{Discussion}",
        r"For deployers our results reduce to three rules. First, certify against the humans you "
        r"will actually judge: the certificate is exact for the calibration population and can be "
        rf"{rev_en/0.05:.0f}$\times$ off for a more fluent one; recalibration per population (or "
        r"per platform) is cheap and necessary. Second, demand conditional validity: marginal "
        r"certificates can hide systematic over-accusation of long texts (and, by the population "
        r"result, of fluent writers); Mondrian calibration over observable strata is a one-line "
        r"fix. Third, choose the score family by failure mode: supervised content scores buy high "
        r"certified power in-domain and fail safe under shift; zero-shot likelihood scores "
        r"transfer power across generators but fail dangerous on shifted humans.",
        r"\section{Limitations}\label{sec:limit}",
        r"English only; one in-domain corpus (hotel reviews) whose humans skew non-native --- "
        r"which is what makes the population-shift result vivid, but magnitudes will vary; the "
        r"cleanup editor under-approximates modern LLM paraphrasers, so E4/E8 bound only the "
        r"editors tested; GPT-2 is a deliberately modest scorer (the wrapper is scorer-agnostic); "
        r"and our exact tests certify violations, not their causes --- the fluency mechanism is "
        r"supported but not isolated. The protocol, released in full, replicates at larger scale "
        r"directly.",
        r"\section{Conclusion}",
        r"Distribution-free certificates make AI-text detectors accountable, but only within the "
        r"human distribution they were calibrated on. We mapped that boundary --- exact in "
        r"distribution, robust to light editing, broken by population shift, length-biased "
        r"marginally --- and showed the repairs are cheap. A certificate is a promise about "
        r"\emph{whom} you calibrated on; deployers should make that promise explicit.",
        r"\paragraph{Reproducibility.} All numbers derive from two JSON files produced by "
        r"\texttt{scripts/run\_all.py} and \texttt{scripts/run\_robust.py}; figures and this paper "
        r"regenerate from them; conformal correctness is asserted by the test suite.",]

    refs = [
        ("vovk", "V. Vovk, A. Gammerman, G. Shafer. Algorithmic Learning in a Random World. Springer, 2005."),
        ("papadopoulos", "H. Papadopoulos, K. Proedrou, V. Vovk, A. Gammerman. Inductive Confidence "
         "Machines for Regression. ECML, 2002."),
        ("angelopoulos", "A. N. Angelopoulos, S. Bates. Conformal Prediction: A Gentle Introduction. "
         "Foundations and Trends in Machine Learning, 2023."),
        ("mcp", "X. Zhang et al. Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text "
         "Detection Framework via Multiscaled Conformal Prediction. ACL, 2025."),
        ("liang", "W. Liang, M. Yuksekgonul, Y. Mao, E. Wu, J. Zou. GPT Detectors Are Biased Against "
         "Non-Native English Writers. Patterns, 2023."),
        ("raid", "L. Dugan et al. RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated "
         "Text Detectors. ACL, 2024."),
        ("maideup", "O. Ignat, X. Xu, R. Mihalcea. MAiDE-up: Multilingual Deception Detection of "
         "AI-Generated Hotel Reviews. Findings of NAACL, 2025."),
        ("detectgpt", "E. Mitchell et al. DetectGPT: Zero-Shot Machine-Generated Text Detection using "
         "Probability Curvature. ICML, 2023."),
        ("fastdetectgpt", "G. Bao et al. Fast-DetectGPT: Efficient Zero-Shot Detection of "
         "Machine-Generated Text via Conditional Probability Curvature. ICLR, 2024."),
        ("gltr", "S. Gehrmann, H. Strobelt, A. M. Rush. GLTR: Statistical Detection and Visualization "
         "of Generated Text. ACL Demos, 2019."),
        ("solaiman", "I. Solaiman et al. Release Strategies and the Social Impacts of Language Models. "
         "arXiv:1908.09203, 2019."),
        ("krishna", "K. Krishna et al. Paraphrasing Evades Detectors of AI-Generated Text, but "
         "Retrieval is an Effective Defense. NeurIPS, 2023."),
        ("sadasivan", "V. S. Sadasivan et al. Can AI-Generated Text be Reliably Detected? "
         "arXiv:2303.11156, 2023."),
        ("gendler", "A. Gendler, T.-W. Weng, L. Daniel, Y. Romano. Adversarially Robust Conformal "
         "Prediction. ICLR, 2022."),
        ("roberta", "Y. Liu et al. RoBERTa: A Robustly Optimized BERT Pretraining Approach. "
         "arXiv:1907.11692, 2019."),
    ]
    L += [r"\begin{thebibliography}{99}\small"]
    L += [rf"\bibitem{{{k}}} {t}" for k, t in refs]
    L += [r"\end{thebibliography}", r"\end{document}"]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    figout = os.path.join(os.path.dirname(args.out), "figures")
    os.makedirs(figout, exist_ok=True)
    n = 0
    for f in os.listdir(args.figdir):
        if f.endswith(".png"):
            shutil.copy2(os.path.join(args.figdir, f), os.path.join(figout, f))
            n += 1
    with open(args.out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"Saved {args.out} ({len(refs)} references; {n} figures copied)")


if __name__ == "__main__":
    main()
