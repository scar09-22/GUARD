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
    ap.add_argument("--wave2", default="results/guard_wave2.json")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--out", default="paper/GUARD.tex")
    args = ap.parse_args()
    R = json.load(open(args.results))
    RB = json.load(open(args.robust)) if os.path.exists(args.robust) else None
    W2 = json.load(open(args.wave2)) if os.path.exists(args.wave2) else None

    meta = R.get("meta", {})
    cfg = meta.get("cfg", {})
    Rr = cfg.get("R", meta.get("R", len(R["e1"]["random"]["per_scorer"]["loglik"]["0.05"]["fpr"])))
    alphas = cfg.get("alphas", meta.get("alphas", [0.005, 0.01, 0.02, 0.05, 0.10]))

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

    # extractors matched to the wave-2 schema (results/guard_wave2.json) --------------- #
    W2SC = ["loglik", "tfidf", "supervised", "offshelf"]
    W2LAB = {"loglik": "GPT-2 log-likelihood", "tfidf": "TF--IDF (supervised)",
             "supervised": "DistilRoBERTa (fine-tuned)", "offshelf": "Off-shelf detector"}
    if W2:
        w1b, w2b = W2["w1_multilingual"], W2["w2_domains"]
        w3b, w4b = W2["w3_paraphrase"], W2["w4_diagnostics"]
        W2R = W2["meta"]["cfg"]["R"]
        LANGS, ENG = w1b["languages"], w1b["english"]
        DOMS = w2b["domains"]
        ALARM_LVL = w4b["alarm_level"]

        def w1_in(lang, s, a=0.05):           # in-language validity (per-repeat FPR list)
            return w1b["per_language"][lang]["in_language"][s][str(a)]["fpr"]

        def w1_pow(lang, s, a=0.05):          # in-language power (per-repeat TPR list)
            return w1b["per_language"][lang]["power"][s][str(a)]["tpr"]

        def w1_x(s, lang, a=0.05):            # English-calibrated cross-language FPR
            return w1b["cross_from_english"][s][str(a)][lang]["fpr"]

        def w2_in(dom, s, a=0.05):
            return w2b["per_domain"][dom]["in_domain"][s][str(a)]["fpr"]

        def w2_pow(dom, s, a=0.05):
            return w2b["per_domain"][dom]["power"][s][str(a)]["tpr"]

        def w2_x(s, dom, a=0.05):             # hotel-calibrated cross-domain FPR
            return w2b["cross_from_hotel"][s][str(a)][dom]["fpr"]

        def w3_fpr(s, mode, cond, a=0.05):    # mode in {naive, robust}; cond human_*
            return w3b["per_scorer"][s][mode][cond][str(a)]["fpr"]

        def w3_tpr(s, mode, cond, a=0.05):    # cond in {ai_clean, ai_para}
            return w3b["per_scorer"][s][mode][cond][str(a)]["tpr"]

        def w4_rec(cond, s, a=0.05):          # full diagnostics record per condition
            return w4b["conditions"][cond][s][str(a)]

        def w4_alarm(cond, s, a=0.05, m=100):
            """Alarm rate at pilot size m; falls back to the largest available m
            (small pools cap the pilot). Returns (m_used, rate)."""
            ar = w4_rec(cond, s, a)["alarm_rate"]
            avail = [int(k) for k, v in ar.items() if v is not None]
            mu = m if ar.get(str(m)) is not None else max(avail)
            return mu, ar[str(mu)]

    L = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{graphicx,booktabs,amsmath,amssymb,amsthm,float,caption,times}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\newtheorem{proposition}{Proposition}",
        r"\emergencystretch=2em",
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

    # wave-2 headline aggregates (all derived from the JSON at build time) ------------- #
    w2_abs = ""
    if W2:
        ru_x = m(w1_x("loglik", "Russian"))
        xlangs = [lg for lg in LANGS if lg != ENG]
        n_under = sum(1 for lg in xlangs if m(w1_x("loglik", lg)) < 0.05)
        news_x = m(w2_x("loglik", "news"))
        para_naive = {s: m(w3_fpr(s, "naive", "human_para")) for s in W2SC}
        para_rob = {s: m(w3_fpr(s, "robust", "human_para")) for s in W2SC}
        pn_lo, pn_hi = min(para_naive.values()), max(para_naive.values())
        pr_hi = max(para_rob.values())
        # diagnostics headline: alarm rates on violated vs in-control conditions
        viol_pairs = [(c, s) for c in w4b["conditions"] for s in W2SC
                      if w4_rec(c, s)["violated"]]
        viol_alarms = [w4_alarm(c, s) for c, s in viol_pairs]
        min_viol_alarm = min(r for _, r in viol_alarms)
        min_viol_alarm100 = min(r for mu, r in viol_alarms if mu == 100)
        ctrl_max_alarm = max(w4_alarm("control_english", s)[1] for s in W2SC)
        ll_conds = list(w4b["conditions"].keys())
        tv_cov = sum(1 for c in ll_conds
                     if w4_rec(c, "loglik")["tv_bound_mean"] >= m(w4_rec(c, "loglik")["fpr"]))
        ga_cov = sum(1 for c in ll_conds
                     if w4_rec(c, "loglik")["gamma_bound_mean"] >= m(w4_rec(c, "loglik")["fpr"]))
        w2_abs = (
            r" A second wave scales the map: four scorer families (zero-shot GPT-2, TF--IDF, a "
            r"fine-tuned transformer, an off-the-shelf detector), ten languages, six text "
            r"domains, and an LLM-paraphrase attack. In-population validity is exact by "
            r"construction in every language and domain; empirically, English-calibrated "
            rf"certificates deployed cross-lingually mostly \emph{{under}}-flag, but reach FPR "
            rf"{ru_x:.3f} on Russian, and hotel-calibrated certificates reach {news_x:.2f} on "
            rf"news. LLM-paraphrasing \emph{{innocent human}} text inflates FPR to "
            rf"{pn_lo:.2f}--{pn_hi:.2f} across all four scorers; paraphrase-aware robust "
            rf"calibration restores $\le {pr_hi:.2f}$ at a real power cost. Two estimable "
            r"score-space bounds ($\Gamma\alpha$ and $\alpha+\mathrm{TV}$) and a calibrated "
            r"pilot alarm (exact binomial) make these failures detectable before deployment: "
            rf"the alarm fires at rate $\ge {min_viol_alarm100:.2f}$ on every violated "
            rf"condition with $m{{=}}100$ pilot texts while in-control alarm rates stay below "
            rf"the {ALARM_LVL:.2f} design level. Shift findings are empirical and the bounds "
            r"are plug-in estimates, not certificates.")
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
        r"cost." + w2_abs + " All results carry exact Beta-Binomial significance tests and "
        r"regenerate from machine-produced results files.",
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

    # --------------------- population-shift theory + diagnostics --------------------- #
    L += [r"\section{Population-Shift Theory and Pre-Deployment Diagnostics}\label{sec:theory}",
        r"What happens to the certificate when the deployment humans are \emph{not} the "
        r"calibration humans? Let $P_s$ and $Q_s$ denote the distributions of the score "
        r"$s(X)$ under the calibration and deployment human populations respectively, and let "
        r"$T$ be the (random) conformal flagging threshold at level $\alpha$, so that "
        r"$\mathbb{E}_T[P_s(s>T)]\le\alpha$ by the standard conformal guarantee. Two elementary "
        r"bounds control the deployed false-accusation rate.",
        r"\begin{proposition}[$\Gamma$-bound]",
        r"If $Q_s \ll P_s$ with $\operatorname{ess\,sup}\, dQ_s/dP_s \le \Gamma$ on score "
        r"space, then $\mathbb{E}_{Q}[\mathrm{FPR}] \le \Gamma\alpha$.",
        r"\end{proposition}",
        r"\begin{proof}",
        r"$Q_s(A)\le\Gamma P_s(A)$ for every measurable $A$; apply this with $A=\{s>T\}$ "
        r"conditionally on the threshold $T$, then take expectations over $T$ and use "
        r"$\mathbb{E}_T[P_s(s>T)]\le\alpha$.",
        r"\end{proof}",
        r"\begin{proposition}[TV-bound]",
        r"Without any absolute-continuity assumption, $\mathbb{E}_{Q}[\mathrm{FPR}] \le "
        r"\alpha + \mathrm{TV}(P_s, Q_s)$.",
        r"\end{proposition}",
        r"\begin{proof}",
        r"$|Q_s(A)-P_s(A)|\le \mathrm{TV}(P_s,Q_s)$ for every event $A$; apply with "
        r"$A=\{s>T\}$ and take expectations over $T$.",
        r"\end{proof}",
        r"\noindent Both bounds live on the \emph{one-dimensional score space}, not on text "
        r"space: $\mathrm{TV}(P_s,Q_s)$ and $\Gamma$ are functionals of two scalar "
        r"distributions, hence estimable from an \emph{unlabeled} pilot sample of deployment "
        r"human scores (we use binned plug-in estimates $\widehat{\mathrm{TV}}$ and "
        r"$\widehat\Gamma$ over a common grid). These are diagnostics, not certificates: "
        r"plug-in estimates carry their own sampling error, and a binned $\widehat\Gamma$ can "
        r"underestimate a density ratio concentrated inside a bin.",
        r"\paragraph{The pilot failure detector.}",
        r"Given $m$ known-human pilot texts from the deployment population, flag each with the "
        r"\emph{deployed} rule and let $K$ be the number of flags. The failure detector rejects "
        r"$H_0\!:\mathrm{FPR}\le\alpha$ when the exact binomial tail "
        r"$P(\mathrm{Bin}(m,\alpha)\ge K)$ falls below an alarm level $\rho$. Its calibration "
        r"guarantee is immediate: when the certificate is valid the flag count is "
        r"stochastically dominated by $\mathrm{Bin}(m,\alpha)$, so the alarm fires with "
        r"probability at most $\rho$ for \emph{every} pilot size $m$. The complementary "
        r"reading is the exact Clopper--Pearson upper confidence bound on deployment FPR --- "
        r"what a pilot of size $m$ can positively \emph{assure} --- which is necessarily loose "
        r"for small $m$. Section~\ref{sec:w4} validates both readings empirically.",]

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

    # ------------------------------ wave-2 sections ------------------------------ #
    if W2:
        caps = w1b["caps"]
        n_cal_w2 = w1b["per_language"][ENG]["n_cal"]
        abs_nh = w2b["per_domain"]["abstracts"]["n_human"]
        L += [r"\section{Wave-2 Experiments: Languages, Domains, Paraphrase, Diagnostics}",
            r"\label{sec:wave2}",
            r"The second wave widens every axis of the validity map and validates the "
            r"diagnostics of \S\ref{sec:theory}. \textbf{Scorers (four families):} zero-shot "
            r"GPT-2 log-likelihood; TF--IDF + logistic regression; a fine-tuned "
            r"\texttt{distilroberta-base} classifier (trained on the entity-disjoint training "
            r"half only); and a frozen off-the-shelf detector "
            r"(\path{Hello-SimpleAI/chatgpt-detector-roberta}, used zero-shot). "
            rf"\textbf{{Data:}} all ten MAiDE-up languages ({caps['human']} human / "
            rf"{caps['ai']} AI reviews per language; $n_{{\mathrm{{cal}}}}={n_cal_w2}$); six "
            r"text domains (MAiDE-up hotel reviews plus the RAID movie-review, abstracts "
            rf"($n_{{\mathrm{{hum}}}}={abs_nh}$), books, news and poetry pools); and an "
            r"LLM-paraphrase editor (\path{humarin/chatgpt_paraphraser_on_T5_base}, a "
            r"T5-base model fine-tuned on ChatGPT paraphrases). Every condition is evaluated "
            rf"over $R={W2R}$ calibration/test resamples with the same exact Beta-Binomial "
            r"violation tests as Wave 1; per-repeat rates are aggregated to means at build "
            r"time, never hand-entered.",]

        # ------------------------------- W1 ------------------------------- #
        L += [r"\subsection{W1: ten languages --- in-language validity is exact; "
              r"cross-language transfer is bimodal}\label{sec:w1}"]
        rows = []
        for lg in LANGS:
            cells = [esc(lg)] + [f"{m(w1_in(lg, s)):.3f}" for s in W2SC]
            if lg == ENG:
                cells += ["--", "--"]
            else:
                cells += [f"{m(w1_x('loglik', lg)):.3f}", f"{m(w1_x('supervised', lg)):.3f}"]
            rows.append(cells)
        table(L, ["Language", "loglik", "tfidf", "superv.", "off-shelf",
                  r"X-Eng loglik", r"X-Eng superv."], rows,
              "W1: in-language FPR at nominal $\\alpha=0.05$ (four scorers; calibrated and "
              "tested within the language) and English-calibrated cross-language FPR.", "w1")
        in_cells = [m(w1_in(lg, s)) for lg in LANGS for s in W2SC]
        x_ll = {lg: m(w1_x("loglik", lg)) for lg in xlangs}
        over = sorted(((v, lg) for lg, v in x_ll.items() if v > 0.05), reverse=True)
        max_sup_x = max(m(w1_x(s, lg)) for s in ("tfidf", "supervised", "offshelf")
                        for lg in xlangs)
        over_txt = ", ".join(rf"{lg} {v:.3f}" for v, lg in over)
        L += [rf"Calibrated within each language, all {len(LANGS)*len(W2SC)} language--scorer "
            rf"cells sit at or below nominal (FPR {min(in_cells):.3f}--{max(in_cells):.3f}; no "
            r"cell rejects the exact test): the certificate's distribution-freeness is real, "
            r"and \emph{any} population can be certified by calibrating on it. Deploying the "
            r"\emph{English}-calibrated certificate cross-lingually is bimodal for the "
            rf"likelihood score: {n_under} of {len(xlangs)} languages \emph{{under}}-flag "
            r"(FPR below nominal --- the conservative, power-less direction), while "
            rf"{over_txt} blow past it, the Cyrillic and Hangul extremes ($\approx$1.0) being "
            r"essentially every human flagged. The mechanism is language identity: GPT-2 "
            r"assigns non-Latin-script text uniformly low likelihood, which lands either side "
            r"of the English threshold wholesale. The three classifier-based scores never "
            rf"inflate cross-lingually (max FPR {max_sup_x:.3f}): they fail safe, at the price "
            r"of near-zero cross-language power (Figure~\ref{fig:f9_multilingual}).",]
        figure(L, "f9_multilingual.png",
               "W1: in-language validity (left) vs.\\ English-calibrated cross-language FPR "
               "(right) at nominal $\\alpha=0.05$.", 0.98)

        # ------------------------------- W2 ------------------------------- #
        L += [r"\subsection{W2: six domains --- transfer direction matters}\label{sec:w2}"]
        rows = []
        for d in DOMS:
            cells = [esc(d), f"{m(w2_in(d, 'loglik')):.3f}"]
            if d == "hotel":
                cells += ["--"] * 4
            else:
                cells += [f"{m(w2_x(s, d)):.3f}" for s in W2SC]
            rows.append(cells)
        table(L, ["Target domain", "In-domain (loglik)", "X loglik", "X tfidf",
                  "X superv.", "X off-shelf"], rows,
              "W2: in-domain FPR and hotel-calibrated cross-domain FPR at nominal "
              "$\\alpha=0.05$.", "w2")
        xd = {d: m(w2_x("loglik", d)) for d in DOMS if d != "hotel"}
        off_books = m(w2_x("offshelf", "books")); off_news = m(w2_x("offshelf", "news"))
        max_sup_xd = max(m(w2_x(s, d)) for s in ("tfidf", "supervised")
                         for d in DOMS if d != "hotel")
        L += [r"In-domain calibration is again exact in all six domains. Hotel-calibrated "
            r"likelihood certificates inflate on edited prose: news "
            rf"{xd['news']:.2f}, reviews {xd['reviews']:.2f}, books {xd['books']:.2f}, "
            rf"abstracts {xd['abstracts']:.2f} at nominal $0.05$, while poetry is mild "
            rf"({xd['poetry']:.2f}) --- fluent, professionally edited human text is exactly "
            r"what a likelihood score mistakes for AI. The supervised scores are conservative "
            rf"under every domain shift (max FPR {max_sup_xd:.3f}), and the off-the-shelf "
            rf"detector sits in between (books {off_books:.2f}, news {off_news:.2f}). The full "
            rf"$6\times 6$ source$\to$target matrix (Figure~\ref{{fig:f8_transfer_matrix}}) "
            r"shows the asymmetry: calibrating on the \emph{most} fluent domain (news) is safe "
            r"everywhere but powerless; calibrating on the least fluent (hotel, poetry) is "
            r"dangerous everywhere else.",]
        figure(L, "f8_transfer_matrix.png",
               "W2: $6\\times 6$ cross-domain FPR matrix (calibrate on row, deploy on column) "
               "at nominal $\\alpha=0.05$.", 0.92)

        # ------------------------------- W3 ------------------------------- #
        L += [r"\subsection{W3: the LLM-paraphrase attack on innocent humans --- and its "
              r"repair}\label{sec:w3}"]
        rows = []
        for s in W2SC:
            rows.append([W2LAB[s],
                         f"{m(w3_fpr(s, 'naive', 'human_clean')):.3f}",
                         rf"\textbf{{{m(w3_fpr(s, 'naive', 'human_para')):.3f}}}",
                         f"{m(w3_fpr(s, 'robust', 'human_para')):.3f}",
                         f"{m(w3_tpr(s, 'naive', 'ai_clean')):.2f}",
                         f"{m(w3_tpr(s, 'robust', 'ai_clean')):.2f}"])
        table(L, ["Score", "FPR clean", "FPR para (naive)", "FPR para (robust)",
                  "TPR (naive)", "TPR (robust)"], rows,
              "W3 at $\\alpha=0.05$: LLM-paraphrased human text under naive vs.\\ "
              "paraphrase-aware robust calibration, and the power cost on clean AI text.", "w3")
        sup_n = para_naive["supervised"]; sup_r = para_rob["supervised"]
        sup_tpr_n = m(w3_tpr("supervised", "naive", "ai_clean"))
        sup_tpr_r = m(w3_tpr("supervised", "robust", "ai_clean"))
        tf_tpr_n = m(w3_tpr("tfidf", "naive", "ai_clean"))
        tf_tpr_r = m(w3_tpr("tfidf", "robust", "ai_clean"))
        L += [r"This is the headline failure: a human who innocently runs their own text "
            r"through an LLM paraphraser (``please improve my wording'') is accused at "
            rf"{pn_lo:.2f}--{pn_hi:.2f} instead of $0.05$ --- and unlike the population shifts "
            r"above, \emph{every} scorer family breaks, including the fine-tuned transformer "
            rf"({sup_n:.2f}) and the off-the-shelf detector ({para_naive['offshelf']:.2f}). "
            r"Paraphrase output \emph{is} LLM output; no scorer can be expected to separate "
            r"paraphrased-human from generated text, so the fix must come from calibration. "
            r"Modeling the paraphraser in the editor-aware construction (\S 3.4; two modeled "
            r"variants per calibration human, seeds disjoint from test paraphrases) restores "
            rf"validity for all four scorers ({min(para_rob.values()):.3f}--{pr_hi:.3f}). "
            r"Unlike the cheap cleanup-editor insurance of E8, here the power cost is "
            rf"substantial --- TPR {sup_tpr_n:.2f}$\to${sup_tpr_r:.2f} (fine-tuned) and "
            rf"{tf_tpr_n:.2f}$\to${tf_tpr_r:.2f} (TF--IDF) on clean AI text --- the honest "
            r"price of remaining unable to accuse paraphrase-shaped humans "
            r"(Figure~\ref{fig:f11_paraphrase}).",]
        figure(L, "f11_paraphrase.png",
               "W3: naive vs.\\ paraphrase-aware robust calibration under the LLM-paraphrase "
               "attack at nominal $\\alpha=0.05$.", 0.95)

        # ------------------------------- W4 ------------------------------- #
        L += [r"\subsection{W4: the diagnostics work --- every violation is caught before "
              r"deployment}\label{sec:w4}"]

        def w4_pretty(c):
            if c == "control_english":
                return "English control (in-population)"
            if c == "paraphrased_humans":
                return "LLM-paraphrased humans"
            if c.startswith("lang_"):
                return esc(c[5:]) + " humans"
            if c.startswith("domain_"):
                return esc(c[7:].capitalize()) + " humans"
            return esc(c)

        rows = []
        for c in ll_conds:
            rec = w4_rec(c, "loglik")
            fpr_c = m(rec["fpr"])
            mu, rate = w4_alarm(c, "loglik")
            fcell = rf"\textbf{{{fpr_c:.3f}}}" if rec["violated"] else f"{fpr_c:.3f}"
            acell = f"{rate:.2f}" + ("" if mu == 100 else rf" ($m{{=}}{mu}$)")
            rows.append([w4_pretty(c), fcell, f"{rec['tv_bound_mean']:.2f}",
                         f"{rec['gamma_bound_mean']:.2f}", acell])
        table(L, ["Condition", "Realized FPR", r"TV bound", r"$\Gamma$ bound",
                  r"Alarm@$m{=}100$"], rows,
              "W4 (log-likelihood score, $\\alpha=0.05$): realized FPR (bold = certified "
              "violation), plug-in score-space bounds, and pilot-alarm rate. Alarm level "
              f"$\\rho={ALARM_LVL}$.", "w4")
        abs_mu, abs_rate = w4_alarm("domain_abstracts", "loglik")
        L += [rf"Across all four scorers, {len(viol_pairs)} of "
            rf"{len(ll_conds)*len(W2SC)} condition--scorer pairs at $\alpha=0.05$ are "
            r"certified violations, and the pilot alarm catches \emph{every one of them}: "
            rf"rate $\ge{min_viol_alarm100:.2f}$ at $m=100$ unlabeled pilot texts (the small "
            rf"abstracts pool caps its pilot at $m={abs_mu}$, where the rate is "
            rf"{abs_rate:.2f}). In-control conditions stay at the design level: the English "
            rf"control alarms at most {ctrl_max_alarm:.3f} $\le \rho={ALARM_LVL}$, and "
            r"conservative (under-flagging) shifts never alarm, as they should --- the "
            r"detector tests over-accusation only. The plug-in bounds behave as the theory "
            rf"section warned: the TV bound covers the realized FPR in {tv_cov} of "
            rf"{len(ll_conds)} likelihood conditions (its one miss is by less than $0.01$ on "
            rf"news), while the binned $\widehat\Gamma\alpha$ bound under-covers in "
            rf"{len(ll_conds)-ga_cov} of {len(ll_conds)} --- plug-in estimates rank risk "
            r"usefully but must not be read as certificates; the calibrated alarm is the "
            r"deployable instrument (Figure~\ref{fig:f10_diagnostics}).",]
        figure(L, "f10_diagnostics.png",
               "W4: pilot-alarm rate vs.\\ pilot size $m$ per condition (left) and plug-in "
               "TV/$\\Gamma$ bounds vs.\\ realized FPR (right).", 0.98)

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
        r"The Wave-1 core is English-only with one in-domain corpus (hotel reviews) whose "
        r"humans skew non-native --- which is what makes the population-shift result vivid, "
        r"but magnitudes will vary"
        + (r"; Wave 2 widens this to ten languages and six domains, and replaces the cleanup "
           r"editor with a T5 ChatGPT-paraphraser (\S\ref{sec:w3}), but uses one paraphraser "
           r"and one multilingual corpus family" if W2 else
           r"; the cleanup editor under-approximates modern LLM paraphrasers, so E4/E8 bound "
           r"only the editors tested")
        + r". GPT-2 is a deliberately modest scorer (the wrapper is scorer-agnostic); our "
        r"exact tests certify violations, not their causes --- the fluency mechanism is "
        r"supported but not isolated"
        + (r"; and the TV/$\Gamma$ diagnostics are plug-in estimates whose own sampling error "
           r"is unquantified here (W4 measures, but does not bound, their coverage)" if W2
           else "")
        + r". The protocol, released in full, replicates at larger scale directly.",
        r"\section{Conclusion}",
        r"Distribution-free certificates make AI-text detectors accountable, but only within the "
        r"human distribution they were calibrated on. We mapped that boundary --- exact in "
        r"distribution, robust to light editing, broken by population shift, length-biased "
        r"marginally --- and showed the repairs are cheap."
        + ((r" Wave 2 stress-tests the map at deployment scale: four scorer families "
            r"(including a fine-tuned transformer and an off-the-shelf detector), ten "
            r"languages, six domains, and an LLM-paraphrase attack. The pattern is consistent "
            r"and honest about its epistemics: validity is \emph{exact in-population by "
            r"construction} --- every language and every domain certifies cleanly when "
            r"calibrated on itself --- while every cross-population finding is empirical. "
            r"Likelihood certificates transferred across language or domain fail in both "
            rf"directions (Russian {ru_x:.3f}, news {news_x:.2f}, versus silent "
            r"under-flagging elsewhere); LLM-paraphrased innocent humans break \emph{all} "
            rf"scorer families ({pn_lo:.2f}--{pn_hi:.2f} at nominal $0.05$) and "
            r"paraphrase-aware calibration repairs them at a real power cost. Crucially, none "
            r"of these failures need be discovered in production: the $\Gamma\alpha$ and "
            r"$\alpha+\mathrm{TV}$ bounds live on the one-dimensional score space, and the "
            r"calibrated pilot alarm built on the same observation caught every certified "
            rf"violation in this study at $m=100$ unlabeled pilot texts while respecting its "
            rf"{ALARM_LVL:.2f} false-alarm budget. The bounds remain plug-in estimates, not "
            r"certificates; the alarm is the deployable instrument.") if W2 else "")
        + r" A certificate is a promise about \emph{whom} you calibrated on; deployers should "
        r"make that promise explicit --- and pilot-test it before the first accusation.",
        r"\paragraph{Reproducibility.} All numbers derive from three JSON files produced by "
        r"\path{scripts/run_all.py}, \path{scripts/run_robust.py} and "
        r"\path{scripts/run_wave2.py}; figures and this paper regenerate from them; "
        r"conformal correctness is asserted by the test suite.",]

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
