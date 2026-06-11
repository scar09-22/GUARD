# GUARD — GUaranteed Abstention for Reliable Detection of machine-generated text

**Working title:** *When Can You Trust the Certificate? Stress-Testing Distribution-Free
False-Accusation Guarantees for AI-Text Detection.*

## 1. Problem

AI-text detectors are deployed to *accuse*: a flagged student, reviewer, or author bears the
cost of a false positive. Detector accuracy is the wrong contract for that setting — what a
deployer needs is a **bound on the false-accusation rate** that holds for *their* data without
distributional assumptions. Split conformal prediction supplies exactly this: calibrate a
score threshold on n held-out human texts and the probability of flagging a fresh human text
is at most α, **finite-sample, distribution-free**, for any base detector score.

Prior work (Multiscaled Conformal Prediction, ACL 2025) established CP as a way to bound
detector FPR. GUARD asks the *next* question, the one a deployer actually faces:

> **Under which realistic deployment conditions does the certificate remain valid, and what
> does it cost in detection power?**

## 2. The certificate asymmetry (conceptual spine)

The split-conformal false-accusation bound depends **only on exchangeability of human
calibration and human test text**. The AI distribution is irrelevant to validity (it only
determines power). This yields sharp, testable predictions:

| Deployment condition | Certificate (FPR ≤ α) | Power (TPR) |
|---|---|---|
| In-domain, exchangeable humans | **holds (provable)** | scorer-dependent |
| **Generator shift** (new LLM, same domain) | **holds (provable)** — humans unchanged | degrades unknown amount (measure) |
| **Human domain shift** (reviews → other domain) | at risk (measure violation) | — |
| **Human edit attack** (innocent paraphrase/grammar tools on human text) | **at risk — the deployment-critical case** | unchanged |
| AI evasion attack (paraphrased AI text) | holds — humans unchanged | degrades (measure) |
| **Entity leakage in calibration** (random vs grouped-by-hotel splits) | optimistic if leaky (measure) | — |
| **Length heterogeneity** | marginal cert can hide per-length miscoverage (measure; fix with Mondrian) | — |

No existing work maps this table empirically. That map — plus the constructive fixes
(grouped calibration, Mondrian-by-length calibration) — is the contribution.

## 3. Method

**Scores.** Scorer-agnostic. Base scores s(x), oriented higher = more AI-like:
four training-free statistics from ONE GPT-2 forward pass per text (mean log-likelihood,
mean log-rank, predictive entropy, Fast-DetectGPT analytic discrepancy) plus one supervised
score (TF-IDF + logistic regression trained on a grouped split). GUARD wraps any of them.

**Certificate.** Split conformal p-value against the human calibration class:
p(x) = (1 + #{i : s(x_i) ≥ s(x)}) / (n + 1), flag as AI iff p(x) ≤ α.
Theorem (standard): if test human text is exchangeable with the n calibration human texts,
P(flag | human) ≤ α. Ties handled conservatively. Conditional on a calibration set, the
realised FPR follows a known Beta law — we verify the entire distribution, not just the mean.

**Variants.**
- *Marginal* (the standard certificate).
- *Mondrian by length*: per-length-bin calibration → per-bin validity.
- *Grouped calibration*: calibration humans drawn entity-disjoint (by hotel) from test
  humans, mirroring deployment on review platforms.

**Decision semantics.** GUARD is an accuser with a certificate: flag (certified at α) or
do-not-accuse. Power = P(flag | AI) at the certified operating point; we report
power–certificate frontiers across α ∈ {0.5%, 1%, 2%, 5%, 10%} per scorer.

## 4. Experiments (all leakage-aware, all multi-repeat)

Data: MAiDE-up English (human vs GPT-4 hotel reviews; grouped by hotel), RAID reviews-domain
pool (8 generators incl. GPT-3.5/4, Cohere, Llama; human + AI), RAID abstracts (human domain
shift). Attacks: synonym-substitution and function-word perturbation applied to *human* test
text (certificate attack — proxy for innocent tool-assisted editing) and to *AI* text
(evasion attack).

E1 Validity in-distribution: R ≥ 200 random calibration/test partitions → empirical FPR
   distribution vs the theoretical Beta envelope, per scorer, per α. (Theory-match figure.)
E2 Generator shift: calibrate on MAiDE-up humans; test power on each RAID generator
   separately; verify FPR on RAID-reviews humans (mild human shift: hotel vs movie reviewers
   — quantifies how far "same broad domain" stretches the certificate).
E3 Human domain shift: calibrate hotel humans → test abstracts humans. Measure violation.
E4 Human edit attack: calibrate clean humans → test edited humans (synonym / FW perturbation
   at increasing edit rates). Violation-vs-edit-rate curves. **Headline experiment.**
E5 Leakage: random vs grouped (entity-disjoint) calibration; miscoverage gap.
E6 Mondrian: per-length-bin FPR under marginal vs Mondrian calibration.
E7 Power frontier: TPR at each certified α per scorer; per-generator breakdown.

Statistics: every number = mean over ≥200 repeats with 95% CIs; violations tested against
the exact Beta/binomial null; all splits seeded and logged.

## 5. Honest-claims contract

- We do NOT claim a new conformal method; the certificate is standard. The contribution is
  the validity map + protocols (grouped, Mondrian) + the certificate-asymmetry framing.
- We do NOT claim SOTA detection power; scorers are deliberately modest and the wrapper is
  scorer-agnostic by design.
- Scope: English reviews + abstracts; single human-edit proxy (lexical); paraphrase-model
  edits left to the released protocol at full scale.
- Every figure regenerates from one results JSON; no number is hand-entered.
