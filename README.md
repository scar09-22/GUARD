# GUARD — GUaranteed Abstention for Reliable Detection of machine-generated text

*When can you trust the certificate? Stress-testing distribution-free false-accusation
guarantees for AI-text detection.*

## What GUARD is

AI-text detectors are deployed to **accuse**, and the accused bears the cost of a false
positive. GUARD therefore replaces "detector accuracy" with a deployment contract: wrap any
base detector score in a split-conformal certificate — flag a text as AI only when its
conformal p-value against `n` human calibration texts is at most `alpha`. If the test human
text is exchangeable with the calibration humans, the false-accusation rate is provably at
most `alpha`: finite-sample, distribution-free, for **any** score function. Everything else
(the AI distribution, the scorer's quality) affects only detection power, never validity.

That asymmetry yields sharp predictions about which deployment conditions break the
certificate, and GUARD maps them empirically (experiments E1–E7, see `METHODOLOGY.md`):

| Deployment condition | Certificate (FPR ≤ α) | Power (TPR) |
|---|---|---|
| In-domain, exchangeable humans | holds (provable) | scorer-dependent |
| Generator shift (new LLM, same domain) | holds — humans unchanged | degrades (measured) |
| Human domain shift | at risk (measured) | — |
| Human edit attack (innocent editing tools) | **at risk — the critical case** | unchanged |
| AI evasion attack (paraphrased AI text) | holds — humans unchanged | degrades (measured) |
| Entity leakage in calibration | optimistic if leaky (measured) | — |
| Length heterogeneity | marginal cert can hide per-length gaps; Mondrian fixes | — |

The contribution is this validity map plus the constructive protocols (entity-disjoint
grouped calibration, Mondrian-by-length calibration) — not a new conformal method.

## Install

Python 3.11+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Data expected in place:

- `data/all_data.csv` — MAiDE-up hotel reviews (the English slice is used: 1000 human /
  1000 GPT-4 reviews over 100 hotels).
- `results/cache/raid_abstracts.parquet` — RAID abstracts pool (human-domain-shift probe).
- `results/cache/raid_reviews_pool.parquet` — RAID reviews-domain pool (optional; all RAID
  arms are skipped gracefully if this cache is absent).

NLTK's WordNet corpus is downloaded automatically on first use of the synonym attack.

## Quickstart

```bash
# 1. Self-checks: conformal validity, attack determinism, data integrity, scorer sanity.
python tests/test_guard.py

# 2. Quick end-to-end pass with reduced repeats (sanity-check the full pipeline).
python scripts/run_all.py --smoke

# 3. Full study: E1–E7, >= 200 seeded repeats per cell, results written as JSON,
#    every figure regenerated from that JSON.
python scripts/run_all.py
```

The GPT-2 scorer test is heavyweight and skipped by default; enable it with
`GUARD_TEST_GPT2=1 python tests/test_guard.py`.

## Honest claims

GUARD does **not** introduce a new conformal method — the certificate is the standard split
conformal construction. It does **not** claim state-of-the-art detection power — the scorers
are deliberately modest because the wrapper is scorer-agnostic by design. Scope is English
reviews and abstracts, with a single lexical proxy for human editing; paraphrase-model edits
are left to the released protocol at full scale. Every reported number is a mean over >= 200
seeded repeats with 95% CIs, violations are tested against the exact Beta/Binomial null, and
every figure regenerates from one results JSON — no number is hand-entered.

## Repository layout

```
GUARD/
├── METHODOLOGY.md          # the study design: claims, E1–E7, statistics
├── README.md
├── requirements.txt
├── data/
│   └── all_data.csv        # MAiDE-up reviews (multilingual; English slice used)
├── results/
│   └── cache/              # RAID pools (parquet) + cached scores / results JSON
├── src/guard/
│   ├── conformal.py        # certificate: p-values, flags, Beta envelope, Mondrian
│   ├── data.py             # MAiDE-up / RAID loaders, grouped vs random splits
│   ├── attacks.py          # function-word and WordNet-synonym perturbations
│   └── scores.py           # GPT-2 zero-shot statistics + TF-IDF supervised scorer
├── scripts/
│   └── run_all.py          # E1–E7 driver (--smoke for a quick pass)
├── tests/
│   └── test_guard.py       # runnable as plain python or via pytest
├── figures/                # regenerated from results JSON
└── paper/
```

## Results at a glance (alpha = 0.05, R = 200 resamples)

| Condition | loglik | entropy | tfidf | Certificate |
|---|---|---|---|---|
| In-domain humans | 0.047 | 0.051 | 0.050 | **exact** |
| Reviews-domain humans (population shift) | 0.45 | 0.54 | 0.00 | **broken** (likelihood) / safe (supervised) |
| Abstracts humans (domain shift) | 0.12 | 0.27 | 0.01 | broken / safe |
| Cleanup-edited humans (grammar-tool proxy) | 0.048 | 0.045 | 0.049 | holds |
| Synonym/FW-edited humans (lexical noise) | ≤0.005 | ≤0.01 | ≤0.05 | holds (conservative) |
| Long humans, marginal calibration | 0.18 | — | — | length-biased |
| Long humans, Mondrian calibration | 0.030 | — | — | repaired |

Power at certified 5% FPR: TF-IDF 0.88 TPR in-domain; robust (editor-aware) calibration costs
< 0.01 TPR. Full numbers: `results/guard_results.json`, `results/guard_robust.json`; paper:
`paper/GUARD.tex` (compiles to 10 pp).
