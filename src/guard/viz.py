"""Publication figures for the GUARD validity map.

Every figure regenerates from the single results dictionary produced by
``guard.experiments.run_all`` (or its JSON round-trip) -- no number is ever
hand-entered.  Figures correspond to the experiments in METHODOLOGY.md:

  f1_validity_envelope  E1  per-repeat empirical FPR histograms vs the exact
                            Beta / Beta-Binomial law implied by split-conformal
                            theory (the theory-match figure)
  f2_certificate_map    E1-E4  headline "when does the certificate hold" map:
                            empirical FPR across deployment conditions, with
                            violated bars marked
  f3_violation_vs_rate  E4  FPR vs human-edit rate per attack kind
  f4_leakage            E5  random vs entity-grouped calibration FPR
                            distributions
  f5_mondrian           E6  per-length-bin FPR, marginal vs Mondrian
  f6_power_frontier     E7  TPR vs certified alpha, clean vs evasion
  f7_per_generator      E2  power per RAID generator per scorer

Extraction is deliberately defensive: experiment blocks may be nested dicts
keyed by scorer / condition / rate, flat lists of per-repeat records (dicts or
``CertifiedResult`` dataclasses), or aggregate dicts holding parallel arrays
("fprs", "tprs", ...).  A missing or malformed block makes the corresponding
figure skip cleanly (return ``None``) rather than crash the run.
"""
from __future__ import annotations

import math
import re
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import matplotlib

matplotlib.use("Agg")  # headless rendering; must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats as scipy_stats

__all__ = [
    "iter_records",
    "records_at",
    "scorer_of",
    "summary_rows",
    "f1_validity_envelope",
    "f2_certificate_map",
    "f3_violation_vs_rate",
    "f4_leakage",
    "f5_mondrian",
    "f6_power_frontier",
    "f7_per_generator",
    "make_all_figures",
]

# --------------------------------- style --------------------------------- #

_STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
}

# --------------------------- record extraction ---------------------------- #

# Aggregate blocks may store parallel arrays under plural keys.
_PLURAL = {"fprs": "fpr", "tprs": "tpr", "violation_ps": "violation_p", "alphas": "alpha"}
# Keys that may legitimately hold per-repeat sequences inside one record dict.
_SEQ_KEYS = ("alpha", "fpr", "tpr", "violation_p", "n_cal", "n_test_human", "n_test_ai")
# Path elements that are container names, never scorer names.
_NON_SCORER = {
    "records", "repeats", "results", "data", "per_scorer", "scorers",
    "by_scorer", "per-scorer", "conditions", "meta",
}
_NON_GENERATOR = {
    "records", "repeats", "results", "data", "power", "fpr", "tpr",
    "per_generator", "generators", "by_generator", "human", "humans",
}


def _finite(x: Any) -> bool:
    """True iff x converts to a finite float."""
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _explode(rec: dict) -> Iterator[dict]:
    """Expand a record whose value fields are parallel sequences into rows."""
    seqs: dict[str, list] = {}
    out = dict(rec)
    for k in _SEQ_KEYS:
        v = out.get(k)
        if isinstance(v, np.ndarray):
            v = v.tolist()
            out[k] = v
        if isinstance(v, (list, tuple)):
            seqs[k] = list(v)
    if not seqs:
        yield out
        return
    n = max(len(v) for v in seqs.values())
    for i in range(n):
        row = dict(out)
        for k, v in seqs.items():
            row[k] = v[i] if i < len(v) else None
        yield row


def iter_records(block: Any, _path: tuple[str, ...] = ()) -> Iterator[dict]:
    """Yield flat per-repeat record dicts from an arbitrarily nested block.

    A record is any dict (or dataclass, e.g. ``CertifiedResult``) carrying an
    ``alpha`` together with ``fpr`` and/or ``tpr`` (scalar or sequence).  Each
    yielded dict gains a ``_path`` tuple of the dict keys traversed to reach
    it, used as a fallback for scorer / condition labels.
    """
    if block is None:
        return
    if is_dataclass(block) and not isinstance(block, type):
        yield from _explode({**asdict(block), "_path": _path})
        return
    if isinstance(block, dict):
        rec = {_PLURAL.get(k, k): v for k, v in block.items()}
        if "alpha" in rec and any(k in rec for k in ("fpr", "tpr")):
            rec["_path"] = _path
            yield from _explode(rec)
            return
        for key, val in block.items():
            yield from iter_records(val, _path + (str(key),))
        return
    if isinstance(block, (list, tuple)):
        for item in block:
            yield from iter_records(item, _path)


def _block(results: Any, key: str) -> Any:
    """Fetch an experiment block tolerating key variants (E1, e1, E1_validity)."""
    if not isinstance(results, dict):
        return None
    if key in results:
        return results[key]
    kl = key.lower()
    for k, v in results.items():
        if str(k).lower().startswith(kl):
            return v
    return None


def _alpha_match(rec: dict, alpha: float) -> bool:
    a = rec.get("alpha")
    return _finite(a) and math.isclose(float(a), float(alpha), rel_tol=1e-6, abs_tol=1e-9)


def records_at(block: Any, alpha: float | None, value: str = "fpr") -> list[dict]:
    """All records in block at the given alpha with a finite `value` field."""
    return [
        r for r in iter_records(block)
        if (alpha is None or _alpha_match(r, alpha)) and _finite(r.get(value))
    ]


def scorer_of(rec: dict) -> str:
    """Scorer label: explicit field first, then first plausible path element."""
    for k in ("scorer", "score_name", "detector"):
        v = rec.get(k)
        if isinstance(v, str) and v:
            return v
    for p in rec.get("_path", ()):
        if str(p).lower() not in _NON_SCORER:
            return str(p)
    return "scorer"


def _match_label(rec: dict, keys: tuple[str, ...], tags: dict[str, tuple[str, ...]]) -> str | None:
    """First tag whose substrings match an explicit field (preferred) or a path element."""
    cands: list[str] = []
    for k in keys:
        v = rec.get(k)
        if v is not None and not isinstance(v, (dict, list, tuple)):
            cands.append(str(v))
    cands.extend(str(p) for p in rec.get("_path", ()))
    for s in cands:
        low = s.lower()
        for tag, subs in tags.items():
            if any(sub in low for sub in subs):
                return tag
    return None


def _attack_of(rec: dict) -> str | None:
    return _match_label(
        rec,
        ("attack", "kind", "perturbation", "edit", "attack_kind"),
        {"function-word": ("fw", "func"), "synonym": ("syn",)},
    )


def _cond_of(rec: dict) -> str:
    """Clean vs evasion condition (E7); defaults to clean when unlabelled."""
    tag = _match_label(
        rec,
        ("condition", "setting", "variant", "attack", "mode"),
        {
            "clean": ("clean", "none", "base", "unattacked"),
            "evasion": ("evas", "paraphr", "syn", "fw", "attack", "edit"),
        },
    )
    return tag or "clean"


def _rate_of(rec: dict) -> float | None:
    """Edit rate: explicit numeric field, else a numeric / 'rate=' path element."""
    for k in ("edit_rate", "rate", "epsilon", "frac"):
        v = rec.get(k)
        if _finite(v):
            return float(v)
    for p in rec.get("_path", ()):
        s = str(p).strip().lower()
        if "rate" in s and "=" in s:
            try:
                return float(s.split("=")[-1])
            except ValueError:
                pass
        try:
            return float(s)
        except ValueError:
            continue
    return None


def _generator_of(rec: dict) -> str | None:
    for k in ("generator", "model", "llm", "gen"):
        v = rec.get(k)
        if isinstance(v, str) and v:
            return v
    sc = scorer_of(rec)
    for p in rec.get("_path", ()):
        s = str(p)
        if s == sc or s.lower() in _NON_GENERATOR:
            continue
        return s
    return None


def _nearest_rate(recs: list[dict], target: float) -> tuple[list[dict], float | None]:
    """Subset of records at the available edit rate closest to `target`."""
    rates = sorted({r2 for r2 in (_rate_of(r) for r in recs) if r2 is not None})
    if not rates:
        return recs, None
    best = min(rates, key=lambda r: abs(r - target))
    sub = [
        r for r in recs
        if _rate_of(r) is not None
        and math.isclose(_rate_of(r), best, rel_tol=1e-9, abs_tol=1e-12)
    ]
    return sub, best


def _group(recs: list[dict], keyfn: Callable[[dict], Any]) -> dict[Any, list[dict]]:
    out: dict[Any, list[dict]] = {}
    for r in recs:
        out.setdefault(keyfn(r), []).append(r)
    return out


def _values(recs: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in recs if _finite(r.get(key))], dtype=float)


def _mean_ci(values: Any) -> tuple[float, float]:
    """Mean and half-width of a normal-approximation 95% CI over repeats."""
    arr = np.asarray([v for v in np.atleast_1d(values) if _finite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), 0.0
    m = float(arr.mean())
    if arr.size < 2:
        return m, 0.0
    return m, 1.96 * float(arr.std(ddof=1)) / math.sqrt(arr.size)


def _typical_int(recs: list[dict], key: str) -> int:
    vals = [float(r[key]) for r in recs if _finite(r.get(key))]
    return int(np.median(vals)) if vals else 0


# --------------------------- plotting utilities --------------------------- #

def _scorer_colors(scorers: list[str]) -> dict[str, Any]:
    cmap = plt.get_cmap("tab10")
    return {sc: cmap(i % 10) for i, sc in enumerate(sorted(scorers))}


def _subplot_grid(n: int, width: float = 3.4, height: float = 2.7):
    ncols = min(3, max(1, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False)
    flat = axes.ravel().tolist()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def _save(fig, outdir: Path | str, name: str) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _bin_sort_key(b: str):
    try:
        return (0, float(b), "")
    except ValueError:
        m = re.search(r"-?\d+\.?\d*", str(b))
        return (1, float(m.group()) if m else math.inf, str(b))


# --------------------------------- figures -------------------------------- #

def f1_validity_envelope(results: dict, outdir: Path | str, alpha: float = 0.05) -> Path | None:
    """E1: histogram of per-repeat FPRs vs the exact conformal Beta law.

    Per scorer: empirical density of the realised FPR across calibration/test
    repartitions, the Beta(k, n+1-k) density of the conditional FPR, the
    Beta-Binomial 5-95% envelope (which adds finite-test-set binomial noise),
    and the nominal alpha line.  Theory and histogram must agree when human
    exchangeability holds.
    """
    recs = records_at(_block(results, "E1"), alpha, "fpr")
    if not recs:
        return None
    by_scorer = _group(recs, scorer_of)
    with plt.rc_context(_STYLE):
        fig, axes = _subplot_grid(len(by_scorer))
        for ax, (sc, rs) in zip(axes, sorted(by_scorer.items())):
            fprs = _values(rs, "fpr")
            n_cal = _typical_int(rs, "n_cal")
            m = _typical_int(rs, "n_test_human")
            hi_env = alpha
            beta_dist = None
            if n_cal > 0 and m > 0:
                k = int(math.floor(alpha * (n_cal + 1)))
                if k >= 1:
                    bb = scipy_stats.betabinom(m, k, n_cal + 1 - k)
                    lo, hi_env = float(bb.ppf(0.05)) / m, float(bb.ppf(0.95)) / m
                    ax.axvspan(lo, hi_env, color="tab:green", alpha=0.15, zorder=0,
                               label="Beta-Binomial 5-95% envelope")
                    beta_dist = scipy_stats.beta(k, n_cal + 1 - k)
            span = float(max(fprs.max() if fprs.size else 0.0, hi_env, alpha)) * 1.25
            span += (1.0 / m) if m > 0 else 0.01
            if m > 0:
                step = max(1, math.ceil(span * m / 30)) / m
                edges = np.arange(0.0, span + step, step) - step / 2.0
            else:
                edges = 20
            ax.hist(fprs, bins=edges, density=True, color="tab:blue", alpha=0.65,
                    edgecolor="white", linewidth=0.4, label="empirical per-repeat FPR")
            if beta_dist is not None:
                xs = np.linspace(0.0, span, 256)
                ax.plot(xs, beta_dist.pdf(xs), color="tab:red", lw=1.2,
                        label="theoretical Beta law")
            ax.axvline(alpha, color="black", ls="--", lw=1.0,
                       label=f"nominal alpha = {alpha:g}")
            ax.set_title(sc)
            ax.set_xlabel("Per-repeat empirical FPR")
            ax.set_ylabel("Density")
        axes[0].legend(loc="upper right")
        fig.suptitle(f"E1 validity in-distribution: realised FPR vs conformal Beta law (alpha = {alpha:g})")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return _save(fig, outdir, "f1_validity_envelope")


def _f2_condition_records(results: dict, alpha: float, edit_rate: float) -> dict[str, list[dict]]:
    conds: dict[str, list[dict]] = {}
    conds["in-domain"] = records_at(_block(results, "E1"), alpha, "fpr")
    e2 = records_at(_block(results, "E2"), alpha, "fpr")
    human_tagged = [
        r for r in e2
        if _match_label(r, ("condition", "split", "population", "test_set", "generator", "model"),
                        {"human": ("human",)}) == "human"
    ]
    conds["reviews-humans"] = human_tagged or e2
    conds["abstracts-humans"] = records_at(_block(results, "E3"), alpha, "fpr")
    e4 = records_at(_block(results, "E4"), alpha, "fpr")
    for tag, name in (("function-word", "fw-edited"), ("synonym", "synonym-edited")):
        sub = [r for r in e4 if _attack_of(r) == tag]
        sub, _ = _nearest_rate(sub, edit_rate)
        conds[name] = sub
    return {k: v for k, v in conds.items() if v}


def f2_certificate_map(
    results: dict,
    outdir: Path | str,
    alpha: float = 0.05,
    edit_rate: float = 0.3,
    violation_level: float = 0.05,
) -> Path | None:
    """E1-E4 headline map: empirical FPR across deployment conditions.

    Grouped bars (mean over repeats, 95% CI) per scorer for: in-domain,
    RAID-reviews humans, abstracts humans, function-word-edited humans,
    synonym-edited humans (at the available edit rate closest to `edit_rate`).
    Bars whose mean exact Beta-Binomial violation p-value falls below
    `violation_level` are hatched and starred: the certificate broke there.
    """
    conds = _f2_condition_records(results, alpha, edit_rate)
    if not conds:
        return None
    cond_names = list(conds)
    scorers = sorted({scorer_of(r) for recs in conds.values() for r in recs})
    if not scorers:
        return None
    colors = _scorer_colors(scorers)
    width = 0.8 / len(scorers)
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(5.6, 1.5 * len(cond_names)), 3.4))
        any_violated = False
        for j, sc in enumerate(scorers):
            for i, cond in enumerate(cond_names):
                rs = [r for r in conds[cond] if scorer_of(r) == sc]
                if not rs:
                    continue
                mean, half = _mean_ci(_values(rs, "fpr"))
                if not _finite(mean):
                    continue
                vio = _values(rs, "violation_p")
                violated = vio.size > 0 and float(vio.mean()) < violation_level
                x = i - 0.4 + (j + 0.5) * width
                bars = ax.bar(x, mean, width=width * 0.92, yerr=half, capsize=2,
                              color=colors[sc], error_kw={"lw": 0.8},
                              label=sc if i == 0 else None)
                if violated:
                    any_violated = True
                    bars[0].set_hatch("///")
                    bars[0].set_edgecolor("black")
                    ax.annotate("*", (x, mean + half), ha="center", va="bottom",
                                fontsize=11, color="black")
        ax.axhline(alpha, color="black", ls="--", lw=1.0)
        ax.set_xticks(range(len(cond_names)))
        ax.set_xticklabels(cond_names, rotation=12)
        ax.set_ylabel("Empirical FPR (mean, 95% CI)")
        ax.set_title(f"When does the false-accusation certificate hold?  (alpha = {alpha:g})")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color="black", ls="--", lw=1.0))
        labels.append(f"nominal alpha = {alpha:g}")
        if any_violated:
            handles.append(Patch(facecolor="white", edgecolor="black", hatch="///"))
            labels.append(f"* violated (mean violation p < {violation_level:g})")
        ax.legend(handles, labels, ncols=2)
        fig.tight_layout()
        return _save(fig, outdir, "f2_certificate_map")


def f3_violation_vs_rate(results: dict, outdir: Path | str, alpha: float = 0.05) -> Path | None:
    """E4: empirical FPR vs human-edit rate, one curve per attack kind, per scorer."""
    recs = records_at(_block(results, "E4"), alpha, "fpr")
    recs = [r for r in recs if _rate_of(r) is not None]
    if not recs:
        return None
    by_scorer = _group(recs, scorer_of)
    attacks = sorted({(_attack_of(r) or str(r.get("attack", "edit"))) for r in recs})
    cmap = plt.get_cmap("tab10")
    attack_colors = {a: cmap(i % 10) for i, a in enumerate(attacks)}
    with plt.rc_context(_STYLE):
        fig, axes = _subplot_grid(len(by_scorer))
        for ax, (sc, rs) in zip(axes, sorted(by_scorer.items())):
            for atk in attacks:
                ars = [r for r in rs if (_attack_of(r) or str(r.get("attack", "edit"))) == atk]
                if not ars:
                    continue
                by_rate = _group(ars, _rate_of)
                rates = sorted(by_rate)
                stats = [_mean_ci(_values(by_rate[rt], "fpr")) for rt in rates]
                means = np.array([s[0] for s in stats])
                halves = np.array([s[1] for s in stats])
                ax.plot(rates, means, marker="o", ms=3, lw=1.3,
                        color=attack_colors[atk], label=atk)
                ax.fill_between(rates, means - halves, means + halves,
                                color=attack_colors[atk], alpha=0.18, lw=0)
            ax.axhline(alpha, color="black", ls="--", lw=1.0,
                       label=f"nominal alpha = {alpha:g}")
            ax.set_title(sc)
            ax.set_xlabel("Human edit rate")
            ax.set_ylabel("Empirical FPR")
        axes[0].legend()
        fig.suptitle(f"E4 human-edit attack: certificate violation vs edit rate (alpha = {alpha:g})")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return _save(fig, outdir, "f3_violation_vs_rate")


def f4_leakage(results: dict, outdir: Path | str, alpha: float = 0.05) -> Path | None:
    """E5: per-repeat FPR distributions, random vs entity-grouped calibration."""
    recs = records_at(_block(results, "E5"), alpha, "fpr")
    if not recs:
        return None

    def split_of(r: dict) -> str | None:
        return _match_label(
            r,
            ("split", "calibration", "scheme", "mode", "condition"),
            {"grouped": ("group", "disjoint", "entity"),
             "random": ("random", "iid", "leak", "pooled")},
        )

    by_scorer = _group([r for r in recs if split_of(r) is not None], scorer_of)
    if not by_scorer:
        return None
    scorers = sorted(by_scorer)
    split_specs = (("random", -0.18, "tab:red"), ("grouped", +0.18, "tab:blue"))
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(4.6, 1.3 * len(scorers)), 3.2))
        for i, sc in enumerate(scorers):
            for split, off, color in split_specs:
                data = _values([r for r in by_scorer[sc] if split_of(r) == split], "fpr")
                if data.size == 0:
                    continue
                try:
                    if data.size < 2 or float(np.std(data)) == 0.0:
                        raise ValueError("degenerate sample for KDE")
                    vp = ax.violinplot([data], positions=[i + off], widths=0.3,
                                       showmeans=True, showextrema=False)
                    for body in vp["bodies"]:
                        body.set_facecolor(color)
                        body.set_alpha(0.55)
                    vp["cmeans"].set_color(color)
                except (ValueError, np.linalg.LinAlgError):
                    ax.plot([i + off] * data.size, data, "o", ms=3, color=color, alpha=0.7)
        ax.axhline(alpha, color="black", ls="--", lw=1.0)
        ax.set_xticks(range(len(scorers)))
        ax.set_xticklabels(scorers, rotation=12)
        ax.set_ylabel("Per-repeat empirical FPR")
        ax.set_title(f"E5 calibration leakage: random vs entity-grouped splits (alpha = {alpha:g})")
        handles = [Patch(facecolor=c, alpha=0.55, label=s) for s, _, c in split_specs]
        handles.append(Line2D([0], [0], color="black", ls="--", lw=1.0,
                              label=f"nominal alpha = {alpha:g}"))
        ax.legend(handles=handles)
        fig.tight_layout()
        return _save(fig, outdir, "f4_leakage")


def f5_mondrian(results: dict, outdir: Path | str, alpha: float = 0.05) -> Path | None:
    """E6: per-length-bin FPR, marginal vs Mondrian calibration, per scorer."""
    recs = records_at(_block(results, "E6"), alpha, "fpr")
    if not recs:
        return None

    def method_of(r: dict) -> str | None:
        return _match_label(
            r,
            ("method", "mode", "calibration", "variant"),
            {"mondrian": ("mondrian", "conditional", "binwise", "per-bin", "per_bin"),
             "marginal": ("marginal", "standard", "global", "pooled")},
        )

    method_subs = ("mondrian", "conditional", "binwise", "per-bin", "per_bin",
                   "marginal", "standard", "global", "pooled", "records", "repeats")

    def bin_of(r: dict) -> str:
        for k in ("bin", "length_bin", "len_bin", "group", "length"):
            v = r.get(k)
            if v is not None and not isinstance(v, (dict, list, tuple)):
                return str(v)
        sc = scorer_of(r)
        cands = [str(p) for p in r.get("_path", ())
                 if str(p) != sc and not any(s in str(p).lower() for s in method_subs)]
        return cands[-1] if cands else "all"

    recs = [r for r in recs if method_of(r) is not None]
    if not recs:
        return None
    by_scorer = _group(recs, scorer_of)
    with plt.rc_context(_STYLE):
        fig, axes = _subplot_grid(len(by_scorer))
        method_specs = (("marginal", -0.2, "tab:orange"), ("mondrian", +0.2, "tab:blue"))
        for ax, (sc, rs) in zip(axes, sorted(by_scorer.items())):
            bins = sorted({bin_of(r) for r in rs}, key=_bin_sort_key)
            xs = np.arange(len(bins))
            for method, off, color in method_specs:
                stats = [
                    _mean_ci(_values([r for r in rs if method_of(r) == method and bin_of(r) == b],
                                     "fpr"))
                    for b in bins
                ]
                means = np.array([s[0] for s in stats])
                halves = np.array([s[1] for s in stats])
                ok = np.isfinite(means)
                ax.bar(xs[ok] + off, means[ok], width=0.38, yerr=halves[ok], capsize=2,
                       color=color, error_kw={"lw": 0.8}, label=method)
            ax.axhline(alpha, color="black", ls="--", lw=1.0,
                       label=f"nominal alpha = {alpha:g}")
            ax.set_xticks(xs)
            ax.set_xticklabels(bins, rotation=12)
            ax.set_title(sc)
            ax.set_xlabel("Length bin")
            ax.set_ylabel("Empirical FPR")
        axes[0].legend()
        fig.suptitle(f"E6 length heterogeneity: marginal vs Mondrian calibration (alpha = {alpha:g})")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return _save(fig, outdir, "f5_mondrian")


def f6_power_frontier(results: dict, outdir: Path | str) -> Path | None:
    """E7: power (TPR) vs certified alpha (log x), per scorer; clean vs evasion linestyles."""
    recs = [r for r in iter_records(_block(results, "E7"))
            if _finite(r.get("tpr")) and _finite(r.get("alpha")) and float(r["alpha"]) > 0]
    if not recs:
        return None
    by_scorer = _group(recs, scorer_of)
    colors = _scorer_colors(sorted(by_scorer))
    linestyles = {"clean": "-", "evasion": "--"}
    seen_conds: set[str] = set()
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for sc in sorted(by_scorer):
            for cond in ("clean", "evasion"):
                crs = [r for r in by_scorer[sc] if _cond_of(r) == cond]
                if not crs:
                    continue
                seen_conds.add(cond)
                by_a = _group(crs, lambda r: float(r["alpha"]))
                alphas = sorted(by_a)
                tprs = [float(_values(by_a[a], "tpr").mean()) for a in alphas]
                ax.plot(alphas, tprs, marker="o", ms=3, lw=1.3,
                        color=colors[sc], ls=linestyles[cond])
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Certified false-accusation level alpha (log scale)")
        ax.set_ylabel("Power: P(flag | AI)")
        ax.set_title("E7 power-certificate frontier")
        scorer_handles = [Line2D([0], [0], color=colors[sc], lw=1.5, label=sc)
                          for sc in sorted(by_scorer)]
        leg1 = ax.legend(handles=scorer_handles, loc="upper left", title="scorer")
        ax.add_artist(leg1)
        if len(seen_conds) > 1:
            cond_handles = [Line2D([0], [0], color="gray", ls=linestyles[c], lw=1.5, label=c)
                            for c in sorted(seen_conds)]
            ax.legend(handles=cond_handles, loc="lower right", title="condition")
        fig.tight_layout()
        return _save(fig, outdir, "f6_power_frontier")


def f7_per_generator(results: dict, outdir: Path | str, alpha: float = 0.05) -> Path | None:
    """E2: power per RAID generator per scorer at the nominal alpha, grouped bars."""
    recs = records_at(_block(results, "E2"), alpha, "tpr")
    recs = [r for r in recs
            if _generator_of(r) is not None and "human" not in str(_generator_of(r)).lower()]
    if not recs:
        return None
    by_gen = _group(recs, _generator_of)
    gens = sorted(by_gen, key=lambda g: -float(_values(by_gen[g], "tpr").mean()))
    scorers = sorted({scorer_of(r) for r in recs})
    colors = _scorer_colors(scorers)
    width = 0.8 / len(scorers)
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(5.6, 0.9 * len(gens) * max(1, len(scorers) / 3)), 3.4))
        for j, sc in enumerate(scorers):
            for i, gen in enumerate(gens):
                rs = [r for r in by_gen[gen] if scorer_of(r) == sc]
                if not rs:
                    continue
                mean, half = _mean_ci(_values(rs, "tpr"))
                if not _finite(mean):
                    continue
                x = i - 0.4 + (j + 0.5) * width
                ax.bar(x, mean, width=width * 0.92, yerr=half, capsize=2,
                       color=colors[sc], error_kw={"lw": 0.8},
                       label=sc if i == 0 else None)
        ax.set_xticks(range(len(gens)))
        ax.set_xticklabels(gens, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"Power (TPR) at alpha = {alpha:g}")
        ax.set_title("E2 generator shift: certified power per RAID generator")
        ax.legend(ncols=2)
        fig.tight_layout()
        return _save(fig, outdir, "f7_per_generator")


# ------------------------------ orchestration ----------------------------- #

def make_all_figures(
    results: dict,
    outdir: Path | str,
    alpha: float = 0.05,
    edit_rate: float = 0.3,
) -> list[Path]:
    """Render every figure whose experiment block exists; return saved paths.

    Each figure function is defensive: missing or malformed blocks are skipped
    (with a stderr note) so a partial results dict still yields the figures it
    can support.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, Callable[[], Path | None]]] = [
        ("f1_validity_envelope", lambda: f1_validity_envelope(results, outdir, alpha=alpha)),
        ("f2_certificate_map", lambda: f2_certificate_map(results, outdir, alpha=alpha,
                                                          edit_rate=edit_rate)),
        ("f3_violation_vs_rate", lambda: f3_violation_vs_rate(results, outdir, alpha=alpha)),
        ("f4_leakage", lambda: f4_leakage(results, outdir, alpha=alpha)),
        ("f5_mondrian", lambda: f5_mondrian(results, outdir, alpha=alpha)),
        ("f6_power_frontier", lambda: f6_power_frontier(results, outdir)),
        ("f7_per_generator", lambda: f7_per_generator(results, outdir, alpha=alpha)),
    ]
    paths: list[Path] = []
    for name, job in jobs:
        try:
            p = job()
        except Exception as exc:  # never let one figure kill the run
            print(f"[viz] {name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if p is None:
            print(f"[viz] {name} skipped: required results block missing or empty",
                  file=sys.stderr)
        else:
            paths.append(p)
    return paths


# ------------------------------ summary table ----------------------------- #

def summary_rows(results: dict, alpha: float = 0.05, edit_rate: float = 0.3) -> list[dict]:
    """Per-scorer headline numbers at the nominal alpha, for the run summary.

    Returns one dict per scorer with keys: scorer, alpha, edit_rate,
    in_domain_fpr (E1), edited_fpr (E4 synonym at the rate closest to
    `edit_rate`), abstracts_fpr (E3), clean_tpr (E7 clean; falls back to E1
    tpr).  Missing quantities are NaN.
    """
    if not isinstance(results, dict):
        return []
    e1 = records_at(_block(results, "E1"), alpha, "fpr")
    e4 = [r for r in records_at(_block(results, "E4"), alpha, "fpr")
          if _attack_of(r) == "synonym"]
    e4, used_rate = _nearest_rate(e4, edit_rate)
    e3 = records_at(_block(results, "E3"), alpha, "fpr")
    e7 = [r for r in iter_records(_block(results, "E7"))
          if _finite(r.get("tpr")) and _alpha_match(r, alpha) and _cond_of(r) == "clean"]
    e1_tpr = [r for r in iter_records(_block(results, "E1"))
              if _finite(r.get("tpr")) and _alpha_match(r, alpha)]

    def mean_for(recs: list[dict], sc: str, key: str) -> float:
        vals = _values([r for r in recs if scorer_of(r) == sc], key)
        return float(vals.mean()) if vals.size else float("nan")

    scorers = sorted({scorer_of(r) for grp in (e1, e4, e3, e7, e1_tpr) for r in grp})
    rows: list[dict] = []
    for sc in scorers:
        clean_tpr = mean_for(e7, sc, "tpr")
        if not _finite(clean_tpr):
            clean_tpr = mean_for(e1_tpr, sc, "tpr")
        rows.append({
            "scorer": sc,
            "alpha": float(alpha),
            "edit_rate": float(used_rate) if used_rate is not None else float(edit_rate),
            "in_domain_fpr": mean_for(e1, sc, "fpr"),
            "edited_fpr": mean_for(e4, sc, "fpr"),
            "abstracts_fpr": mean_for(e3, sc, "fpr"),
            "clean_tpr": clean_tpr,
        })
    return rows
