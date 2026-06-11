"""Wave-2 publication figures for the GUARD validity map.

Renders the four wave-2 figures from the single results dictionary produced by
``guard.wave2.run_all`` (or its JSON round-trip) -- no number is ever hand-entered:

  f8_transfer_matrix   cross-domain calibration transfer: heatmap of the empirical
                       false-accusation rate when the certificate is calibrated on
                       humans from one domain and deployed on test humans from another
                       (rows = calibration domain, cols = test-human domain) at the
                       nominal alpha. Diverging colormap centred at alpha; cells whose
                       violation test rejects are outlined. The diagonal is the
                       exchangeable control.
  f9_multilingual      (a) per-language in-language validity (FPR with 95% CI, which
                       must hug alpha -- the certificate is language-agnostic when
                       calibration matches deployment) and (b) the population-shift
                       result: English-calibrated certificates deployed on test humans
                       of every other language.
  f10_diagnostics      pre-deployment diagnostics built on guard.conformal:
                       (a) alarm rate of the pilot failure detector (pilot_fpr_ucb)
                       vs pilot size m per deployment condition -- the in-domain
                       control must sit at the alarm level while shifted conditions
                       trace detection-power curves; and (b) realized deployment FPR
                       vs the predicted TV bound (predicted_fpr_bounds): points on or
                       above the y = x line mean the bound held.
  f11_paraphrase       the repair: naive vs editor-aware robust calibration
                       (robust_calibration_scores) on paraphrased human text -- FPR
                       per scorer at the nominal alpha, plus the TPR cost on AI text.

Extraction follows guard.viz's defensive philosophy: wave-2 blocks may be nested
dicts keyed by domain / language / scorer / alpha, flat lists of per-repeat record
dicts, or aggregates holding parallel arrays; labels are read from explicit record
fields first and from the dict-traversal path as a fallback. A missing or malformed
block makes the corresponding figure skip cleanly (return ``None``) rather than
crash the run.
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import matplotlib

matplotlib.use("Agg")  # headless rendering; must precede pyplot import

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import ScalarFormatter

from guard.scores import GPT2_STATS, TFIDF_STAT
from guard.viz import (
    _STYLE,
    _finite,
    _group,
    _match_label,
    _mean_ci,
    _save,
    _scorer_colors,
    _subplot_grid,
)

__all__ = [
    "f8_transfer_matrix",
    "f9_multilingual",
    "f10_diagnostics",
    "f11_paraphrase",
    "make_wave2_figures",
]

# --------------------------- record extraction ---------------------------- #

# Aggregate blocks may store parallel arrays under plural keys.
_PLURAL = {
    "fprs": "fpr", "tprs": "tpr", "rates": "rate", "alarms": "alarm",
    "violation_ps": "violation_p", "alphas": "alpha", "tv_bounds": "tv_bound",
}
# Presence of any of these (after plural renaming) marks a dict as a leaf record.
_VALUE_KEYS = ("fpr", "tpr", "rate", "alarm", "alarm_rate", "tv_bound",
               "fpr_ucb", "fpr_point")
# Keys that may legitimately hold per-repeat parallel sequences inside one record.
_SEQ_KEYS = (
    "alpha", "fpr", "tpr", "rate", "violation_p", "alarm", "alarm_rate",
    "tv_bound", "tv_hat", "gamma_bound", "gamma_hat", "fpr_point", "fpr_ucb",
    "realized_fpr", "empirical_fpr", "test_pvalue", "flags", "m",
    "n_cal", "n_test_human", "n_test_ai",
)
# Path elements that are container names, never domain / language / condition labels.
_STRUCTURAL = {
    "records", "repeats", "results", "data", "per_scorer", "scorers", "by_scorer",
    "conditions", "per_condition", "meta", "summary", "per_language", "languages",
    "by_language", "language", "per_domain", "domains", "by_domain", "domain",
    "pairs", "cells", "matrix", "fpr", "tpr", "rate", "rates", "alarm", "alarms",
    "power", "human", "humans", "ai", "per_mode", "modes", "naive", "robust",
    "clean", "fpr_summary", "tpr_summary", "wave2", "diagnostics", "pilot",
    "bounds", "bound", "transfer", "multilingual", "paraphrase", "alpha", "alphas",
    "envelope", "fpr_pool_humans", "power_per_generator", "per_generator",
    "generators", "in_domain", "in_language", "cross_from_hotel",
    "cross_from_english",
}

_F8_NAMES = ("transfer", "cross_domain", "cross-domain", "domain_transfer",
             "calibration_transfer", "domains", "domain", "e9", "f8")
_F9_NAMES = ("multilingual", "cross_language", "cross-language", "language",
             "languages", "lang", "e10", "f9")
_F10_NAMES = ("diagnostic", "pilot", "alarm", "bounds", "population_shift",
              "e11", "f10")
_F11_NAMES = ("paraphrase", "robust", "editor", "e12", "e8", "f11")

_CAL_DOMAIN_FIELDS = ("cal_domain", "calibration_domain", "source_domain", "cal",
                      "from_domain", "train_domain", "cal_pop", "source")
_TEST_DOMAIN_FIELDS = ("test_domain", "target_domain", "test_human_domain",
                       "deploy_domain", "to_domain", "test_pop", "target", "test")
_LANG_FIELDS = ("language", "lang", "test_language", "test_lang", "review_language")
_CAL_LANG_FIELDS = ("cal_language", "cal_lang", "calibration_language",
                    "source_language")
_IN_TAGS = ("in_lang", "in-lang", "inlang", "in language", "within", "native",
            "monolingual", "same_lang", "self")
_CROSS_TAGS = ("cross", "english_cal", "en_cal", "en-cal", "english-cal",
               "english_calibrated", "from_english", "transfer")

_M_RE = re.compile(r"^m\s*[=_]?\s*(\d+)$", re.IGNORECASE)


def _is_value(v: Any) -> bool:
    """True iff v is a scalar value or a sequence of scalars (not nested records)."""
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        for item in v:
            if item is None:
                continue
            return not isinstance(item, (dict, list, tuple))
        return False
    return _finite(v)


def _explode(rec: dict) -> Iterator[dict]:
    """Expand a record whose value fields are parallel sequences into flat rows."""
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


def _iter_w2(block: Any, _path: tuple[str, ...] = ()) -> Iterator[dict]:
    """Yield flat per-repeat record dicts from an arbitrarily nested wave-2 block.

    A record is any dict (or dataclass) carrying at least one wave-2 value key
    (fpr / tpr / rate / alarm / tv_bound / fpr_ucb / ...), scalar or sequence.
    Each yielded dict gains a ``_path`` tuple of the dict keys traversed to reach
    it, used as a fallback for scorer / domain / language / condition labels.
    """
    if block is None:
        return
    if is_dataclass(block) and not isinstance(block, type):
        yield from _explode({**asdict(block), "_path": _path})
        return
    if isinstance(block, dict):
        rec = {_PLURAL.get(k, k): v for k, v in block.items()}
        if any(k in rec and _is_value(rec[k]) for k in _VALUE_KEYS):
            rec["_path"] = _path
            yield from _explode(rec)
            return
        for key, val in block.items():
            yield from _iter_w2(val, _path + (str(key),))
        return
    if isinstance(block, (list, tuple)):
        for item in block:
            yield from _iter_w2(item, _path)


def _find_block(results: Any, names: Sequence[str], depth: int = 2) -> Any:
    """First sub-dict of `results` whose key matches one of `names` (two levels deep)."""
    if not isinstance(results, dict) or depth <= 0:
        return None
    for nm in names:
        for k, v in results.items():
            kl = str(k).lower()
            if kl == "meta":
                continue
            if kl == nm or kl.startswith(nm) or nm in kl:
                return v
    for v in results.values():
        if isinstance(v, dict):
            hit = _find_block(v, names, depth - 1)
            if hit is not None:
                return hit
    return None


def _candidate_blocks(results: Any, names: Sequence[str]) -> Iterator[Any]:
    """Blocks to try in order: the name-matched block, then the whole results dict.

    Name matching can hit a spurious sub-dict (or the experiment may live at an
    unexpected key), so callers take the FIRST candidate that yields usable
    records and fall back to scanning everything -- the figure-specific record
    signatures (domains, languages, alarms, modes) keep the full scan precise.
    """
    blk = _find_block(results, names)
    if blk is not None:
        yield blk
    if isinstance(results, (dict, list, tuple)):
        yield results


def _at_alpha(recs: list[dict], alpha: float) -> list[dict]:
    """Records at the given alpha; records carrying no alpha field are kept (lenient)."""
    out = []
    for r in recs:
        a = r.get("alpha")
        if a is None or not _finite(a) or math.isclose(
            float(a), float(alpha), rel_tol=1e-6, abs_tol=1e-9
        ):
            out.append(r)
    return out


def _value_of(rec: dict, keys: Sequence[str] = ("fpr", "rate")) -> float | None:
    """First finite value among `keys`, in priority order."""
    for k in keys:
        v = rec.get(k)
        if _finite(v):
            return float(v)
    return None


def _field(rec: dict, names: Sequence[str]) -> str | None:
    """First non-empty scalar field among `names`, as a string."""
    for k in names:
        v = rec.get(k)
        if v is not None and not isinstance(v, (dict, list, tuple)):
            s = str(v).strip()
            if s:
                return s
    return None


def _known_scorers(recs: list[dict]) -> set[str]:
    """Lowercase scorer vocabulary: canonical names plus any explicit record fields."""
    known = {s.lower() for s in GPT2_STATS} | {TFIDF_STAT.lower()}
    for r in recs:
        for k in ("scorer", "score_name", "detector"):
            v = r.get(k)
            if isinstance(v, str) and v:
                known.add(v.lower())
    return known


def _scorer_w2(rec: dict, known: set[str]) -> str:
    """Scorer label: explicit field first, then a path element from the known set."""
    for k in ("scorer", "score_name", "detector"):
        v = rec.get(k)
        if isinstance(v, str) and v:
            return v
    for p in rec.get("_path", ()):
        if str(p).lower() in known:
            return str(p)
    return "scorer"


def _path_candidates(
    rec: dict, known: set[str], exclude_subs: Sequence[str] = ()
) -> list[str]:
    """Path elements plausible as domain / language / condition labels.

    Filters out structural container names, known scorer names, anything containing
    a digit (alpha / rate / m keys), and elements containing any `exclude_subs`.
    """
    out: list[str] = []
    for p in rec.get("_path", ()):
        s = str(p).strip()
        low = s.lower()
        if not s or low in _STRUCTURAL or low in known:
            continue
        if any(ch.isdigit() for ch in s):
            continue
        if any(sub in low for sub in exclude_subs):
            continue
        out.append(s)
    return out


def _violated(
    rs: list[dict],
    alpha: float,
    level: float,
    value_keys: Sequence[str] = ("fpr", "rate"),
) -> bool:
    """Certificate-violation call for one cell of repeats.

    Primary criterion: mean exact violation p-value (Beta-Binomial tail, computed
    upstream) below `level`. Fallback when no p-values were stored: the 95% CI of
    the mean FPR lies entirely above alpha.
    """
    vio = np.asarray([float(r["violation_p"]) for r in rs if _finite(r.get("violation_p"))])
    if vio.size:
        return bool(float(vio.mean()) < level)
    vals = np.asarray(
        [v for v in (_value_of(r, value_keys) for r in rs) if v is not None], dtype=float
    )
    if vals.size == 0:
        return False
    mean, half = _mean_ci(vals)
    return bool(_finite(mean) and mean - half > alpha)


# --------------------------- plotting utilities --------------------------- #

def _grouped_bars(
    ax,
    cats: list[str],
    labels: Sequence[str],
    colors: dict[str, Any],
    recs_for: Callable[[str, str], list[dict]],
    value_keys: Sequence[str],
    hline: float | None,
    violation_level: float,
) -> bool:
    """Grouped bars (mean, 95% CI) of `value_keys` per (category, series label).

    When `hline` (the nominal alpha) is given, it is drawn and violated bars are
    hatched and starred. Returns True iff any bar was marked violated.
    """
    width = 0.8 / max(1, len(labels))
    any_violated = False
    for j, lab in enumerate(labels):
        first = True
        for i, cat in enumerate(cats):
            rs = recs_for(cat, lab)
            if not rs:
                continue
            vals = np.asarray(
                [v for v in (_value_of(r, value_keys) for r in rs) if v is not None],
                dtype=float,
            )
            mean, half = _mean_ci(vals)
            if not _finite(mean):
                continue
            x = i - 0.4 + (j + 0.5) * width
            bars = ax.bar(x, mean, width=width * 0.92, yerr=half, capsize=2,
                          color=colors[lab], error_kw={"lw": 0.8},
                          label=lab if first else None)
            first = False
            if hline is not None and _violated(rs, hline, violation_level, value_keys):
                any_violated = True
                bars[0].set_hatch("///")
                bars[0].set_edgecolor("black")
                ax.annotate("*", (x, mean + half), ha="center", va="bottom",
                            fontsize=10, color="black")
    if hline is not None:
        ax.axhline(hline, color="black", ls="--", lw=1.0)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=25, ha="right")
    return any_violated


def _no_data(ax, what: str) -> None:
    ax.text(0.5, 0.5, f"no {what} in results", ha="center", va="center",
            transform=ax.transAxes, fontsize=9, color="0.4")
    ax.set_axis_off()


# --------------------------------- figures -------------------------------- #

def _domains_of(rec: dict, known: set[str]) -> tuple[str, str] | None:
    """(calibration domain, test-human domain) of one transfer record.

    Explicit fields first; then 'A->B' or 'cal=A' / 'test=B' path elements; then
    the first two plausible path labels in traversal order (outer dict key =
    calibration domain by convention).
    """
    cal = _field(rec, _CAL_DOMAIN_FIELDS)
    test = _field(rec, _TEST_DOMAIN_FIELDS)
    if cal and test:
        return cal, test
    pcal: str | None = None
    ptest: str | None = None
    for p in rec.get("_path", ()):
        s = str(p).replace("→", "->")
        if "->" in s:
            a, b = s.split("->", 1)
            if a.strip() and b.strip():
                return a.strip(), b.strip()
        if "=" in s:
            k, v = s.split("=", 1)
            kl, v = k.strip().lower(), v.strip()
            if v and any(t in kl for t in ("cal", "source", "from", "train")):
                pcal = v
            elif v and any(t in kl for t in ("test", "target", "to", "deploy")):
                ptest = v
    cal, test = cal or pcal, test or ptest
    if cal and test:
        return cal, test
    cands = _path_candidates(rec, known, exclude_subs=("matrix",))
    if cal:
        rest = [c for c in cands if c != cal]
        return (cal, rest[0]) if rest else None
    if test:
        rest = [c for c in cands if c != test]
        return (rest[0], test) if rest else None
    if len(cands) >= 2:
        return cands[0], cands[1]
    return None


def f8_transfer_matrix(
    results: dict,
    outdir: Path | str,
    alpha: float = 0.05,
    violation_level: float = 0.05,
) -> Path | None:
    """Cross-domain calibration-transfer FPR heatmap at the nominal alpha.

    One heatmap per scorer: rows = calibration domain, columns = test-human
    domain, cell = mean empirical FPR over repeats, annotated. Diverging colormap
    centred at alpha (blue = conservative, red = inflated); cells whose exact
    violation test rejects (or whose CI clears alpha when p-values are absent)
    are outlined in black. The diagonal is the exchangeable control.
    """
    rows: list[tuple[str, str, dict]] = []
    known: set[str] = set()
    for block in _candidate_blocks(results, _F8_NAMES):
        recs = [r for r in _at_alpha(list(_iter_w2(block)), alpha)
                if _value_of(r, ("fpr", "rate")) is not None]
        known = _known_scorers(recs)
        rows = []
        for r in recs:
            doms = _domains_of(r, known)
            if doms is not None:
                rows.append((doms[0], doms[1], r))
        if rows:
            break
    if not rows:
        return None
    by_scorer: dict[str, list[tuple[str, str, dict]]] = {}
    for cal, test, r in rows:
        by_scorer.setdefault(_scorer_w2(r, known), []).append((cal, test, r))
    scorers = sorted(by_scorer)
    cal_doms = sorted({c for c, _, _ in rows})
    test_doms = sorted({t for _, t, _ in rows})

    mats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    vmax = 2.0 * float(alpha)
    for sc in scorers:
        cellmap: dict[tuple[str, str], list[dict]] = {}
        for cal, test, r in by_scorer[sc]:
            cellmap.setdefault((cal, test), []).append(r)
        M = np.full((len(cal_doms), len(test_doms)), np.nan)
        V = np.zeros(M.shape, dtype=bool)
        for (cal, test), rs in cellmap.items():
            i, j = cal_doms.index(cal), test_doms.index(test)
            vals = np.asarray(
                [v for v in (_value_of(rr, ("fpr", "rate")) for rr in rs) if v is not None],
                dtype=float,
            )
            if vals.size == 0:
                continue
            M[i, j] = float(vals.mean())
            V[i, j] = _violated(rs, alpha, violation_level)
            vmax = max(vmax, float(M[i, j]))
        mats[sc] = (M, V)

    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=float(alpha), vmax=vmax * 1.02)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.88")
    with plt.rc_context(_STYLE):
        w = max(3.2, 0.85 * len(test_doms) + 1.8)
        h = max(2.6, 0.65 * len(cal_doms) + 1.5)
        fig, axes = _subplot_grid(len(scorers), width=w, height=h)
        fig.subplots_adjust(wspace=0.3)
        ncols = min(3, max(1, len(scorers)))
        im = None
        for k, (ax, sc) in enumerate(zip(axes, scorers)):
            M, V = mats[sc]
            im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, aspect="auto")
            ax.grid(False)
            ax.set_xticks(range(len(test_doms)))
            ax.set_xticklabels(test_doms, rotation=25, ha="right")
            ax.set_yticks(range(len(cal_doms)))
            ax.set_yticklabels(cal_doms)
            for i in range(len(cal_doms)):
                for j in range(len(test_doms)):
                    if not np.isfinite(M[i, j]):
                        continue
                    frac = float(norm(M[i, j]))
                    txt_color = "white" if (frac > 0.8 or frac < 0.15) else "black"
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                            fontsize=7, color=txt_color)
                    if V[i, j]:
                        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="black", lw=1.8))
            ax.set_title(sc)
            ax.set_xlabel("test-human domain")
            if k % ncols == 0:
                ax.set_ylabel("calibration domain")
        cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
        cbar.set_label(f"empirical FPR (centre = alpha = {alpha:g})")
        fig.suptitle(
            f"Cross-domain calibration transfer: false-accusation rate "
            f"(alpha = {alpha:g}; outlined = violated)"
        )
        return _save(fig, outdir, "f8_transfer_matrix")


def _f9_condition(rec: dict) -> str | None:
    """'in' (in-language calibration) vs 'cross' (English-calibrated transfer)."""
    cl = _field(rec, _CAL_LANG_FIELDS)
    tl = _field(rec, _LANG_FIELDS)
    if cl and tl:
        return "in" if cl.lower() == tl.lower() else "cross"
    return _match_label(
        rec,
        ("condition", "setting", "mode", "variant", "split", "calibration"),
        {"in": _IN_TAGS, "cross": _CROSS_TAGS},
    )


def _language_of(rec: dict, known: set[str]) -> str | None:
    """Test-human language: explicit field, else a plausible path element."""
    v = _field(rec, _LANG_FIELDS)
    if v:
        return v
    cands = _path_candidates(rec, known,
                             exclude_subs=_IN_TAGS + _CROSS_TAGS + ("lang", "calib"))
    return cands[0] if cands else None


def f9_multilingual(
    results: dict,
    outdir: Path | str,
    alpha: float = 0.05,
    violation_level: float = 0.05,
) -> Path | None:
    """Multilingual validity: in-language calibration vs English-calibrated transfer.

    Panel (a): per-language FPR (mean, 95% CI over repeats) when calibration humans
    are drawn from the SAME language -- exchangeability holds, so every bar must hug
    alpha regardless of language. Panel (b): the population-shift result -- the same
    certificate calibrated on English humans and deployed on each other language's
    test humans; violated bars are hatched and starred.
    """
    panels: dict[str, list[tuple[str, dict]]] = {"in": [], "cross": []}
    known: set[str] = set()
    for block in _candidate_blocks(results, _F9_NAMES):
        # Full language x language matrix cells are excluded: panel (b) is the
        # English-calibrated row only, which the cross_from_* records carry.
        recs = [r for r in _at_alpha(list(_iter_w2(block)), alpha)
                if _value_of(r, ("fpr", "rate")) is not None
                and not any("matrix" in str(p).lower() for p in r.get("_path", ()))]
        known = _known_scorers(recs)
        panels = {"in": [], "cross": []}
        for r in recs:
            cond = _f9_condition(r)
            if cond not in panels:
                continue
            lang = _language_of(r, known)
            if lang:
                panels[cond].append((lang, r))
        if panels["in"] or panels["cross"]:
            break
    if not panels["in"] and not panels["cross"]:
        return None
    scorers = sorted({_scorer_w2(r, known) for grp in panels.values() for _, r in grp})
    colors = _scorer_colors(scorers)
    n_lang = max(1, len({l for grp in panels.values() for l, _ in grp}))
    with plt.rc_context(_STYLE):
        w = max(4.0, min(9.0, 0.2 * n_lang * max(1, len(scorers)) + 2.2))
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(2 * w, 3.4))
        titles = {
            "in": "(a) in-language calibration: validity per language",
            "cross": "(b) English-calibrated, deployed cross-language",
        }
        any_violated = False
        for ax, key in ((ax_a, "in"), (ax_b, "cross")):
            grp = panels[key]
            if not grp:
                _no_data(ax, f"{titles[key]} records")
                continue
            langs = sorted({l for l, _ in grp})

            def recs_for(cat: str, lab: str, _grp=grp) -> list[dict]:
                return [r for l, r in _grp if l == cat and _scorer_w2(r, known) == lab]

            v = _grouped_bars(ax, langs, scorers, colors, recs_for,
                              ("fpr", "rate"), alpha, violation_level)
            any_violated = any_violated or v
            ax.set_ylabel("Empirical FPR (mean, 95% CI)")
            ax.set_xlabel("Test-human language")
            ax.set_title(titles[key])
        handles = [Patch(facecolor=colors[s], label=s) for s in scorers]
        handles.append(Line2D([0], [0], color="black", ls="--", lw=1.0,
                              label=f"nominal alpha = {alpha:g}"))
        if any_violated:
            handles.append(Patch(facecolor="white", edgecolor="black", hatch="///",
                                 label=f"* violated (mean violation p < {violation_level:g})"))
        leg_ax = ax_a if panels["in"] else ax_b
        leg_ax.set_ylim(0, leg_ax.get_ylim()[1] * 1.45)  # headroom for the legend
        leg_ax.legend(handles=handles, ncols=2, loc="upper left")
        fig.suptitle(
            f"Multilingual certificate: in-language validity vs English-calibrated "
            f"population shift (alpha = {alpha:g})"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        return _save(fig, outdir, "f9_multilingual")


def _m_of(rec: dict) -> int | None:
    """Pilot size m: explicit field, else an 'm=50'-style or bare-integer path element."""
    for k in ("m", "pilot_m", "pilot_size", "n_pilot", "m_pilot"):
        v = rec.get(k)
        if _finite(v) and float(v) > 0:
            return int(float(v))
    for p in rec.get("_path", ()):
        s = str(p).strip()
        mt = _M_RE.match(s)
        if mt:
            return int(mt.group(1))
        if s.isdigit():
            return int(s)
    return None


def _cond_w2(rec: dict, known: set[str]) -> str:
    """Deployment-condition label: explicit field first, then a path element."""
    v = _field(rec, ("condition", "shift", "population", "test_set", "scenario",
                     "setting", "domain", "target"))
    if v:
        return v
    cands = _path_candidates(rec, known, exclude_subs=("alarm", "pilot", "bound",
                                                       "diagnost", "tv"))
    return cands[0] if cands else "all"


def _alarm_of(rec: dict) -> float:
    v = rec.get("alarm_rate")
    if _finite(v):
        return float(v)
    return float(rec.get("alarm"))


def f10_diagnostics(results: dict, outdir: Path | str, alpha: float = 0.05) -> Path | None:
    """Pre-deployment diagnostics: pilot-alarm power curves and TV-bound validity.

    Panel (a): alarm rate (fraction of repeats where the exact binomial pilot test
    rejects H0: FPR <= alpha) vs pilot size m, one curve per deployment condition.
    Calibrated by construction: the in-domain control curve must sit at the alarm
    level for every m, while shifted conditions rise toward 1 -- the failure
    detector's power curve. Panel (b): realized deployment FPR (x) against the
    predicted TV bound alpha + TV_hat (y); points on or above y = x mean the
    plug-in bound was valid for that run.
    """
    alarm_recs: list[dict] = []
    bound_recs: list[dict] = []
    known: set[str] = set()
    for block in _candidate_blocks(results, _F10_NAMES):
        recs = _at_alpha(list(_iter_w2(block)), alpha)
        known = _known_scorers(recs)
        alarm_recs = [r for r in recs
                      if (_finite(r.get("alarm")) or _finite(r.get("alarm_rate")))
                      and _m_of(r) is not None]
        bound_recs = [r for r in recs if _finite(r.get("tv_bound"))
                      and _value_of(r, ("realized_fpr", "empirical_fpr", "fpr",
                                        "fpr_point", "rate")) is not None]
        if alarm_recs or bound_recs:
            break
    if not alarm_recs and not bound_recs:
        return None
    cmap = plt.get_cmap("tab10")
    with plt.rc_context(_STYLE):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.4, 3.5))

        if alarm_recs:
            conds = sorted({_cond_w2(r, known) for r in alarm_recs})
            ccolors = {c: cmap(i % 10) for i, c in enumerate(conds)}
            for cond in conds:
                crs = [r for r in alarm_recs if _cond_w2(r, known) == cond]
                by_m = _group(crs, _m_of)
                ms = sorted(by_m)
                stats = [_mean_ci(np.asarray([_alarm_of(r) for r in by_m[m]], dtype=float))
                         for m in ms]
                means = np.array([s[0] for s in stats])
                halves = np.array([s[1] for s in stats])
                ax_a.plot(ms, means, marker="o", ms=3.5, lw=1.3,
                          color=ccolors[cond], label=cond)
                ax_a.fill_between(ms, np.clip(means - halves, 0, 1),
                                  np.clip(means + halves, 0, 1),
                                  color=ccolors[cond], alpha=0.15, lw=0)
            levels = [float(r["alarm_level"]) for r in alarm_recs
                      if _finite(r.get("alarm_level"))]
            level = float(np.mean(levels)) if levels else 0.10
            ax_a.axhline(level, color="black", ls="--", lw=1.0,
                         label=f"alarm level = {level:g}")
            ms_all = sorted({m for m in (_m_of(r) for r in alarm_recs) if m})
            if len(ms_all) > 1 and ms_all[-1] / max(ms_all[0], 1) >= 8:
                ax_a.set_xscale("log")
                ax_a.set_xticks(ms_all)
                ax_a.xaxis.set_major_formatter(ScalarFormatter())
                ax_a.minorticks_off()
            ax_a.set_xlabel("Pilot size m (known-human texts)")
            ax_a.set_ylabel("Alarm rate over repeats")
            ax_a.set_ylim(-0.02, 1.05)
            ax_a.set_title("(a) pilot failure-detector power vs pilot size")
            ax_a.legend()
        else:
            _no_data(ax_a, "pilot-alarm records")

        if bound_recs:
            conds = sorted({_cond_w2(r, known) for r in bound_recs})
            ccolors = {c: cmap(i % 10) for i, c in enumerate(conds)}
            top = float(alpha)
            n_pts, n_ok = 0, 0
            for cond in conds:
                xs, ys = [], []
                for r in bound_recs:
                    if _cond_w2(r, known) != cond:
                        continue
                    x = _value_of(r, ("realized_fpr", "empirical_fpr", "fpr",
                                      "fpr_point", "rate"))
                    y = float(r["tv_bound"])
                    xs.append(x)
                    ys.append(y)
                    n_pts += 1
                    n_ok += int(y >= x)
                ax_b.scatter(xs, ys, s=12, alpha=0.55, color=ccolors[cond],
                             label=cond, edgecolors="none")
                top = max([top] + xs + ys)
            lim = top * 1.08 + 1e-3
            ax_b.plot([0, lim], [0, lim], color="black", ls="--", lw=1.0,
                      label="y = x (bound = realized)")
            ax_b.set_xlim(0, lim)
            ax_b.set_ylim(0, lim)
            if n_pts:
                ax_b.text(0.97, 0.04,
                          f"bound >= realized FPR in {100.0 * n_ok / n_pts:.0f}% of runs",
                          transform=ax_b.transAxes, ha="right", va="bottom", fontsize=8)
            ax_b.set_xlabel("Realized deployment FPR")
            ax_b.set_ylabel("Predicted TV bound (alpha + TV-hat)")
            ax_b.set_title("(b) population-shift bound vs realized FPR")
            ax_b.legend(loc="upper left")
        else:
            _no_data(ax_b, "TV-bound records")

        fig.suptitle(
            f"Pre-deployment diagnostics for the certificate (alpha = {alpha:g})"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        return _save(fig, outdir, "f10_diagnostics")


def f11_paraphrase(
    results: dict,
    outdir: Path | str,
    alpha: float = 0.05,
    violation_level: float = 0.05,
) -> Path | None:
    """Naive vs editor-aware robust calibration on paraphrased humans, plus power cost.

    Panel (a): per scorer, grouped bars of the empirical FPR on paraphrased human
    text under naive (clean-score) vs robust (max-over-modeled-edits,
    guard.conformal.robust_calibration_scores) calibration at the nominal alpha;
    violated bars hatched and starred. Panel (b): the price -- TPR on AI text under
    both calibrations, with the per-scorer power cost (naive minus robust) annotated.
    Falls back to the dominant non-clean human condition if no paraphrase-tagged
    condition exists (the condition used is named in the panel title).
    """
    def mode_of(r: dict) -> str | None:
        return _match_label(
            r,
            ("mode", "calibration", "cal_mode", "method", "variant"),
            {"naive": ("naive", "standard", "plain", "baseline"),
             "robust": ("robust", "editor", "aware", "max")},
        )

    recs: list[dict] = []
    known: set[str] = set()
    for block in _candidate_blocks(results, _F11_NAMES):
        cand = [r for r in _at_alpha(list(_iter_w2(block)), alpha)
                if _value_of(r, ("fpr", "tpr", "rate")) is not None]
        known = _known_scorers(cand)
        recs = [r for r in cand if mode_of(r) is not None]
        if recs:
            break
    if not recs:
        return None

    def cond_str(r: dict) -> str:
        v = _field(r, ("condition", "population", "test_set", "split", "target",
                       "setting"))
        if v:
            return v
        return "/".join(str(p) for p in r.get("_path", ()))

    def is_human(r: dict) -> bool:
        c = cond_str(r).lower()
        if "human" in c:
            return True
        if c.startswith("ai") or "_ai" in c or "machine" in c:
            return False
        return _finite(r.get("fpr"))

    def is_ai(r: dict) -> bool:
        c = cond_str(r).lower()
        if c.startswith("ai") or "_ai" in c or "machine" in c:
            return True
        if "human" in c:
            return False
        return _finite(r.get("tpr"))

    clean_re = re.compile(r"clean(?!up)")  # 'clean' but not 'cleanup' (an edit kind)
    human_recs = [r for r in recs if is_human(r)]
    para_tags = ("paraphr", "pegasus", "dipper", "reword", "rewrit", "backtrans",
                 "back_trans")
    para = [r for r in human_recs if any(t in cond_str(r).lower() for t in para_tags)]
    if not para:  # fall back to the dominant edited (non-clean) human condition
        para = [r for r in human_recs if not clean_re.search(cond_str(r).lower())]
    if not para:
        return None
    cond_h = Counter(cond_str(r) for r in para).most_common(1)[0][0]
    para = [r for r in para if cond_str(r) == cond_h]

    ai_recs = [r for r in recs if is_ai(r)]
    ai_pref = [r for r in ai_recs if clean_re.search(cond_str(r).lower())] or ai_recs
    cond_a = Counter(cond_str(r) for r in ai_pref).most_common(1)[0][0] if ai_pref else None
    ai_use = [r for r in ai_pref if cond_str(r) == cond_a]

    scorers = sorted({_scorer_w2(r, known) for r in para + ai_use})
    modes = ("naive", "robust")
    mode_colors = {"naive": "tab:orange", "robust": "tab:blue"}

    with plt.rc_context(_STYLE):
        w = max(4.0, 0.65 * len(scorers) + 2.4)
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(2 * w, 3.5))

        def recs_for_fpr(sc: str, mode: str) -> list[dict]:
            return [r for r in para if _scorer_w2(r, known) == sc and mode_of(r) == mode]

        any_violated = _grouped_bars(ax_a, scorers, modes, mode_colors, recs_for_fpr,
                                     ("fpr", "rate"), alpha, violation_level)
        ax_a.set_ylabel("Empirical FPR (mean, 95% CI)")
        ax_a.set_title(f"(a) FPR on '{cond_h}' (alpha = {alpha:g})")

        if ai_use:
            def recs_for_tpr(sc: str, mode: str) -> list[dict]:
                return [r for r in ai_use
                        if _scorer_w2(r, known) == sc and mode_of(r) == mode]

            _grouped_bars(ax_b, scorers, modes, mode_colors, recs_for_tpr,
                          ("tpr", "rate"), None, violation_level)
            for i, sc in enumerate(scorers):
                tops, means = [], {}
                for mode in modes:
                    vals = np.asarray(
                        [v for v in (_value_of(r, ("tpr", "rate"))
                                     for r in recs_for_tpr(sc, mode)) if v is not None],
                        dtype=float,
                    )
                    mean, half = _mean_ci(vals)
                    if _finite(mean):
                        means[mode] = mean
                        tops.append(mean + half)
                if len(means) == 2:
                    cost = means["naive"] - means["robust"]
                    ax_b.annotate(f"cost {cost:+.2f}", (i, max(tops) + 0.02),
                                  ha="center", va="bottom", fontsize=7)
            ax_b.set_ylim(0, 1.1)
            ax_b.set_ylabel("Power: P(flag | AI)")
            ax_b.set_title(f"(b) TPR on '{cond_a}' and the power cost of robustness")
        else:
            _no_data(ax_b, "AI-power records")

        handles = [Patch(facecolor=mode_colors[m], label=f"{m} calibration")
                   for m in modes]
        handles.append(Line2D([0], [0], color="black", ls="--", lw=1.0,
                              label=f"nominal alpha = {alpha:g}"))
        if any_violated:
            handles.append(Patch(facecolor="white", edgecolor="black", hatch="///",
                                 label=f"* violated (mean violation p < {violation_level:g})"))
        ax_a.set_ylim(0, ax_a.get_ylim()[1] * 1.4)  # headroom for the legend
        ax_a.legend(handles=handles, loc="upper left")
        fig.suptitle(
            "Editor-aware robust calibration vs paraphrased humans: "
            "validity repaired, power cost measured"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        return _save(fig, outdir, "f11_paraphrase")


# ------------------------------ orchestration ----------------------------- #

def make_wave2_figures(
    results: dict,
    outdir: Path | str,
    alpha: float = 0.05,
) -> list[str]:
    """Render every wave-2 figure whose results block exists; return saved paths.

    Each figure function is defensive: a missing or malformed block is skipped
    (with a stderr note) so a partial wave-2 results dict still yields the
    figures it can support. Paths are returned as strings, in figure order.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, Callable[[], Path | None]]] = [
        ("f8_transfer_matrix", lambda: f8_transfer_matrix(results, outdir, alpha=alpha)),
        ("f9_multilingual", lambda: f9_multilingual(results, outdir, alpha=alpha)),
        ("f10_diagnostics", lambda: f10_diagnostics(results, outdir, alpha=alpha)),
        ("f11_paraphrase", lambda: f11_paraphrase(results, outdir, alpha=alpha)),
    ]
    paths: list[str] = []
    for name, job in jobs:
        try:
            p = job()
        except Exception as exc:  # never let one figure kill the run
            print(f"[viz_wave2] {name} failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue
        if p is None:
            print(f"[viz_wave2] {name} skipped: required results block missing or empty",
                  file=sys.stderr)
        else:
            paths.append(str(p))
    return paths
