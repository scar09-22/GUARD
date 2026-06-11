"""Run the full GUARD experiment suite (E1-E7), persist results, render figures.

This is the single entry point of the study: it builds the experiment
configuration from the defaults declared in ``guard.experiments.run_all``
(optionally overridden by a JSON config and/or smoke-test caps), executes the
suite, writes one numpy-safe results JSON from which every paper figure and
number regenerates (METHODOLOGY.md, section 5), renders all figures via
``guard.viz.make_all_figures``, and prints a per-scorer summary table at the
nominal certificate level.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import inspect
import json
import time
from dataclasses import asdict, is_dataclass

import numpy as np

# Nominal design levels from METHODOLOGY.md (certificate level / headline edit
# rate) -- presentation choices, not experimental results.
SUMMARY_ALPHA = 0.05
SUMMARY_EDIT_RATE = 0.3

# Smoke-mode caps: (candidate parameter names in experiments.run_all, value).
# Only parameters actually present in the run_all signature are set.
_SMOKE_CAPS = (
    (("R", "n_repeats", "repeats", "n_splits"), 50),
    (("max_gpt2_texts", "max_texts", "gpt2_max_texts"), 400),
    (("edit_rates", "rates"), [0.3]),
    (("edit_rate",), 0.3),
)


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder tolerant of numpy scalars/arrays, dataclasses, Paths, sets."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (set, frozenset)):
            return sorted(o, key=str)
        return super().default(o)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run GUARD experiments E1-E7, save results JSON, render figures.",
    )
    ap.add_argument("--config", type=Path, default=None,
                    help="JSON file overriding experiments.run_all keyword defaults")
    ap.add_argument("--smoke", action="store_true",
                    help="fast smoke run: R=50, max_gpt2_texts=400, single edit rate 0.3")
    ap.add_argument("--out", type=Path, default=Path("results/guard_results.json"),
                    help="output path for the results JSON")
    ap.add_argument("--figdir", type=Path, default=Path("figures"),
                    help="output directory for figures")
    return ap.parse_args(argv)


def _build_dict_config(defaults: dict, args: argparse.Namespace) -> dict:
    """Override dict for the cfg-dict style ``run_all(cfg)``: JSON config, then smoke caps.

    Keys are validated against the experiment module's DEFAULT_CFG so a typo in
    --config is reported instead of silently ignored by the dict merge.
    """
    cfg: dict = {}
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as fh:
            overrides = json.load(fh)
        if not isinstance(overrides, dict):
            raise SystemExit(f"--config {args.config} must contain a JSON object")
        for key, val in overrides.items():
            if key in defaults:
                cfg[key] = val
            else:
                print(f"[run_all] ignoring unknown config key: {key}", file=sys.stderr)
    if args.smoke:
        for names, value in _SMOKE_CAPS:
            for name in names:
                if name in defaults:
                    cfg[name] = value
                    break
    return cfg


def _build_config(sig: inspect.Signature, args: argparse.Namespace) -> dict:
    """Defaults from the run_all signature, then JSON config, then smoke caps."""
    cfg = {
        name: p.default
        for name, p in sig.parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                     for p in sig.parameters.values())

    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as fh:
            overrides = json.load(fh)
        if not isinstance(overrides, dict):
            raise SystemExit(f"--config {args.config} must contain a JSON object")
        for key, val in overrides.items():
            if key in sig.parameters or has_var_kw:
                cfg[key] = val
            else:
                print(f"[run_all] ignoring unknown config key: {key}", file=sys.stderr)

    if args.smoke:
        if "smoke" in sig.parameters:
            cfg["smoke"] = True
        for names, value in _SMOKE_CAPS:
            for name in names:
                if name in sig.parameters:
                    cfg[name] = value
                    break
    return cfg


def _check_required(sig: inspect.Signature, cfg: dict) -> None:
    required = [
        name for name, p in sig.parameters.items()
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                       inspect.Parameter.KEYWORD_ONLY)
    ]
    missing = [r for r in required if r not in cfg]
    if missing:
        raise SystemExit(
            "experiments.run_all requires parameters with no default that were "
            f"not supplied via --config: {missing}"
        )


def _fmt(x: float) -> str:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"{xf:.4f}" if np.isfinite(xf) else "n/a"


def _print_summary(rows: list[dict], alpha: float, edit_rate: float) -> None:
    print(f"\nSummary at alpha = {alpha:g} (means over repeats)")
    if not rows:
        print("  no summary rows could be extracted from the results dict")
        return
    headers = [
        "scorer",
        "in-domain FPR",
        f"edited-human FPR (syn@{edit_rate:g})",
        "abstracts FPR",
        "clean TPR",
    ]
    table = [
        [
            str(r["scorer"]),
            _fmt(r["in_domain_fpr"]),
            _fmt(r["edited_fpr"]),
            _fmt(r["abstracts_fpr"]),
            _fmt(r["clean_tpr"]),
        ]
        for r in rows
    ]
    widths = [max(len(h), *(len(row[i]) for row in table)) for i, h in enumerate(headers)]
    sep = "  "
    print(sep.join(h.ljust(w) for h, w in zip(headers, widths)))
    print(sep.join("-" * w for w in widths))
    for row in table:
        print(sep.join(c.ljust(w) for c, w in zip(row, widths)))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from guard import experiments, viz

    sig = inspect.signature(experiments.run_all)
    defaults = getattr(experiments, "DEFAULT_CFG", None)
    dict_style = list(sig.parameters) == ["cfg"] and isinstance(defaults, dict)

    if dict_style:
        # experiments.run_all takes ONE config dict merged over DEFAULT_CFG.
        cfg = _build_dict_config(defaults, args)
        print("[run_all] configuration (DEFAULT_CFG + overrides):")
        print(json.dumps({**defaults, **cfg}, indent=2, cls=NumpyJSONEncoder, default=str))
        call = lambda: experiments.run_all(cfg)  # noqa: E731
    else:
        cfg = _build_config(sig, args)
        _check_required(sig, cfg)
        print("[run_all] configuration:")
        print(json.dumps(cfg, indent=2, cls=NumpyJSONEncoder, default=str))
        call = lambda: experiments.run_all(**cfg)  # noqa: E731

    t0 = time.time()
    results = call()
    elapsed = time.time() - t0
    print(f"[run_all] experiments finished in {elapsed:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, cls=NumpyJSONEncoder)
    print(f"[run_all] results written to {args.out.resolve()}")

    fig_paths = viz.make_all_figures(results, args.figdir,
                                     alpha=SUMMARY_ALPHA, edit_rate=SUMMARY_EDIT_RATE)
    for p in fig_paths:
        print(f"[run_all] figure: {Path(p).resolve()}")

    rows = viz.summary_rows(results, alpha=SUMMARY_ALPHA, edit_rate=SUMMARY_EDIT_RATE)
    edit_rate_shown = rows[0]["edit_rate"] if rows else SUMMARY_EDIT_RATE
    _print_summary(rows, SUMMARY_ALPHA, edit_rate_shown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
