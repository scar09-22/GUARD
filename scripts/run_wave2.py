"""Run the GUARD wave-2 study (W1-W4), persist results, render figures.

Single entry point for the wave-2 suite (guard.wave2.run_all): multilingual validity
and cross-language calibration transfer (W1), cross-domain transfer over the RAID
domains (W2), the paraphrase-model certificate/evasion attack with editor-aware robust
calibration (W3), and the pre-deployment pilot diagnostics evaluated against realized
FPR under every shift condition (W4).

Writes one numpy-safe results JSON (serialised with the same ``_jsonable`` helper as
the wave-1 engine) from which every downstream figure and number regenerates, prints a
compact W4 summary table (per shift condition: realized FPR at alpha = 0.05, pilot
alarm rate at m = 100, mean plug-in TV bound), and renders figures via
``guard.viz_wave2.make_wave2_figures`` when that sibling module is importable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import time

# Presentation choice for the printed summary (not an experimental result).
SUMMARY_ALPHA = 0.05
SUMMARY_PILOT_M = 100

# Smoke caps: small enough to exercise every code path on a laptop without
# touching the network (fetched domains are used only if already cached).
_SMOKE_OVERRIDES = {
    "R": 25,
    "lang_cap_human": 80,
    "lang_cap_ai": 40,
    "max_para": 40,
    "para_ai_cap": 24,
    "fetch_if_missing": False,
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run GUARD wave-2 studies W1-W4, save results JSON, render figures.",
    )
    ap.add_argument("--out", type=Path, default=Path("results/guard_wave2.json"),
                    help="output path for the results JSON")
    ap.add_argument("--figdir", type=Path, default=Path("figures"),
                    help="output directory for figures (guard.viz_wave2)")
    ap.add_argument("--smoke", action="store_true",
                    help="fast smoke run: R=25, ~80 humans/40 AI per pool, no network fetch")
    ap.add_argument("--with_supervised", action="store_true",
                    help="add the 'supervised' and 'offshelf' scorers from guard.supervised")
    ap.add_argument("--device", default="mps",
                    help="inference device for GPT-2 / paraphraser / supervised (default: mps)")
    ap.add_argument("--max_para", type=int, default=None,
                    help="cap on paraphrased humans in W3 (default: engine default, 500)")
    ap.add_argument("--R", type=int, default=None,
                    help="number of calibration/test resamples (default: engine default, 200)")
    ap.add_argument("--seed", type=int, default=None,
                    help="master seed (default: engine default, 0)")
    ap.add_argument("--config", type=Path, default=None,
                    help="JSON file overriding guard.wave2.DEFAULT_CFG keys")
    return ap.parse_args(argv)


def _build_cfg(args: argparse.Namespace, defaults: dict) -> dict:
    """Override dict merged over wave2.DEFAULT_CFG: JSON config, CLI flags, smoke caps."""
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
                print(f"[run_wave2] ignoring unknown config key: {key}", file=sys.stderr)
    cfg["device"] = args.device
    cfg["with_supervised"] = bool(args.with_supervised)
    if args.max_para is not None:
        cfg["max_para"] = int(args.max_para)
    if args.R is not None:
        cfg["R"] = int(args.R)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.smoke:
        cfg.update(_SMOKE_OVERRIDES)
        if args.max_para is not None:  # explicit CLI caps win over the smoke caps
            cfg["max_para"] = int(args.max_para)
        if args.R is not None:
            cfg["R"] = int(args.R)
    return cfg


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _print_w4_summary(results: dict, alpha: float, pilot_m: int) -> None:
    """Compact per-condition diagnostics table at the nominal certificate level."""
    w4 = results.get("w4_diagnostics") or {}
    conditions = w4.get("conditions") or {}
    akey = f"{alpha:g}"
    rows: list[list[str]] = []
    for cond_name, per_scorer in conditions.items():
        for scorer, per_alpha in per_scorer.items():
            rec = per_alpha.get(akey)
            if rec is None:
                continue
            alarm = (rec.get("alarm_rate") or {}).get(str(pilot_m))
            violated = rec.get("violated")
            rows.append([
                cond_name,
                scorer,
                _fmt((rec.get("fpr_summary") or {}).get("mean")),
                _fmt(alarm),
                _fmt(rec.get("tv_bound_mean")),
                "yes" if violated else ("n/a" if violated is None else "no"),
            ])
    print(f"\nW4 diagnostics at alpha = {alpha:g} (means over repeats)")
    if not rows:
        print("  no diagnostics records found in the results dict")
        return
    headers = ["condition", "scorer", "realized FPR", f"alarm@m={pilot_m}", "tv_bound", "violated"]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "  "
    print(sep.join(h.ljust(w) for h, w in zip(headers, widths)))
    print(sep.join("-" * w for w in widths))
    for r in rows:
        print(sep.join(cell.ljust(w) for cell, w in zip(r, widths)))


def _render_figures(results: dict, figdir: Path) -> None:
    """Render wave-2 figures if guard.viz_wave2 exists (written in parallel)."""
    try:
        from guard import viz_wave2
    except ImportError:
        print("[run_wave2] guard.viz_wave2 not importable; figures skipped")
        return
    fn = getattr(viz_wave2, "make_wave2_figures", None)
    if not callable(fn):
        print("[run_wave2] guard.viz_wave2.make_wave2_figures missing; figures skipped")
        return
    try:
        import inspect

        kwargs = {}
        params = inspect.signature(fn).parameters
        if "alpha" in params:
            kwargs["alpha"] = SUMMARY_ALPHA
        paths = fn(results, figdir, **kwargs)
        for p in paths or []:
            print(f"[run_wave2] figure: {Path(p).resolve()}")
    except Exception as exc:
        print(f"[run_wave2] figure rendering failed ({exc!r}); results JSON is unaffected")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from guard import wave2
    from guard.experiments import _jsonable  # same serializer as the wave-1 engine

    cfg = _build_cfg(args, wave2.DEFAULT_CFG)
    print("[run_wave2] configuration (DEFAULT_CFG + overrides):")
    print(json.dumps({**wave2.DEFAULT_CFG, **cfg}, indent=2, default=str))

    t0 = time.time()
    results = wave2.run_all(cfg)
    print(f"[run_wave2] wave-2 study finished in {time.time() - t0:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(results), fh, indent=1)
    print(f"[run_wave2] results written to {args.out.resolve()}")

    _print_w4_summary(results, SUMMARY_ALPHA, SUMMARY_PILOT_M)
    _render_figures(results, args.figdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
