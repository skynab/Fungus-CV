"""One self-contained HTML page per experiment, to send to a colleague or attach to a notebook.

It fits every plot (as `fungus report` does) and puts on one page: the figures, the fitted
models with their intervals and evidence weights, the quantities read off the best curve,
what was left out and why (flags, jumps, frames excluded by hand), the analysis settings and
which software produced it. Images are embedded, so the file opens anywhere, offline.
Spread maps and validation plots already made for the experiment are included too.
"""

from __future__ import annotations

import base64
import html
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from fungus_cv.analyze.fit import DEFAULT_MODELS, best_fit, format_derived
from fungus_cv.analyze.pipeline import MEASUREMENTS_NAME, RUN_INFO_NAME
from fungus_cv.analyze.report import (
    DEFAULT_EXCLUDE,
    ReportResult,
    list_plots,
    load_measurements,
    make_report,
)
from fungus_cv.provenance import software_provenance
from fungus_cv.storage import Experiment

STYLE = """
:root { --bg:#fbfaf8; --card:#ffffff; --ink:#1f1e1c; --ink2:#52514e; --line:#e4e3df;
        --accent:#2a78d6; --warn:#b3541e; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#1b1a19; --card:#242321; --ink:#ecebe8; --ink2:#b3b1ac; --line:#3a3936;
          --accent:#6aa6ee; --warn:#e9925f; }
  img.figure { background:#fff; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width:1040px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:26px; margin:0 0 4px; } h2 { font-size:20px; margin:32px 0 8px; }
h3 { font-size:16px; margin:20px 0 6px; }
.sub { color:var(--ink2); margin:0 0 16px; }
section.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
               padding:16px 18px; margin:16px 0; }
.facts { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:10px; }
.fact b { display:block; font-size:18px; } .fact span { color:var(--ink2); font-size:13px; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line);
         vertical-align:top; }
th { color:var(--ink2); font-weight:600; }
td.num { font-variant-numeric:tabular-nums; white-space:nowrap; }
tr.best td { font-weight:600; }
img.figure { max-width:100%; height:auto; border-radius:6px; border:1px solid var(--line);
             margin:8px 0; }
.warn { color:var(--warn); }
code, pre { font-family:ui-monospace, Menlo, Consolas, monospace; font-size:13px; }
pre { background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:10px;
      overflow-x:auto; max-height:420px; }
details summary { cursor:pointer; color:var(--accent); }
"""


@dataclass
class SummaryResult:
    path: Path
    reports: list[ReportResult] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _e(value) -> str:
    return html.escape(str(value))


def _num(value: float, digits: int = 4) -> str:
    return "–" if value is None or not math.isfinite(value) else f"{value:.{digits}g}"


def _image(path: Path, alt: str) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img class="figure" alt="{_e(alt)}" src="data:image/png;base64,{data}">'


def _interval(ci) -> str:
    return f"[{_num(ci[0])}, {_num(ci[1])}]" if ci else ""


def _fits_table(report: ReportResult) -> str:
    best = best_fit(report.fits)
    rows = []
    for fit in sorted(report.fits, key=lambda f: -(f.akaike_weight
                                                   if math.isfinite(f.akaike_weight) else -1)):
        if not fit.ok:
            rows.append(f"<tr><td>{_e(fit.model)}</td><td colspan=5 class=warn>did not fit: "
                        f"{_e(fit.message)}</td></tr>")
            continue
        params = "<br>".join(
            f"{_e(k)} = {_num(v)} ± {_num(fit.stderr.get(k, math.nan), 2)} "
            f"{_interval(fit.ci95.get(k))}" for k, v in fit.params.items())
        checks = [f"R² {_num(fit.r2, 4)}"]
        if math.isfinite(fit.reduced_chi2):
            checks.append(f"χ²/ν {_num(fit.reduced_chi2, 3)}")
        if fit.error_model == "ar1":
            checks.append(f"AR(1) φ {_num(fit.ar1_phi, 2)}")
        if fit.warnings:
            checks.append('<span class="warn">' + _e("; ".join(fit.warnings)) + "</span>")
        cls = ' class="best"' if fit is best else ""
        rows.append(
            f"<tr{cls}><td>{_e(fit.model)}<br><code>{_e(fit.formula)}</code></td>"
            f"<td class=num>{_num(fit.akaike_weight, 3)}</td>"
            f"<td class=num>{_num(fit.aicc, 5)}</td><td class=num>{params}</td>"
            f"<td>{_e(format_derived(fit))}</td><td>{'<br>'.join(checks)}</td></tr>")
    return ("<div class=scroll><table><tr><th>Model</th><th>Akaike weight</th><th>AICc</th>"
            "<th>Parameters ± SE [95% CI]</th><th>Read off the curve [95% CI]</th>"
            "<th>Checks</th></tr>" + "".join(rows) + "</table></div>")


def _left_out(report_dir: Path) -> str:
    info = json.loads((report_dir / "fits.json").read_text(encoding="utf-8"))
    unit = f"days (daily {info['daily']})" if info.get("daily") else "frames"
    parts = [f"{info['n_used']} {unit} fitted, {info['n_excluded']} left out"]
    if info.get("hours"):
        start, end = info["hours"]
        parts.append(f"only frames taken {start:g}–{end:g} h ({info.get('timezone')} time)")
    if info.get("excluded_flags"):
        parts.append("frames flagged " + ", ".join(info["excluded_flags"]) + " are not fitted")
    if info.get("jumps"):
        parts.append(f"{info['jumps']} jump(s) "
                     + ("left out" if info.get("jumps_excluded") else "kept, marked in the plot"))
    if info.get("retreats"):
        parts.append(f"{info['retreats']} frame(s) where the value went back")
    text = "; ".join(parts) + "."
    hand = info.get("excluded_by_hand") or []
    if hand:
        items = "".join(f"<li><code>{_e(Path(h['frame_file']).name)}</code>: {_e(h['reason'])}"
                        "</li>" for h in hand)
        text += f"<br>Excluded by hand:<ul>{items}</ul>"
    return f"<p>{text}</p>"


def _flag_counts(rows: list[dict]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        counts.update(filter(None, row.get("flags", "").split(";")))
    return counts


def make_summary(experiment: Experiment, metric: str | None = None,
                 results_dir: Path | None = None, out: Path | None = None,
                 models: tuple[str, ...] = DEFAULT_MODELS, bootstrap: int = 500,
                 exclude_flags: tuple[str, ...] = DEFAULT_EXCLUDE, exclude_jumps: bool = False,
                 time_to: tuple[float, ...] = (), seed: int = 0,
                 hours: tuple[float, float] | None = None,
                 daily: str | None = None) -> SummaryResult:
    results_dir = Path(results_dir) if results_dir else experiment.root / "results"
    if not (results_dir / MEASUREMENTS_NAME).exists():
        raise FileNotFoundError(f"{results_dir / MEASUREMENTS_NAME} not found; "
                                "run `fungus analyze` first")
    out = Path(out) if out else results_dir / "summary.html"
    result = SummaryResult(out)
    cfg = experiment.config
    rows = load_measurements(experiment, results_dir=results_dir)
    run_info_path = results_dir / RUN_INFO_NAME
    run_info = json.loads(run_info_path.read_text(encoding="utf-8")) \
        if run_info_path.exists() else {}
    plots = list_plots(experiment, results_dir)

    body = [f"<h1>{_e(cfg.name)}</h1>"]
    if cfg.description:
        body.append(f'<p class="sub">{_e(cfg.description)}</p>')
    frames = sorted({r["frame_file"] for r in rows})
    times = sorted(r["timestamp_utc"] for r in rows)
    scale = run_info.get("scale") or {}
    facts = [("Frames measured", len(frames)), ("First", times[0] if times else "–"),
             ("Last", times[-1] if times else "–"), ("Plots", ", ".join(plots)),
             ("Settings", run_info.get("settings_hash", "–")),
             ("Scale", f"{_num(scale['mm_per_px'])} mm/px ± {_num(scale.get('se_mm_per_px'), 2)}"
              if scale.get("mm_per_px") else "none (pixels)")]
    body.append('<section class="card"><div class="facts">' + "".join(
        f'<div class="fact"><span>{_e(k)}</span><b>{_e(v)}</b></div>' for k, v in facts)
        + "</div></section>")

    flags = _flag_counts(rows)
    if flags:
        items = ", ".join(f"{_e(k)} {v}" for k, v in flags.most_common())
        body.append(f"<p>Frame flags (rows, over all plots): {items}.</p>")

    for plot in plots:
        try:
            report = make_report(experiment, metric=metric, plot=plot, models=models,
                                 bootstrap=bootstrap, exclude_flags=exclude_flags,
                                 exclude_jumps=exclude_jumps, results_dir=results_dir,
                                 time_to=time_to, seed=seed, formats=("png",),
                                 hours=hours, daily=daily)
        except (ValueError, RuntimeError) as exc:
            result.problems.append(f"{plot}: {exc}")
            body.append(f"<h2>{_e(plot)}</h2><p class=warn>Not fitted: {_e(exc)}</p>")
            continue
        result.reports.append(report)
        report_dir = Path(report.files[0]).parent
        body.append(f'<section class="card"><h2>{_e(plot)}: {_e(report.metric)}</h2>')
        body.append(f'<p class="sub">Time in {_e(report.time_unit)} since '
                    f"{_e(report.t0_utc)}.</p>")
        for png in (p for p in map(Path, report.files) if p.suffix == ".png"):
            body.append(_image(png, png.stem))
        body.append(_fits_table(report))
        body.append(_left_out(report_dir))
        body.append("</section>")

    extras = sorted((results_dir / "spread").rglob("spread_map.png")) \
        if (results_dir / "spread").exists() else []
    extras += sorted((results_dir / "validation").glob("*_agreement.png")) \
        if (results_dir / "validation").exists() else []
    if extras:
        body.append('<section class="card"><h2>Spread and validation</h2>')
        for png in extras:
            body.append(f"<h3>{_e(png.relative_to(results_dir).as_posix())}</h3>"
                        + _image(png, png.stem))
        body.append("</section>")

    software = software_provenance()
    made = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body.append('<section class="card"><h2>How this was made</h2>')
    body.append(f"<p>Made {made} by fungus-cv {_e(software.get('fungus_cv_version'))}"
                + (f", commit <code>{_e(software['git_commit'][:12])}</code>"
                   + (" <span class=warn>(with uncommitted changes)</span>"
                      if software.get("git_dirty") else "")
                   if software.get("git_commit") else "")
                + f". Fits: {bootstrap} bootstrap refits per model, seed {seed}; intervals are "
                "95% bootstrap percentiles; Akaike weights compare the models fitted "
                "together.</p>")
    if run_info.get("settings"):
        body.append("<details><summary>Analysis settings</summary><pre>"
                    + _e(json.dumps(run_info["settings"], indent=2)) + "</pre></details>")
    body.append("</section>")

    page = ("<!doctype html><html lang=en><head><meta charset=utf-8>"
            '<meta name=viewport content="width=device-width, initial-scale=1">'
            f"<title>{_e(cfg.name)} summary</title><style>{STYLE}</style></head>"
            f"<body><main>{''.join(body)}</main></body></html>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return result
