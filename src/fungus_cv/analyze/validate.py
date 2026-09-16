"""Agreement between automatic and hand measurements (Bland-Altman analysis)."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

from fungus_cv.analyze.pipeline import RESULTS_DIR
from fungus_cv.analyze.report import GRID, INK, INK_2, SERIES, _style, load_measurements
from fungus_cv.learn.export import pick_evenly
from fungus_cv.storage import Experiment


def write_template(experiment: Experiment, path: Path, count: int = 20) -> int:
    """CSV of evenly spaced analyzed frames with an empty ``value`` column to fill in."""
    rows = load_measurements(experiment)
    chosen = [rows[i] for i in pick_evenly(len(rows), count)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp_utc", "value", "observer", "notes"])
        for r in chosen:
            writer.writerow([Path(r["frame_file"]).name, r["timestamp_utc"], "", "", ""])
    return len(chosen)


@dataclass
class Agreement:
    metric: str
    n: int
    bias: float  # mean(auto - hand)
    bias_ci95: tuple[float, float]
    sd_diff: float
    limits_of_agreement: tuple[float, float]  # bias +/- 1.96 sd
    mae: float
    rmse: float
    pearson_r: float
    slope: float  # auto = slope * hand + intercept
    intercept: float
    worst_frame: str
    worst_diff: float
    unmatched: list[str] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)


def _match(key: str, rows: list[dict]) -> dict | None:
    key = key.strip()
    for r in rows:
        name = Path(r["frame_file"]).name
        if key in (name, Path(name).stem, r["frame_file"]):
            return r
    hits = [r for r in rows if key and r["timestamp_utc"].startswith(key)]
    return hits[0] if len(hits) == 1 else None


def validate(experiment: Experiment, hand_csv: Path, metric: str = "extent_mm") -> Agreement:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_measurements(experiment)
    if rows and metric not in rows[0]:
        raise ValueError(f"unknown metric {metric!r}")
    with open(hand_csv, newline="", encoding="utf-8") as f:
        hand = [h for h in csv.DictReader(f) if (h.get("value") or "").strip()]
    if not hand:
        raise ValueError(f"{hand_csv} has no filled-in 'value' entries")

    pairs, unmatched = [], []
    for h in hand:
        key = h.get("frame") or h.get("timestamp_utc") or ""
        r = _match(key, rows)
        if r is None or r[metric] in ("", None):
            unmatched.append(key)
            continue
        pairs.append((Path(r["frame_file"]).name, float(h["value"]), float(r[metric]),
                      h.get("observer", "")))
    if len(pairs) < 3:
        raise ValueError(f"only {len(pairs)} hand measurement(s) matched analyzed frames; "
                         "need at least 3")

    names = [p[0] for p in pairs]
    hand_v = np.array([p[1] for p in pairs])
    auto_v = np.array([p[2] for p in pairs])
    diff = auto_v - hand_v
    n = len(diff)
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    half_ci = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    slope, intercept = np.polyfit(hand_v, auto_v, 1)
    r = float(np.corrcoef(hand_v, auto_v)[0, 1]) if hand_v.std() > 0 and auto_v.std() > 0 \
        else math.nan
    worst = int(np.argmax(np.abs(diff)))

    out_dir = experiment.root / RESULTS_DIR / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = Agreement(
        metric=metric, n=n, bias=bias, bias_ci95=(bias - half_ci, bias + half_ci),
        sd_diff=sd, limits_of_agreement=(bias - 1.96 * sd, bias + 1.96 * sd),
        mae=float(np.abs(diff).mean()), rmse=float(np.sqrt((diff ** 2).mean())),
        pearson_r=r, slope=float(slope), intercept=float(intercept),
        worst_frame=names[worst], worst_diff=float(diff[worst]), unmatched=unmatched,
    )

    paired = out_dir / f"{metric}_pairs.csv"
    with open(paired, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "hand", "automatic", "difference", "observer"])
        for (name, hv, av, obs), d in zip(pairs, diff):
            writer.writerow([name, hv, av, round(float(d), 6), obs])
    result.files.append(paired)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)
    for ax in (ax1, ax2):
        _style(ax)
    lo = float(min(hand_v.min(), auto_v.min()))
    hi = float(max(hand_v.max(), auto_v.max()))
    ax1.plot([lo, hi], [lo, hi], color=GRID, lw=2, label="perfect agreement", zorder=1)
    ax1.plot(hand_v, auto_v, "o", ms=6, color=SERIES[0], label="frames", zorder=3)
    ax1.set_xlabel(f"Hand measurement ({metric})", color=INK)
    ax1.set_ylabel(f"Automatic ({metric})", color=INK)
    ax1.set_title(f"Agreement (r = {r:.4f})", color=INK, loc="left", fontsize=11)
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK)

    mean_v = (hand_v + auto_v) / 2
    ax2.axhline(bias, color=SERIES[0], lw=2, label=f"bias {bias:+.3f}")
    for v in result.limits_of_agreement:
        ax2.axhline(v, color=SERIES[1], lw=2, ls=(0, (6, 3)))
    ax2.plot([], [], color=SERIES[1], lw=2, ls=(0, (6, 3)),
             label=f"95% limits {result.limits_of_agreement[0]:+.3f} / "
                   f"{result.limits_of_agreement[1]:+.3f}")
    ax2.plot(mean_v, diff, "o", ms=6, color=INK_2, zorder=3)
    ax2.set_xlabel("Mean of hand and automatic", color=INK)
    ax2.set_ylabel("Automatic − hand", color=INK)
    ax2.set_title(f"Bland–Altman (n = {n})", color=INK, loc="left", fontsize=11)
    ax2.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    png = out_dir / f"{metric}_agreement.png"
    fig.savefig(png)
    plt.close(fig)
    result.files.append(png)
    return result
