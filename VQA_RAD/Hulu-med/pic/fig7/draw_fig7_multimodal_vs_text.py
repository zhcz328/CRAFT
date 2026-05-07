import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Real-data-only figure script for Fig.7.
# If a panel source file is missing, the panel is marked as unavailable.
# No synthetic trajectories are generated.

output_dir = Path(r"c:\Users\29818\Desktop\logit_lens\VQA_RAD\Hulu-med\pic\fig7")
output_dir.mkdir(parents=True, exist_ok=True)
repo_root = output_dir.parents[3]

# Edit these paths to your exact runs when needed.
PANEL_SOURCES = {
    "M1+image": repo_root / r"VQA_RAD\Hulu-med\tuned_lens\results\trajectory_before_question_idreg_tmp_ok\trajectories.jsonl",
    "M1-text": None,  # not trained yet
    "M2+image": repo_root / r"VQA_RAD\Hulu-med\tuned_lens\results\trajectory_before_question\trajectories.jsonl",
    "M2-text": None,  # not trained yet
}

# Optional filter; set to None to use all records in the file.
# Example: {"prompt_type": "conflict"}
RECORD_FILTER = {"prompt_type": "conflict"}

MM_COLOR = "#1f77b4"
TXT_COLOR = "#d62728"
CURVE_MM = "#2b6cb0"
CURVE_TXT = "#c53030"


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def keep_record(rec: dict) -> bool:
    if RECORD_FILTER is None:
        return True
    for k, v in RECORD_FILTER.items():
        if rec.get(k) != v:
            return False
    return True


def summarize_trajectory(path: Path):
    records = [r for r in read_jsonl(path) if keep_record(r)]
    if not records:
        return None

    deltas = [r.get("tuned_delta_by_layer") for r in records if isinstance(r.get("tuned_delta_by_layer"), list)]
    if not deltas:
        return None

    min_len = min(len(x) for x in deltas)
    deltas = [x[:min_len] for x in deltas]
    mean_curve = np.mean(np.asarray(deltas, dtype=float), axis=0)

    flip_layers = []
    for r in records:
        tuned_stats = r.get("tuned_stats") or {}
        fl = tuned_stats.get("flip_layer")
        if isinstance(fl, int) and fl >= 0:
            flip_layers.append(fl)

    lstar = int(np.median(flip_layers)) if flip_layers else None
    return {
        "curve": mean_curve,
        "layers": np.arange(min_len),
        "lstar": lstar,
        "n": len(records),
        "n_flip": len(flip_layers),
    }


stats = {}
for key, p in PANEL_SOURCES.items():
    if p is None or not p.exists():
        stats[key] = None
    else:
        stats[key] = summarize_trajectory(p)

# Shared y-axis from available panels only
curves = [s["curve"] for s in stats.values() if s is not None]
if not curves:
    raise RuntimeError("No valid trajectories found. Please set PANEL_SOURCES to real trajectories.jsonl files.")

all_y = np.concatenate(curves)
ymin = float(np.min(all_y)) - 0.1
ymax = float(np.max(all_y)) + 0.1

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
plt.subplots_adjust(wspace=0.14, hspace=0.20)

panels = [
    ("(a) M1 + Image", "M1+image", axes[0, 0]),
    ("(b) M1 Text-Only", "M1-text", axes[0, 1]),
    ("(c) M2 + Image", "M2+image", axes[1, 0]),
    ("(d) M2 Text-Only", "M2-text", axes[1, 1]),
]

for title, key, ax in panels:
    panel = stats[key]
    is_mm = "+image" in key
    curve_color = CURVE_MM if is_mm else CURVE_TXT
    line_color = MM_COLOR if is_mm else TXT_COLOR

    ax.set_title(title, fontsize=12, loc="left", pad=8)
    ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.25, linestyle=":", linewidth=0.8)

    if panel is None:
        ax.text(
            0.5,
            0.5,
            "Not available\n(not trained)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#666666",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="#cccccc", pad=4),
        )
        continue

    x = panel["layers"]
    y = panel["curve"]
    ax.plot(x, y, color=curve_color, linewidth=2.4)
    ax.set_xlim(0, int(x[-1]))

    lstar = panel["lstar"]
    if lstar is not None and lstar <= int(x[-1]):
        ax.axvline(lstar, color=line_color, linestyle="--", linewidth=1.8)
        ax.text(
            lstar + 0.35,
            ymin + 0.12,
            rf"$l^*={lstar}$",
            color=line_color,
            fontsize=10,
            rotation=90,
            va="bottom",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.8),
        )

    ax.text(
        0.02,
        0.98,
        f"n={panel['n']}\nflip-n={panel['n_flip']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#444444",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.2),
    )

# VAS shading for M1 and M2 only when both image/text l* exist
for mm_key, txt_key, ax, ytext in [
    ("M1+image", "M1-text", axes[0, 0], ymax - 0.12),
    ("M2+image", "M2-text", axes[1, 0], ymax - 0.12),
]:
    mm = stats.get(mm_key)
    txt = stats.get(txt_key)
    if not mm or not txt:
        continue
    l_mm = mm.get("lstar")
    l_txt = txt.get("lstar")
    if l_mm is None or l_txt is None:
        continue

    lo, hi = sorted([l_mm, l_txt])
    ax.axvspan(lo, hi, color="#9ecae1", alpha=0.25)
    ax.text(
        (lo + hi) / 2,
        ytext,
        f"VAS = {abs(l_mm - l_txt)} layer{'s' if abs(l_mm - l_txt) != 1 else ''}",
        color="#2c5282",
        fontsize=10,
        ha="center",
        va="top",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
    )

for ax in axes[1, :]:
    ax.set_xlabel("Layer l", fontsize=11)
for ax in axes[:, 0]:
    ax.set_ylabel(r"Trajectory score $\bar{\delta}_l$", fontsize=11)

fig.suptitle(
    "Fig.7  Multimodal vs Text-Only Tuned Lens Trajectory Comparison (Real Data)",
    fontsize=14,
    y=0.98,
)

png_path = output_dir / "fig7_multimodal_vs_text_fullwidth.png"
pdf_path = output_dir / "fig7_multimodal_vs_text_fullwidth.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
for k, v in stats.items():
    if v is None:
        print(f"{k}: unavailable")
    else:
        print(f"{k}: n={v['n']}, n_flip={v['n_flip']}, l*={v['lstar']}")
