#!/usr/bin/env python3
"""Generate a time-cost tradeoff figure for TPC-H methods."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


METHODS = [
    {
        "name": "LLMR2\n(Claude-4.5)",
        "time": 384.50,
        "cost": 6.73,
        "color": "#4c78a8",
        "marker": "o",
    },
    {
        "name": "R-Bot\n(Claude-4.5)",
        "time": 239.24,
        "cost": 145.61,
        "color": "#e45756",
        "marker": "s",
    },
    {
        "name": "QUITE\n(DS-R1+Claude-4.5)",
        "time": 283.41,
        "cost": 48.87,
        "color": "#72b7b2",
        "marker": "^",
    },
    {
        "name": "PASR\n(DS-R1+Claude-4.5)",
        "time": 236.58,
        "cost": 12.95,
        "color": "#f2a541",
        "marker": "D",
    },
]


def configure_times_new_roman() -> None:
    font_paths = [
        Path("/mnt/c/Windows/Fonts/times.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
        Path("/mnt/c/Windows/Fonts/timesi.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbi.ttf"),
    ]
    for path in font_paths:
        if path.exists():
            font_manager.fontManager.addfont(str(path))


def main() -> None:
    configure_times_new_roman()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
            ],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
        }
    )

    fig, ax = plt.subplots(figsize=(7.8, 5.0))

    times = [item["time"] for item in METHODS]
    costs = [item["cost"] for item in METHODS]
    min_time, max_time = min(times), max(times)
    min_cost, max_cost = min(costs), max(costs)

    ax.set_xlim(min_time - 18, max_time + 20)
    ax.set_ylim(0, 165)

    # Shade the preferred direction to emphasize the left-bottom objective.
    ax.axvspan(min_time - 18, 255, color="#f5f5f5", zorder=0)
    ax.axhspan(0, 25, color="#faf6e8", zorder=0)
    for item in METHODS:
        ax.scatter(
            item["time"],
            item["cost"],
            s=130,
            marker=item["marker"],
            color=item["color"],
            edgecolor="#222222",
            linewidth=0.9,
            zorder=4,
            label=item["name"].replace("\n", " "),
        )
        ax.plot(
            [item["time"], item["time"]],
            [0, item["cost"]],
            linestyle="--",
            linewidth=0.7,
            color=item["color"],
            alpha=0.4,
            zorder=1,
        )
        ax.plot(
            [min_time - 18, item["time"]],
            [item["cost"], item["cost"]],
            linestyle="--",
            linewidth=0.7,
            color=item["color"],
            alpha=0.4,
            zorder=1,
        )

    annotations = {
        "LLMR2\n(Claude-4.5)": (-10, 8, "right", "bottom"),
        "R-Bot\n(Claude-4.5)": (10, -6, "left", "top"),
        "QUITE\n(DS-R1+Claude-4.5)": (12, 6, "left", "bottom"),
        "PASR\n(DS-R1+Claude-4.5)": (10, 8, "left", "bottom"),
    }
    for item in METHODS:
        dx, dy, ha, va = annotations[item["name"]]
        ax.annotate(
            f"{item['name']}\n({item['time']:.2f}s, {item['cost']:.2f} RMB)",
            xy=(item["time"], item["cost"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=9.2,
            color="#222222",
        )

    ax.set_xlabel("Average Time (s)")
    ax.set_ylabel("Cost (RMB)")
    ax.set_title("TPC-H Time-Cost", pad=10)
    ax.set_yticks(np.arange(0, 161, 20))
    ax.grid(True, which="both", axis="both", linestyle="--", linewidth=0.7, color="#d9d9d9")

    for spine in ax.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(1.0)

    out_dir = Path("/root/AgentRewrite/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "cost_efficiency_tradeoff.png"
    pdf_path = out_dir / "cost_efficiency_tradeoff.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
