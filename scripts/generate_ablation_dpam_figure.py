#!/usr/bin/env python3
"""Generate DPAM ablation figure for TPC-H."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


METHODS = ["PASR", "Parametric Only", "Retrieval Only", "w/o DPAM"]
DATA = {
    "Mean (s)": [10.69, 15.67, 14.75, 18.69],
    "Imp. Rate (%)": [42.85, 30.15, 31.74, 28.57],
    "Avg. Time (s)": [236.58, 251.40, 247.75, 273.10],
    "Cost (RMB)": [12.95, 13.86, 13.76, 14.98],
}

COLORS = ["#f6b26b", "#a7c7e7", "#8cb8df", "#6fa8dc"]
EDGES = ["#b0722d", "#5f84a3", "#4f79a0", "#3d6f97"]
HATCHES = ["", "///", "\\\\\\", "xx"]


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
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.8))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.22, wspace=0.22, hspace=0.36)

    x = np.arange(len(METHODS))
    metric_items = list(DATA.items())
    legend_handles = []

    for idx, (ax, (metric, values)) in enumerate(zip(axes.flat, metric_items)):
        bars = []
        for i, value in enumerate(values):
            bar = ax.bar(
                i,
                value,
                width=0.62,
                color=COLORS[i],
                edgecolor=EDGES[i],
                hatch=HATCHES[i],
                linewidth=0.9,
                zorder=3,
            )
            bars.append(bar)

            if metric in {"Mean (s)", "Imp. Rate (%)", "Cost (RMB)"}:
                label = f"{value:.2f}"
            else:
                label = f"{value:.1f}"
            offset = max(value * 0.025, 0.12)
            ax.text(
                i,
                value + offset,
                label,
                ha="center",
                va="bottom",
                fontsize=8.2,
                color="#333333",
            )

        ax.set_title(metric, pad=7)
        ax.set_xticks(x)
        ax.set_xticklabels(["PASR", "Parametric", "Retrieval", "w/o DPAM"])
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="#d9d9d9")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#666666")
        ax.spines["bottom"].set_color("#666666")

        if metric == "Mean (s)":
            ax.set_ylim(0, 22)
        elif metric == "Imp. Rate (%)":
            ax.set_ylim(0, 50)
        elif metric == "Avg. Time (s)":
            ax.set_ylim(220, 285)
        elif metric == "Cost (RMB)":
            ax.set_ylim(12, 15.5)

        ax.text(
            0.02,
            1.02,
            f"({chr(ord('a') + idx)})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

        if not legend_handles:
            legend_handles = [bars[i] for i in range(len(METHODS))]

    fig.legend(
        legend_handles,
        METHODS,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=True,
        handlelength=1.4,
        columnspacing=1.2,
    )

    out_dir = Path("/root/AgentRewrite/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "ablation_dpam.png"
    pdf_path = out_dir / "ablation_dpam.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
