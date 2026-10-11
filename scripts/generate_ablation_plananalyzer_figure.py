#!/usr/bin/env python3
"""Generate PlanAnalyzer ablation figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


BENCHMARKS = {
    "TPC-H": {
        "variants": ["PASR", "w/o\nPlanAnalyzer"],
        "imp_rate": [42.85, 37.85],
        "mean": [10.69, 14.78],
        "median": [7.48, 8.94],
        "75th": [11.74, 13.74],
        "95th": [26.34, 45.83],
    },
    "DSB": {
        "variants": ["PASR", "w/o\nPlanAnalyzer"],
        "imp_rate": [38.46, 35.26],
        "mean": [3.12, 7.56],
        "median": [0.95, 1.37],
        "75th": [2.42, 5.48],
        "95th": [11.12, 31.64],
    },
    "Calcite": {
        "variants": ["PASR", "w/o\nPlanAnalyzer"],
        "imp_rate": [43.10, 30.17],
        "mean": [8.37, 15.39],
        "median": [1.60, 2.41],
        "75th": [8.65, 10.97],
        "95th": [20.90, 63.82],
    },
}

BAR_STYLES = {
    "mean": {"label": "Mean", "color": "#7ebce6", "hatch": "///"},
    "median": {"label": "Median", "color": "#185e78", "hatch": "\\\\\\"},
    "75th": {"label": "75th", "color": "#944005", "hatch": "xxx"},
    "95th": {"label": "95th", "color": "#f79e38", "hatch": "..."},
}

LINE_STYLE = {
    "color": "#4e49b7",
    "marker": "o",
    "linewidth": 1.8,
    "markersize": 5.2,
}


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
                "Times New Roman PS MT",
                "Nimbus Roman",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.1))
    fig.subplots_adjust(left=0.06, right=0.94, top=0.82, bottom=0.28, wspace=0.42)

    x = np.arange(2)
    width = 0.16
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    legend_handles = []
    legend_labels = []

    for idx, (ax, (benchmark, values)) in enumerate(zip(axes, BENCHMARKS.items())):
        bars_for_legend = []
        for bar_idx, (metric, style) in enumerate(BAR_STYLES.items()):
            bar_values = values[metric]
            bars = ax.bar(
                x + offsets[bar_idx],
                bar_values,
                width=width,
                color=style["color"],
                edgecolor="#111111",
                hatch=style["hatch"],
                linewidth=0.9,
                zorder=3,
                label=style["label"],
            )
            bars_for_legend.append(bars)

            for rect, val in zip(bars, bar_values):
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + max(0.18, rect.get_height() * 0.04),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    color="#222222",
                )

        ax2 = ax.twinx()
        (line,) = ax2.plot(
            x,
            values["imp_rate"],
            **LINE_STYLE,
            zorder=4,
            label="Imp. Rate",
        )

        for xpos, val in zip(x, values["imp_rate"]):
            ax2.text(
                xpos,
                val + 1.1,
                f"{val:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8.2,
                color=LINE_STYLE["color"],
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(values["variants"])
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="#d9d9d9")

        if idx == 0:
            ax.set_ylabel("Execution Time (s)")
            ax2.tick_params(axis="y", labelright=False)
            ax2.spines["right"].set_visible(False)
        elif idx == len(BENCHMARKS) - 1:
            ax2.set_ylabel("Imp. Rate (%)")
        else:
            ax2.tick_params(axis="y", labelright=False)
            ax2.spines["right"].set_visible(False)

        ax.set_ylim(0, max(values["95th"]) * 1.22)
        ax2.set_ylim(0, 50)
        ax2.set_yticks(np.arange(0, 51, 10))

        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax.spines["left"].set_color("#666666")
        ax.spines["bottom"].set_color("#666666")
        ax2.spines["right"].set_color("#666666")

        ax.text(
            0.5,
            -0.27,
            f"({chr(ord('a') + idx)}) {benchmark}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

        if not legend_handles:
            legend_handles = [bars_for_legend[0], bars_for_legend[1], bars_for_legend[2], bars_for_legend[3], line]
            legend_labels = ["Mean", "Median", "75th", "95th", "Imp. Rate"]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.99),
        handlelength=1.7,
        columnspacing=1.1,
    )

    out_dir = Path("/root/AgentRewrite/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "ablation_plananalyzer.png"
    pdf_path = out_dir / "ablation_plananalyzer.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
