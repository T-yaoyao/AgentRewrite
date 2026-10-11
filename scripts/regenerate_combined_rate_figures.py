#!/usr/bin/env python3
"""Generate combined benchmark figures with bar+line subplots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


METHODS = [
    "LR",
    "LLM-R$^2$\n(Claude-4.5)",
    "LLM-R$^2$\n(DS-R1)",
    "R-Bot\n(Claude-4.5)",
    "R-Bot\n(DS-R1)",
    "QUITE",
    "PASR",
]

DATA = {
    "TPC-H": {
        "improved": [21, 16, 15, 19, 26, 25, 27],
        "equivalence": [84.13, 93.65, 92.06, 93.65, 93.65, 98.00, 100.00],
    },
    "DSB": {
        "improved": [31, 23, 46, 30, 37, 47, 60],
        "equivalence": [83.33, 80.13, 82.05, 96.15, 94.87, 96.15, 96.79],
    },
    "Calcite": {
        "improved": [12, 15, 21, 13, 21, 23, 25],
        "equivalence": [81.03, 82.14, 81.03, 81.25, 75.86, 86.20, 96.55],
    },
}

BAR_COLOR = "#f6b26b"
BAR_EDGE = "#c17f42"
LINE_COLOR = "#7f8c8d"
GRID_COLOR = "#d9d9d9"
TEXT_COLOR = "#333333"


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


def left_axis_limit(values: list[int]) -> int:
    vmax = max(values)
    if vmax <= 30:
        return 25 if vmax <= 25 else 30
    if vmax <= 60:
        return 70
    return int(np.ceil((vmax + 5) / 10.0) * 10)


def draw_combined(output_path: Path) -> None:
    configure_times_new_roman()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
            ],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    fig.subplots_adjust(left=0.06, right=0.94, top=0.82, bottom=0.38, wspace=0.42)

    legend_handles = None

    for idx, (ax, (bench, values)) in enumerate(zip(axes, DATA.items())):
        improved = values["improved"]
        equivalence = values["equivalence"]
        x = np.arange(len(METHODS))

        bars = ax.bar(
            x,
            improved,
            width=0.56,
            color=BAR_COLOR,
            edgecolor=BAR_EDGE,
            linewidth=0.8,
            zorder=3,
            label="Improved Queries",
        )
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color=GRID_COLOR)
        if idx == 0:
            ax.set_ylabel("Improved Query Numbers", color=TEXT_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, rotation=45, ha="right")
        ax.set_ylim(0, left_axis_limit(improved))

        ax2 = ax.twinx()
        (line,) = ax2.plot(
            x,
            equivalence,
            color=LINE_COLOR,
            linewidth=1.4,
            marker="o",
            markersize=3.2,
            zorder=4,
            label="Equivalence Ratio",
        )
        ax2.set_ylim(0, 100)
        ax2.set_yticks(np.arange(0, 101, 20))
        ax2.set_yticklabels([f"{int(t)}%" for t in np.arange(0, 101, 20)])
        if idx == len(DATA) - 1:
            ax2.set_ylabel("Equivalence Ratio", color=TEXT_COLOR)

        for rect, val in zip(bars, improved):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.55,
                f"{val}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                color=TEXT_COLOR,
                fontfamily="serif",
            )

        for x_pos, val in zip(x, equivalence):
            ax2.text(
                x_pos,
                val + 1.8,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color=LINE_COLOR,
                fontfamily="serif",
            )

        ax.text(
            0.5,
            -0.46,
            f"({chr(ord('a') + idx)}) {bench} Benchmark",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            fontfamily="serif",
        )

        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax.spines["left"].set_color("#666666")
        ax.spines["bottom"].set_color("#666666")
        ax2.spines["right"].set_color("#666666")

        if legend_handles is None:
            legend_handles = [bars, line]

    fig.legend(
        legend_handles,
        ["Improved Queries", "Equivalence Ratio"],
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.985),
        handlelength=1.3,
        columnspacing=1.2,
        prop={"family": "serif", "size": 8.2},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = Path("/root/AgentRewrite/figures")

    primary_names = [
        "rewrite_equiv_improvement.png",
        "rewrite_equiv_improvement.pdf",
        "improvement_rate.png",
        "improvement_rate.pdf",
        "equivalence_rate.png",
        "equivalence_rate.pdf",
    ]

    for name in primary_names:
        path = out_dir / name
        draw_combined(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
