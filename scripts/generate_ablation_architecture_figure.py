#!/usr/bin/env python3
"""Generate multi-agent architecture ablation figure for Calcite."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


METHODS = ["PASR", "Single-Agent"]
COLORS = {"PASR": "#f39c34", "Single-Agent": "#6baed6"}
DATA = [
    {
        "label": "Mean (s)",
        "subtitle": "lower is better",
        "values": {"PASR": 8.37, "Single-Agent": 23.28},
        "xlim": (0, 26),
        "xticks": np.arange(0, 26, 5),
    },
    {
        "label": "Imp. Rate (%)",
        "subtitle": "higher is better",
        "values": {"PASR": 43.10, "Single-Agent": 29.31},
        "xlim": (20, 48),
        "xticks": np.arange(20, 49, 5),
    },
    {
        "label": "Equ. Rate (%)",
        "subtitle": "higher is better",
        "values": {"PASR": 96.55, "Single-Agent": 84.48},
        "xlim": (80, 100),
        "xticks": np.arange(80, 101, 5),
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
                "Times New Roman PS MT",
                "Nimbus Roman",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.24, wspace=0.34)

    x = np.arange(len(METHODS))

    for idx, (ax, metric) in enumerate(zip(axes, DATA)):
        values = [metric["values"][method] for method in METHODS]
        bars = ax.bar(
            x,
            values,
            width=0.56,
            color=[COLORS[method] for method in METHODS],
            edgecolor="#333333",
            linewidth=0.8,
            zorder=3,
        )

        for bar, value in zip(bars, values):
            offset = max((metric["xlim"][1] - metric["xlim"][0]) * 0.02, value * 0.03)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
                color="#333333",
            )

        ax.set_title(metric["label"], pad=8)
        ax.text(
            0.5,
            1.02,
            metric["subtitle"],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#555555",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS)
        ax.set_ylim(*metric["xlim"])
        ax.set_yticks(metric["xticks"])
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, color="#d9d9d9")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#777777")
        ax.spines["bottom"].set_color("#777777")

        if idx == 0:
            ax.set_ylabel("Value")

        ax.text(
            0.02,
            1.04,
            f"({chr(ord('a') + idx)})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    out_dir = Path("/root/AgentRewrite/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "ablation_architecture.png"
    pdf_path = out_dir / "ablation_architecture.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
