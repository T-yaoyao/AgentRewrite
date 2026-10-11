#!/usr/bin/env python3
"""Generate ablation study figure for structured rule knowledge (Table 6)."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


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


def main():
    configure_times_new_roman()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
            ],
            "mathtext.fontset": "stix",
            "font.size": 15,
            "axes.labelsize": 17,
            "axes.titlesize": 17,
            "xtick.labelsize": 16,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
        }
    )

    methods = [
        "PASR",
        "without\nRule Group",
        "without\nRule Example",
        "without\nRule Knowledge",
    ]

    # Data from Table 6: Method(dsb) & Mean & Median & 75th & 95th
    data = {
        "Mean": [3.12, 6.14, 11.62, 21.15],
        "Median": [0.95, 0.94, 1.08, 1.74],
        "75th": [2.42, 7.81, 11.26, 19.79],
        "95th": [11.12, 26.58, 42.43, 84.01],
    }

    # Style definitions matching the user's reference image with distinct hatch patterns
    styles = {
        "Mean": {
            "color": "#7ebce6",  # Light blue
            "edgecolor": "#111111",
            "hatch": "///",
        },
        "Median": {
            "color": "#185e78",  # Dark teal
            "edgecolor": "#111111",
            "hatch": "\\\\\\",
        },
        "75th": {
            "color": "#944005",  # Dark reddish brown
            "edgecolor": "#111111",
            "hatch": "xxx",
        },
        "95th": {
            "color": "#f79e38",  # Orange
            "edgecolor": "#111111",
            "hatch": "...",
        },
    }

    x = np.arange(len(methods))
    total_group_width = 0.68
    n_bars = len(data)
    bar_width = total_group_width / n_bars
    offsets = np.arange(n_bars) * bar_width - (total_group_width / 2.0) + (bar_width / 2.0)

    fig, ax = plt.subplots(figsize=(11.2, 5.3))
    ax.set_yscale("log")

    bars_list = []
    for i, (metric, values) in enumerate(data.items()):
        pos = x + offsets[i]
        st = styles[metric]
        bars = ax.bar(
            pos,
            values,
            width=bar_width,
            label=metric,
            color=st["color"],
            edgecolor=st["edgecolor"],
            hatch=st["hatch"],
            linewidth=1.0,
            zorder=3,
        )
        bars_list.append(bars)

        # Add data values on top of bars
        for rect, val in zip(bars, values):
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h * 1.14,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=12.5,
                color="#222222",
                fontfamily="serif",
            )

    ax.set_ylabel("Execution Time (s)", color="#111111", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.5, 150)
    ticks = [0.5, 1, 2, 5, 10, 20, 50, 100]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])

    # Grid and full box border like reference image
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="#d9d9d9", alpha=0.8, which="both")
    for spine in ax.spines.values():
        spine.set_color("#111111")
        spine.set_linewidth(1.0)

    # Legend centered at top in a clean box
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        frameon=True,
        edgecolor="#222222",
        fancybox=False,
        columnspacing=2.2,
        handlelength=2.2,
        handleheight=1.2,
    )

    fig.subplots_adjust(top=0.81, bottom=0.22, left=0.11, right=0.985)

    out_dir = Path("/root/AgentRewrite/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    output_names = [
        "ablation_rule.png",
        "ablation_rule.pdf",
        "ablation_6.png",
        "ablation_6.pdf",
    ]
    for name in output_names:
        path = out_dir / name
        save_kwargs = {"dpi": 300, "bbox_inches": "tight"} if path.suffix == ".png" else {"bbox_inches": "tight"}
        fig.savefig(path, **save_kwargs)
    plt.close(fig)
    print("Saved ablation_rule/ablation_6 figures")


if __name__ == "__main__":
    main()
