#!/usr/bin/env python3
"""Regenerate improvement/equivalence rate bar charts with PASR label."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

METHODS = [
    "LearnedRewrite",
    "LLM-R2\n(Claude-4.5)",
    "LLM-R2\n(DS-R1)",
    "R-Bot\n(Claude-4.5)",
    "R-Bot\n(DS-R1)",
    "QUITE",
    "PASR",
]

COLORS = {
    "TPCH": "#acbd86",
    "DSB": "#ffd6a0",
    "Calcite": "#ffa06f",
}

SERIES_ORDER = ["TPCH", "DSB", "Calcite"]

FIGURES: Dict[str, Dict[str, object]] = {
    "improvement_rate": {
        "y_label": "Improvement Rate (%)",
        "y_min": 0.0,
        "y_max": 50.0,
        "tick_step": 10.0,
        "series": {
            "TPCH": [33.30, 25.40, 23.81, 30.16, 41.26, 39.68, 42.85],
            "DSB": [19.87, 14.74, 29.49, 19.23, 23.71, 30.13, 38.46],
            "Calcite": [20.69, 25.86, 36.21, 22.41, 36.21, 39.66, 43.10],
        },
    },
    "equivalence_rate": {
        "y_label": "Equivalence Rate (%)",
        "y_min": 70.0,
        "y_max": 105.0,
        "tick_step": 5.0,
        "series": {
            "TPCH": [84.13, 93.65, 92.06, 93.65, 93.65, 98.00, 100.00],
            "DSB": [83.33, 80.13, 82.05, 96.15, 94.87, 96.15, 96.79],
            "Calcite": [81.03, 82.14, 81.03, 81.25, 75.86, 86.20, 96.55],
        },
    },
}


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align="center")
    return box[2] - box[0], box[3] - box[1]


def format_value(val: float) -> str:
    return f"{val:.2f}".rstrip("0").rstrip(".")


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    fill: Tuple[int, int, int],
    width: int = 1,
    dash: int = 6,
    gap: int = 4,
) -> None:
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    pos = 0.0
    drawing = True
    while pos < length:
        seg = dash if drawing else gap
        end = min(pos + seg, length)
        if drawing:
            sx = x1 + dx * pos
            sy = y1 + dy * pos
            ex = x1 + dx * end
            ey = y1 + dy * end
            draw.line((sx, sy, ex, ey), fill=fill, width=width)
        pos = end
        drawing = not drawing


def draw_chart(config: Dict[str, object], output_path: Path) -> None:
    width, height = 2000, 920
    margin_left, margin_right = 175, 45
    margin_top, margin_bottom = 100, 235
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    axis_color = (85, 85, 85)
    grid_color = (180, 180, 180)
    text_color = (51, 51, 51)

    y_label = str(config["y_label"])
    y_min = float(config["y_min"])
    y_max = float(config["y_max"])
    tick_step = float(config["tick_step"])
    series = config["series"]  # type: ignore[assignment]

    label_font = load_font(38)
    small_font = load_font(32)
    value_font = load_font(16)
    legend_font = load_font(30)
    tick_font = load_font(20)

    label_img = Image.new("RGBA", (480, 110), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((0, 0), y_label, font=label_font, fill=text_color)
    label_img = label_img.rotate(90, expand=True)
    image.paste(
        label_img,
        (20, plot_top + plot_height // 2 - label_img.size[1] // 2),
        label_img,
    )

    ticks: List[float] = []
    current = y_min
    while current <= y_max + 1e-9:
        ticks.append(round(current, 2))
        current += tick_step

    def value_to_y(v: float) -> float:
        return plot_bottom - (v - y_min) / (y_max - y_min) * plot_height

    for t in ticks:
        y = value_to_y(t)
        draw_dashed_line(draw, plot_left, y, plot_right, y, grid_color, width=1)
        label = f"{t:.0f}" if abs(t - round(t)) < 1e-9 else f"{t:.2f}"
        tw, th = text_size(draw, label, tick_font)
        draw.text((plot_left - 18 - tw, y - th / 2), label, font=tick_font, fill=text_color)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=axis_color, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=axis_color, width=2)

    group_count = len(METHODS)
    group_width = plot_width / group_count
    bar_width = group_width * 0.22
    intra_gap = 0.0
    group_inner_width = 3 * bar_width + 2 * intra_gap

    for idx, method in enumerate(METHODS):
        group_center = plot_left + group_width * (idx + 0.5)
        start_x = group_center - group_inner_width / 2
        for s_idx, key in enumerate(SERIES_ORDER):
            val = float(series[key][idx])  # type: ignore[index]
            x0 = start_x + s_idx * (bar_width + intra_gap)
            x1 = x0 + bar_width
            y0 = value_to_y(val)
            draw.rectangle((x0, y0, x1, plot_bottom), fill=hex_to_rgb(COLORS[key]))

            val_label = format_value(val)
            tw, th = text_size(draw, val_label, value_font)
            draw.text(
                (x0 + (bar_width - tw) / 2, y0 - th - 8),
                val_label,
                font=value_font,
                fill=text_color,
            )

        draw.multiline_text(
            (group_center, plot_bottom + 22),
            method,
            font=small_font,
            fill=text_color,
            anchor="ma",
            align="center",
            spacing=4,
        )

    legend_items = SERIES_ORDER
    legend_box_w = 42
    legend_box_h = 28
    legend_text_gap = 14
    legend_item_gap = 50
    legend_text_widths = []
    for key in legend_items:
        tw, _ = text_size(draw, key, legend_font)
        legend_text_widths.append(tw)
    legend_total_w = sum(
        legend_box_w + legend_text_gap + tw + legend_item_gap for tw in legend_text_widths
    ) - legend_item_gap
    legend_x = plot_left + (plot_width - legend_total_w) / 2
    legend_y = 24
    cursor_x = legend_x
    for key, tw in zip(legend_items, legend_text_widths):
        draw.rectangle(
            (cursor_x, legend_y, cursor_x + legend_box_w, legend_y + legend_box_h),
            fill=hex_to_rgb(COLORS[key]),
        )
        draw.text((cursor_x + legend_box_w + legend_text_gap, legend_y - 1), key, font=legend_font, fill=text_color)
        cursor_x += legend_box_w + legend_text_gap + tw + legend_item_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".png":
        image.save(output_path, dpi=(300, 300))
    else:
        image.save(output_path.with_suffix(".png"), dpi=(300, 300))
        image.save(output_path, "PDF", resolution=300.0)


def main() -> None:
    out_dir = Path("/root/AgentRewrite/figures")
    for stem, config in FIGURES.items():
        for ext in ("png", "pdf"):
            out_path = out_dir / f"{stem}.{ext}"
            draw_chart(config, out_path)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
