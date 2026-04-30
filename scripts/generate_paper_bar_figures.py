#!/usr/bin/env python3

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
    "AgentRewrite",
]

COLORS = {
    "TPCH": "#4C78A8",
    "DSB": "#F58518",
    "Calcite": "#54A24B",
}

SERIES_ORDER = ["TPCH", "DSB", "Calcite"]

LAYOUTS: Dict[str, Dict[str, object]] = {
    "default": {
        "width": 2200,
        "height": 1200,
        "margin_left": 180,
        "margin_right": 70,
        "margin_top": 120,
        "margin_bottom": 210,
        "title_size": 34,
        "label_size": 24,
        "small_size": 20,
        "value_size": 20,
        "legend_size": 20,
        "tick_size": 18,
        "title_y": 28,
        "show_title": True,
        "y_label_x": 65,
        "legend_width": 520,
        "legend_y": 38,
        "bar_radius": 6,
        "grid_width": 2,
        "axis_width": 3,
        "value_gap": 10,
        "label_gap": 18,
        "label_line_gap": 24,
    },
    "compact": {
        "width": 1500,
        "height": 760,
        "margin_left": 120,
        "margin_right": 35,
        "margin_top": 42,
        "margin_bottom": 155,
        "title_size": 22,
        "label_size": 16,
        "small_size": 15,
        "value_size": 15,
        "legend_size": 15,
        "tick_size": 12,
        "title_y": 18,
        "show_title": False,
        "y_label_x": 38,
        "legend_width": 320,
        "legend_y": 16,
        "bar_radius": 4,
        "grid_width": 1,
        "axis_width": 2,
        "value_gap": 6,
        "label_gap": 12,
        "label_line_gap": 16,
    },
    "column": {
        "width": 1320,
        "height": 720,
        "margin_left": 118,
        "margin_right": 28,
        "margin_top": 42,
        "margin_bottom": 168,
        "title_size": 22,
        "label_size": 18,
        "small_size": 15,
        "value_size": 13,
        "legend_size": 15,
        "tick_size": 15,
        "title_y": 10,
        "show_title": False,
        "y_label_x": 36,
        "legend_width": 430,
        "legend_y": 12,
        "bar_radius": 3,
        "grid_width": 1,
        "axis_width": 1,
        "value_gap": 6,
        "label_gap": 16,
        "label_line_gap": 18,
    },
}


FIGURES: Dict[str, Dict[str, object]] = {
    "fig3_improvement_rate": {
        "title": "Improvement Rate Across Benchmarks",
        "y_label": "Improvement Rate (%)",
        "y_min": 0.0,
        "y_max": 50.0,
        "tick_step": 10.0,
        "series": {
            "TPCH": [33.30, 23.80, 26.99, 41.27, 36.50, 39.68, 42.85],
            "DSB": [19.87, 29.49, 25.64, 24.00, 22.44, 35.90, 38.46],
            "Calcite": [20.69, 27.59, 26.98, 27.59, 31.03, 34.48, 43.10],
        },
    },
    "fig4_equivalence_rate": {
        "title": "Equivalence Rate Across Benchmarks",
        "y_label": "Equivalence Rate (%)",
        "y_min": 70.0,
        "y_max": 102.0,
        "tick_step": 5.0,
        "series": {
            "TPCH": [93.65, 93.65, 93.65, 98.41, 96.23, 98.00, 100.00],
            "DSB": [73.71, 85.89, 85.25, 82.05, 83.30, 95.51, 96.79],
            "Calcite": [81.03, 82.75, 82.75, 87.93, 86.20, 91.37, 96.55],
        },
    },
}


def load_font(preferred: Sequence[str], size: int) -> ImageFont.ImageFont:
    for path in preferred:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


TITLE_FONT = load_font(
    [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
    ],
    34,
)
LABEL_FONT = load_font(
    [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    ],
    24,
)
SMALL_FONT = load_font(
    [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ],
    20,
)
TICK_FONT = load_font(
    [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ],
    18,
)


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align="center")
    return box[2] - box[0], box[3] - box[1]


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    if bold:
        return load_font(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
            ],
            size,
        )
    return load_font(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ],
        size,
    )


def layout_values(layout_name: str) -> Dict[str, float]:
    layout = LAYOUTS[layout_name]
    return {k: float(v) if isinstance(v, int) else v for k, v in layout.items()}  # type: ignore[return-value]


def approx_text_width(text: str, font_size: float, bold: bool = False) -> float:
    lines = text.split("\n")
    factor = 0.60 if bold else 0.54
    return max(len(line) for line in lines) * font_size * factor


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def draw_png(output_path: Path, config: Dict[str, object], layout_name: str = "default") -> None:
    layout = layout_values(layout_name)
    width, height = int(layout["width"]), int(layout["height"])
    margin_left, margin_right = int(layout["margin_left"]), int(layout["margin_right"])
    margin_top, margin_bottom = int(layout["margin_top"]), int(layout["margin_bottom"])
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    axis_color = (60, 60, 60)
    grid_color = (220, 220, 220)
    text_color = (25, 25, 25)

    title = str(config["title"])
    y_label = str(config["y_label"])
    y_min = float(config["y_min"])
    y_max = float(config["y_max"])
    tick_step = float(config["tick_step"])
    series = config["series"]  # type: ignore[assignment]

    title_font = get_font(int(layout["title_size"]), bold=True)
    label_font = get_font(int(layout["label_size"]), bold=False)
    small_font = get_font(int(layout["small_size"]), bold=False)
    value_font = get_font(int(layout.get("value_size", layout["small_size"])), bold=False)
    legend_font = get_font(int(layout.get("legend_size", layout["small_size"])), bold=False)
    tick_font = get_font(int(layout["tick_size"]), bold=False)

    if bool(layout["show_title"]):
        draw.text((plot_left, int(layout["title_y"])), title, font=title_font, fill=text_color)

    # Y-axis label
    label_img = Image.new("RGBA", (320, 80), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((0, 0), y_label, font=label_font, fill=text_color)
    label_img = label_img.rotate(90, expand=True)
    image.paste(label_img, (int(layout["y_label_x"]) - 25, plot_top + plot_height // 2 - label_img.size[1] // 2), label_img)

    # Grid and y ticks
    ticks = []
    current = y_min
    while current <= y_max + 1e-9:
        ticks.append(round(current, 2))
        current += tick_step

    def value_to_y(v: float) -> float:
        return plot_bottom - (v - y_min) / (y_max - y_min) * plot_height

    for t in ticks:
        y = value_to_y(t)
        draw.line((plot_left, y, plot_right, y), fill=grid_color, width=int(layout["grid_width"]))
        label = f"{t:.0f}" if abs(t - round(t)) < 1e-9 else f"{t:.2f}"
        w, h = text_size(draw, label, tick_font)
        draw.text((plot_left - 12 - w, y - h / 2), label, font=tick_font, fill=text_color)

    # Axes
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=axis_color, width=int(layout["axis_width"]))
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=axis_color, width=int(layout["axis_width"]))

    group_count = len(METHODS)
    group_width = plot_width / group_count
    bar_width = group_width * 0.19
    intra_gap = bar_width * 0.18
    group_inner_width = 3 * bar_width + 2 * intra_gap

    for idx, method in enumerate(METHODS):
        group_center = plot_left + group_width * (idx + 0.5)
        start_x = group_center - group_inner_width / 2
        for s_idx, key in enumerate(SERIES_ORDER):
            val = float(series[key][idx])  # type: ignore[index]
            x0 = start_x + s_idx * (bar_width + intra_gap)
            x1 = x0 + bar_width
            y1 = plot_bottom
            y0 = value_to_y(val)
            draw.rounded_rectangle(
                (x0, y0, x1, y1),
                radius=int(layout["bar_radius"]),
                fill=hex_to_rgb(COLORS[key]),
                outline=None,
            )

            val_label = f"{val:.2f}".rstrip("0").rstrip(".")
            tw, th = text_size(draw, val_label, value_font)
            draw.text((x0 + (bar_width - tw) / 2, y0 - th - int(layout["value_gap"])), val_label, font=value_font, fill=text_color)

        draw.multiline_text(
            (group_center, plot_bottom + int(layout["label_gap"])),
            method,
            font=small_font,
            fill=text_color,
            anchor="ma",
            align="center",
            spacing=max(2, int(layout["label_line_gap"] - layout["small_size"])),
        )

    # Legend
    legend_x = plot_right - int(layout["legend_width"])
    legend_y = int(layout["legend_y"])
    cursor_x = legend_x
    for key in SERIES_ORDER:
        draw.rounded_rectangle(
            (cursor_x, legend_y, cursor_x + 28, legend_y + 18),
            radius=4,
            fill=hex_to_rgb(COLORS[key]),
        )
        draw.text((cursor_x + 36, legend_y - 2), key, font=legend_font, fill=text_color)
        if layout_name == "compact":
            cursor_x += 100
        elif layout_name == "column":
            cursor_x += 125
        else:
            cursor_x += 150

    image.save(output_path, dpi=(300, 300))


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def draw_svg(output_path: Path, config: Dict[str, object], layout_name: str = "default") -> None:
    layout = layout_values(layout_name)
    width, height = int(layout["width"]), int(layout["height"])
    margin_left, margin_right = int(layout["margin_left"]), int(layout["margin_right"])
    margin_top, margin_bottom = int(layout["margin_top"]), int(layout["margin_bottom"])
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    y_min = float(config["y_min"])
    y_max = float(config["y_max"])
    tick_step = float(config["tick_step"])
    title = str(config["title"])
    y_label = str(config["y_label"])
    series = config["series"]  # type: ignore[assignment]

    def value_to_y(v: float) -> float:
        return plot_bottom - (v - y_min) / (y_max - y_min) * plot_height

    ticks = []
    current = y_min
    while current <= y_max + 1e-9:
        ticks.append(round(current, 2))
        current += tick_step

    group_count = len(METHODS)
    group_width = plot_width / group_count
    bar_width = group_width * 0.19
    intra_gap = bar_width * 0.18
    group_inner_width = 3 * bar_width + 2 * intra_gap

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    if bool(layout["show_title"]):
        parts.append(
            f'<text x="{plot_left}" y="{int(layout["title_y"]) + int(layout["title_size"])}" '
            f'font-family="DejaVu Serif, serif" font-size="{int(layout["title_size"])}" font-weight="700" fill="#191919">{svg_escape(title)}</text>'
        )
    parts.append(
        f'<text x="{int(layout["y_label_x"])}" y="{plot_top + plot_height / 2}" '
        f'transform="rotate(-90 {int(layout["y_label_x"])} {plot_top + plot_height / 2})" '
        f'font-family="DejaVu Serif, serif" font-size="{int(layout["label_size"])}" fill="#191919">{svg_escape(y_label)}</text>'
    )

    for t in ticks:
        y = value_to_y(t)
        label = f"{t:.0f}" if abs(t - round(t)) < 1e-9 else f"{t:.2f}"
        parts.append(
            f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" '
            f'stroke="#dddddd" stroke-width="{int(layout["grid_width"])}"/>'
        )
        parts.append(
            f'<text x="{plot_left - 14}" y="{y + 5:.2f}" text-anchor="end" '
            f'font-family="DejaVu Sans, sans-serif" font-size="{int(layout["tick_size"])}" fill="#191919">{svg_escape(label)}</text>'
        )

    parts.append(
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#3c3c3c" stroke-width="{int(layout["axis_width"])}"/>'
    )
    parts.append(
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#3c3c3c" stroke-width="{int(layout["axis_width"])}"/>'
    )

    for idx, method in enumerate(METHODS):
        group_center = plot_left + group_width * (idx + 0.5)
        start_x = group_center - group_inner_width / 2
        for s_idx, key in enumerate(SERIES_ORDER):
            val = float(series[key][idx])  # type: ignore[index]
            x0 = start_x + s_idx * (bar_width + intra_gap)
            y0 = value_to_y(val)
            height_bar = plot_bottom - y0
            parts.append(
                f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{bar_width:.2f}" height="{height_bar:.2f}" '
                f'rx="{int(layout["bar_radius"])}" ry="{int(layout["bar_radius"])}" fill="{COLORS[key]}"/>'
            )
            val_label = f"{val:.2f}".rstrip("0").rstrip(".")
            parts.append(
                f'<text x="{x0 + bar_width / 2:.2f}" y="{y0 - int(layout["value_gap"]):.2f}" text-anchor="middle" '
                f'font-family="DejaVu Sans, sans-serif" font-size="{int(layout.get("value_size", layout["small_size"]))}" fill="#191919">{svg_escape(val_label)}</text>'
            )

        lines = method.split("\n")
        base_y = plot_bottom + int(layout["label_gap"]) + int(layout["small_size"])
        for line_idx, line in enumerate(lines):
            parts.append(
                f'<text x="{group_center:.2f}" y="{base_y + int(layout["label_line_gap"]) * line_idx:.2f}" text-anchor="middle" '
                f'font-family="DejaVu Sans, sans-serif" font-size="{int(layout["small_size"])}" fill="#191919">{svg_escape(line)}</text>'
            )

    legend_x = plot_right - int(layout["legend_width"])
    legend_y = int(layout["legend_y"])
    cursor_x = legend_x
    for key in SERIES_ORDER:
        parts.append(f'<rect x="{cursor_x}" y="{legend_y}" width="28" height="18" rx="4" ry="4" fill="{COLORS[key]}"/>')
        parts.append(
            f'<text x="{cursor_x + 36}" y="{legend_y + 14}" font-family="DejaVu Sans, sans-serif" font-size="{int(layout.get("legend_size", layout["small_size"]))}" fill="#191919">{key}</text>'
        )
        if layout_name == "compact":
            cursor_x += 100
        elif layout_name == "column":
            cursor_x += 125
        else:
            cursor_x += 150

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def draw_pdf(output_path: Path, config: Dict[str, object], layout_name: str = "default") -> None:
    layout = layout_values(layout_name)
    width, height = float(layout["width"]), float(layout["height"])
    margin_left, margin_right = float(layout["margin_left"]), float(layout["margin_right"])
    margin_top, margin_bottom = float(layout["margin_top"]), float(layout["margin_bottom"])
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    y_min = float(config["y_min"])
    y_max = float(config["y_max"])
    tick_step = float(config["tick_step"])
    title = str(config["title"])
    y_label = str(config["y_label"])
    series = config["series"]  # type: ignore[assignment]

    ticks = []
    current = y_min
    while current <= y_max + 1e-9:
        ticks.append(round(current, 2))
        current += tick_step

    def value_to_y_top(v: float) -> float:
        return plot_bottom - (v - y_min) / (y_max - y_min) * plot_height

    def pdf_y(y_top: float) -> float:
        return height - y_top

    group_count = len(METHODS)
    group_width = plot_width / group_count
    bar_width = group_width * 0.19
    intra_gap = bar_width * 0.18
    group_inner_width = 3 * bar_width + 2 * intra_gap

    content: List[str] = []

    def set_fill_rgb(hex_color: str) -> None:
        r, g, b = hex_to_rgb(hex_color)
        content.append(f"{r/255:.4f} {g/255:.4f} {b/255:.4f} rg")

    def set_stroke_rgb(hex_color: str) -> None:
        r, g, b = hex_to_rgb(hex_color)
        content.append(f"{r/255:.4f} {g/255:.4f} {b/255:.4f} RG")

    def draw_line(x1: float, y1_top: float, x2: float, y2_top: float, hex_color: str, line_width: float) -> None:
        set_stroke_rgb(hex_color)
        content.append(f"{line_width:.2f} w")
        content.append(f"{x1:.2f} {pdf_y(y1_top):.2f} m {x2:.2f} {pdf_y(y2_top):.2f} l S")

    def fill_rect(x: float, y_top: float, w: float, h: float, hex_color: str) -> None:
        set_fill_rgb(hex_color)
        content.append(f"{x:.2f} {pdf_y(y_top + h):.2f} {w:.2f} {h:.2f} re f")

    def draw_text(
        text: str,
        x: float,
        y_top: float,
        font_size: float,
        font_key: str = "F1",
        align: str = "left",
        rotate: bool = False,
        line_gap: float | None = None,
    ) -> None:
        lines = text.split("\n")
        if line_gap is None:
            line_gap = font_size + 4
        for idx, line in enumerate(lines):
            tx = x
            if align == "center":
                tx = x - approx_text_width(line, font_size, bold=(font_key == "F2")) / 2
            elif align == "right":
                tx = x - approx_text_width(line, font_size, bold=(font_key == "F2"))
            ty_top = y_top + idx * line_gap
            ty = pdf_y(ty_top)
            esc = pdf_escape(line)
            content.append("BT")
            content.append(f"/{font_key} {font_size:.2f} Tf")
            content.append("0.0980 0.0980 0.0980 rg")
            if rotate:
                content.append(f"0 1 -1 0 {tx:.2f} {ty:.2f} Tm")
            else:
                content.append(f"1 0 0 1 {tx:.2f} {ty:.2f} Tm")
            content.append(f"({esc}) Tj")
            content.append("ET")

    # background
    fill_rect(0, 0, width, height, "#FFFFFF")

    if bool(layout["show_title"]):
        draw_text(title, plot_left, float(layout["title_y"]) + float(layout["title_size"]), float(layout["title_size"]), font_key="F2")

    draw_text(
        y_label,
        float(layout["y_label_x"]),
        plot_top + plot_height / 2,
        float(layout["label_size"]),
        font_key="F1",
        rotate=True,
    )

    for t in ticks:
        y = value_to_y_top(t)
        label = f"{t:.0f}" if abs(t - round(t)) < 1e-9 else f"{t:.2f}"
        draw_line(plot_left, y, plot_right, y, "#DDDDDD", float(layout["grid_width"]))
        draw_text(label, plot_left - 14, y + 5, float(layout["tick_size"]), font_key="F1", align="right")

    draw_line(plot_left, plot_top, plot_left, plot_bottom, "#3C3C3C", float(layout["axis_width"]))
    draw_line(plot_left, plot_bottom, plot_right, plot_bottom, "#3C3C3C", float(layout["axis_width"]))

    for idx, method in enumerate(METHODS):
        group_center = plot_left + group_width * (idx + 0.5)
        start_x = group_center - group_inner_width / 2
        for s_idx, key in enumerate(SERIES_ORDER):
            val = float(series[key][idx])  # type: ignore[index]
            x0 = start_x + s_idx * (bar_width + intra_gap)
            y0 = value_to_y_top(val)
            bar_h = plot_bottom - y0
            fill_rect(x0, y0, bar_width, bar_h, COLORS[key])
            val_label = f"{val:.2f}".rstrip("0").rstrip(".")
            draw_text(
                val_label,
                x0 + bar_width / 2,
                y0 - float(layout["value_gap"]),
                float(layout.get("value_size", layout["small_size"])),
                align="center",
            )

        draw_text(
            method,
            group_center,
            plot_bottom + float(layout["label_gap"]) + float(layout["small_size"]),
            float(layout["small_size"]),
            align="center",
            line_gap=float(layout["label_line_gap"]),
        )

    legend_x = plot_right - float(layout["legend_width"])
    legend_y = float(layout["legend_y"])
    cursor_x = legend_x
    for key in SERIES_ORDER:
        fill_rect(cursor_x, legend_y, 28, 18, COLORS[key])
        draw_text(key, cursor_x + 36, legend_y + 14, float(layout.get("legend_size", layout["small_size"])))
        if layout_name == "compact":
            cursor_x += 100
        elif layout_name == "column":
            cursor_x += 125
        else:
            cursor_x += 150

    content_stream = "\n".join(content).encode("latin-1", errors="replace")
    objects: List[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.2f} {height:.2f}] /Contents 4 0 R "
        f"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>".encode("latin-1")
    )
    objects.append(f"<< /Length {len(content_stream)} >>\nstream\n".encode("latin-1") + content_stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    pdf = bytearray()
    pdf.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode("latin-1")
    )
    output_path.write_bytes(pdf)


def main() -> None:
    out_dir = Path("/root/AgentRewrite/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fig3_improvement_rate": ("fig3_improvement_rate_column.png", "improvement_rate.png"),
        "fig4_equivalence_rate": ("fig4_equivalence_rate_column.png", "equivalence_rate.png"),
    }
    for stem, filenames in outputs.items():
        config = FIGURES[stem]
        for filename in filenames:
            output_path = out_dir / filename
            draw_png(output_path, config, layout_name="column")
            print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
