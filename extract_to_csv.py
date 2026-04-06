#!/usr/bin/env python3
"""
从 JSON 提取数据到 CSV 文件。

兼容两种结构：
1. 旧结构：`rewritten_query` 字段为 dict，内部有 `tpch` 数组。
2. 新结构：`rewritten_query` 字段直接为字符串（如 `wiriien_query.json`）。
"""

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.llm_json_utils import load_with_repair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 JSON 文件提取查询信息到 CSV。"
    )
    parser.add_argument(
        "-i",
        "--input",
        default="output/tpch_test_deepseekR1/rewritten_queries.json",
        help="输入 JSON 文件路径（默认：output/tpch_test_deepseekR1/rewritten_queries.json）",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output/tpch_test_deepseekR1/rewritten_queries.csv",
        help="输出 CSV 文件路径（默认：output/tpch_test_deepseekR1/rewritten_queries.csv）",
    )
    return parser.parse_args()


def normalize_sql(sql_text: str) -> str:
    """将多行 SQL 压缩成单行，去掉多余空格。"""
    if not sql_text:
        return ""
    return " ".join(sql_text.split())


def main() -> None:
    args = parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    # 读取JSON文件
    with open(input_file, "r", encoding="utf-8") as f:
        data = load_with_repair(f)

    # 准备CSV数据
    csv_rows = []

    for item in data:
        # 原始 SQL（压缩为单行）
        original_sql = normalize_sql(item.get("original_query", ""))

        row = {
            "id": item.get("id", ""),
            "original_query": original_sql,
            "original_costs": "",
            "rewrite_costs": "",
            "costs_reduction_rate": "",
            "time_cost": item.get("time_cost", ""),
            "llm_costs": item.get("llm_costs", ""),
            "rewritten_query": "",
        }

        # 提取 rewritten_query
        rewritten_query_data = item.get("rewritten_query", {})

        # 1) 旧结构：dict，内部有 tpch 数组
        if isinstance(rewritten_query_data, dict) and "tpch" in rewritten_query_data:
            tpch_list = rewritten_query_data["tpch"]
            if tpch_list:
                tpch_item = tpch_list[0]
                rewritten_sql = normalize_sql(tpch_item.get("rewritten_query", ""))
                row["rewritten_query"] = rewritten_sql
                row["original_costs"] = tpch_item.get("original_costs", "")
                row["rewrite_costs"] = tpch_item.get("rewrite_costs", "")
                row["costs_reduction_rate"] = tpch_item.get(
                    "costs_reduction_rate", ""
                )

        # 2) 新结构：rewritten_query 直接是字符串（如 wiriien_query.json）
        elif isinstance(rewritten_query_data, str):
            row["rewritten_query"] = normalize_sql(rewritten_query_data)

        csv_rows.append(row)

    # 写入CSV文件
    fieldnames = [
        "id",
        "original_query",
        "rewritten_query",
        "original_costs",
        "rewrite_costs",
        "costs_reduction_rate",
        "time_cost",
        "llm_costs",
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"✅ 成功从 {input_file} 提取 {len(csv_rows)} 条记录到 {output_file}")


if __name__ == "__main__":
    main()


