#!/usr/bin/env python3
"""
从 rewritten_queries.json 中提取：
1) 全部原始SQL（不管是否被重写）
   输出到 original_unrewritten.csv（或用户指定文件名），列: id, original_query
2) 应用了规则或与原始SQL不同的重写SQL
   输出到 rewrite_sql.csv（或用户指定文件名），列: id, rewritten_query
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def load_rewrites(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON 根应为列表(list)")
        return data


def extract_original_unrewritten(
    data: List[Dict[str, Any]], output_path: Path
) -> None:
    """
    提取全部原始 SQL，写入 original_unrewritten.csv（或用户指定的文件名）：

    支持两种 JSON 结构：
    1) 旧结构（来自 rewritten_queries.json）：
       - item["original_query"]
    2) 实验结果结构（QUITE_tpch_63queries.json）：
       - item["original_query"]
    
    注意：现在输出所有记录的原始SQL，不管是否被重写
    """
    rows = []

    def clean_sql(sql: str) -> str:
        """
        清理 SQL：
        - 删除单行注释和多行注释
        - 去掉多余换行和空格，压成一行
        - 末尾没有分号则补上分号
        """
        if not sql:
            return ""
        # 删除单行注释 --
        sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
        # 删除多行注释 /* ... */
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        # 压缩空白为一行
        sql = " ".join(sql.replace("\n", " ").split())
        # 如果末尾没有分号，则补上分号
        if sql and not sql.rstrip().endswith(";"):
            sql = sql.rstrip() + ";"
        return sql

    for item in data:
        # 跳过非字典项（例如统计信息列表等）
        if not isinstance(item, dict):
            continue

        _id = str(item.get("id", "")).strip()
        if not _id:
            continue

        original_sql = item.get("original_query", "") or ""
        if not isinstance(original_sql, str):
            original_sql = str(original_sql)

        # 提取所有原始SQL（不管是否被重写）
        if original_sql:
            flat_sql = clean_sql(original_sql)
            rows.append({"id": _id, "original_query": flat_sql})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "original_query"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"✅ {output_path.name} 生成完成，记录数: {len(rows)}")


def extract_rewrite_sql(data: List[Dict[str, Any]], output_path: Path) -> None:
    """
    提取“被实际重写”的 SQL，写入 rewrite_sql.csv：

    规则：只要重写 SQL 存在，且与原始 SQL 文本不一致，就认为是“被重写”的 SQL，
    不再依赖 rewrite_rules 字段。
    """
    rows: List[Dict[str, str]] = []

    def clean_sql(sql: str) -> str:
        """
        清理 SQL：
        - 删除单行注释和多行注释
        - 去掉多余换行和空格，压成一行
        - 末尾没有分号则补上分号
        """
        if not sql:
            return ""
        sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        sql = " ".join(sql.replace("\n", " ").split())
        if sql and not sql.rstrip().endswith(";"):
            sql = sql.rstrip() + ";"
        return sql

    for item in data:
        # 跳过非字典项
        if not isinstance(item, dict):
            continue

        _id = str(item.get("id", "")).strip()
        if not _id:
            continue

        original_sql = item.get("original_query", "") or ""
        if not isinstance(original_sql, str):
            original_sql = str(original_sql)

        rewritten_field = item.get("rewritten_query")

        # 情况 1：旧结构，rewritten_query 是一个包含 tpch 列表的 dict
        if isinstance(rewritten_field, dict):
            tpch_data = rewritten_field.get("tpch")

            # tpch 为列表（正常情况）
            if isinstance(tpch_data, list) and tpch_data:
                try:
                    first = tpch_data[0]
                    if not isinstance(first, dict):
                        continue
                    rewritten_sql = first.get("rewritten_query", "") or ""
                    if rewritten_sql and original_sql.strip() != rewritten_sql.strip():
                        flat_sql = clean_sql(rewritten_sql)
                        rows.append({"id": _id, "rewritten_query": flat_sql})
                except (IndexError, KeyError, TypeError):
                    continue

            # tpch 为单个对象
            elif isinstance(tpch_data, dict):
                rewritten_sql = tpch_data.get("rewritten_query", "") or ""
                if rewritten_sql and original_sql.strip() != rewritten_sql.strip():
                    flat_sql = clean_sql(rewritten_sql)
                    rows.append({"id": _id, "rewritten_query": flat_sql})

            # 其他情况：tpch 字段不存在或为空，跳过
            else:
                continue

        # 情况 2：实验结果结构，rewritten_query 为字符串
        elif isinstance(rewritten_field, str):
            rewritten_sql = rewritten_field
            if rewritten_sql and original_sql.strip() != rewritten_sql.strip():
                flat_sql = clean_sql(rewritten_sql)
                rows.append({"id": _id, "rewritten_query": flat_sql})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "rewritten_query"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"✅ rewrite_sql.csv 生成完成，记录数: {len(rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从 rewritten_queries.json 中提取：\n"
            "1) 全部原始SQL（不管是否被重写），输出到 original_unrewritten.csv（或用户指定文件名）\n"
            "2) 应用了规则或与原始SQL不同的重写SQL，输出到 rewrite_sql.csv（或用户指定文件名）"
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="output/tpch_test/rewritten_queries.json",
        help="输入 JSON 文件路径（默认: output/tpch_test/rewritten_queries.json）",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output/tpch_test",
        help="输出目录（默认: output/tpch_test）",
    )
    parser.add_argument(
        "--original-filename",
        type=str,
        default="original_unrewritten.csv",
        help="全部原始 SQL 输出文件名（默认: original_unrewritten.csv）",
    )
    parser.add_argument(
        "--rewrite-filename",
        type=str,
        default="rewrite_sql.csv",
        help="重写 SQL 输出文件名（默认: rewrite_sql.csv）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    json_file = Path(args.input)

    if not json_file.exists():
        print(f"❌ 找不到 JSON 文件: {json_file}")
        return

    output_dir = Path(args.output_dir)

    print(f"📂 读取 JSON: {json_file}")
    print(f"📁 输出目录: {output_dir}")
    data = load_rewrites(json_file)

    original_csv = output_dir / args.original_filename
    rewrite_csv = output_dir / args.rewrite_filename

    extract_original_unrewritten(data, original_csv)
    extract_rewrite_sql(data, rewrite_csv)


if __name__ == "__main__":
    main()


