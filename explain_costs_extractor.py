#!/usr/bin/env python3
"""
从 JSON 文件中提取 SQL 查询，执行 EXPLAIN 获取成本，并输出到 CSV
支持两种 JSON 格式：
1. rewritten_queries.json 格式：rewritten_query.tpch[0].rewritten_query
2. QUITE_tpch_63queries.json 格式：直接的 original_query 和 rewritten_query 字段
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor


def create_db_connection(database_name: Optional[str] = None):
    """创建 PostgreSQL 数据库连接"""
    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5432"))
    # 优先使用传入的参数，其次使用环境变量，最后使用默认值
    database = database_name or os.getenv("DB_NAME", "dsb")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "123456")

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        return conn
    except psycopg2.Error as e:
        print(f"❌ 数据库连接失败: {e}")
        sys.exit(1)


def extract_cost_from_explain(explain_result: any) -> Optional[float]:
    """
    从 EXPLAIN 结果中提取总成本
    支持多种格式：
    - JSON 格式：{"Plan": {"Total Cost": xxx}}
    - 列表格式：[{"Plan": {"Total Cost": xxx}}]
    """
    try:
        # 如果是字符串，尝试解析 JSON
        if isinstance(explain_result, str):
            try:
                explain_result = json.loads(explain_result)
            except json.JSONDecodeError:
                return None

        # 如果是列表，取第一个元素
        if isinstance(explain_result, list) and len(explain_result) > 0:
            explain_result = explain_result[0]

        # 如果是字典
        if isinstance(explain_result, dict):
            # 格式1: {"Plan": {"Total Cost": xxx}}
            if "Plan" in explain_result:
                plan = explain_result["Plan"]
                if isinstance(plan, dict) and "Total Cost" in plan:
                    return float(plan["Total Cost"])

            # 格式2: {"total_cost": xxx}
            if "total_cost" in explain_result:
                return float(explain_result["total_cost"])

            # 格式3: 直接包含 "Total Cost"
            if "Total Cost" in explain_result:
                return float(explain_result["Total Cost"])

    except (KeyError, TypeError, ValueError) as e:
        print(f"⚠️  解析 EXPLAIN 结果失败: {e}")
        return None

    return None


def get_query_cost(conn, sql: str) -> Optional[float]:
    """
    执行 EXPLAIN (FORMAT JSON) 并提取成本
    """
    try:
        # 如果事务被中断，先回滚
        try:
            conn.rollback()
        except:
            pass
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(f"EXPLAIN (FORMAT JSON) {sql}")
        result = cursor.fetchone()
        cursor.close()
        conn.commit()  # 提交成功的事务

        if result:
            # PostgreSQL 返回格式: [{"QUERY PLAN": [{"Plan": {...}}]}]
            query_plan = result.get("QUERY PLAN", None)
            if query_plan:
                cost = extract_cost_from_explain(query_plan)
                return cost

    except psycopg2.Error as e:
        # 发生错误时回滚事务
        try:
            conn.rollback()
        except:
            pass
        print(f"⚠️  执行 EXPLAIN 失败 (SQL: {sql[:50]}...): {e}")
        return None
    except Exception as e:
        # 发生错误时回滚事务
        try:
            conn.rollback()
        except:
            pass
        print(f"⚠️  获取成本时发生错误: {e}")
        return None

    return None


def extract_queries_from_item(item: dict) -> List[Tuple[str, str]]:
    """
    从 JSON 项中提取 (original_query, rewritten_query) 对
    支持两种格式
    """
    queries = []

    # 格式1: rewritten_queries.json 格式
    if "rewritten_query" in item and isinstance(item["rewritten_query"], dict):
        rewritten_data = item["rewritten_query"]
        if "tpch" in rewritten_data and isinstance(rewritten_data["tpch"], list):
            for tpch_item in rewritten_data["tpch"]:
                if "rewritten_query" in tpch_item:
                    original = item.get("original_query", "")
                    rewritten = tpch_item.get("rewritten_query", "")
                    if original and rewritten:
                        queries.append((original, rewritten))
            return queries

    # 格式2: QUITE_tpch_63queries.json 格式
    if "original_query" in item and "rewritten_query" in item:
        original = item.get("original_query", "")
        rewritten = item.get("rewritten_query", "")
        if original and rewritten:
            queries.append((original, rewritten))
        return queries

    return queries


def process_json_file(input_file: Path, output_file: Path, database_name: Optional[str] = None):
    """
    处理 JSON 文件，提取 SQL 查询，执行 EXPLAIN，并输出到 CSV
    """
    print(f"📂 读取 JSON 文件: {input_file}")

    # 读取 JSON 文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取 JSON 文件失败: {e}")
        sys.exit(1)

    # 过滤掉非字典项（如 QUITE_tpch_63queries.json 的第一项是统计信息列表）
    if isinstance(data, list):
        data = [item for item in data if isinstance(item, dict)]

    # 连接数据库
    db_name = database_name or os.getenv("DB_NAME", "tpch")
    print(f"🔌 连接数据库: {db_name}...")
    conn = create_db_connection(database_name)
    print("✅ 数据库连接成功")

    # 准备 CSV 数据
    csv_rows = []
    total = len(data)
    processed = 0
    error_ids = []  # 记录处理失败的 ID 列表

    print(f"\n📊 开始处理 {total} 条查询...")
    print("=" * 60)

    for item in data:
        # 跳过非字典项
        if not isinstance(item, dict):
            continue

        query_id = str(item.get("id", ""))
        if not query_id:
            continue

        # 提取查询对
        query_pairs = extract_queries_from_item(item)
        if not query_pairs:
            print(f"⚠️  跳过 ID {query_id}: 未找到查询")
            error_ids.append(query_id)
            continue

        # 取第一个查询对（通常只有一个）
        original_sql, rewritten_sql = query_pairs[0]

        processed += 1
        print(f"\n[{processed}/{total}] 处理查询 ID: {query_id}")

        # 获取原始 SQL 成本
        print("  📈 获取原始 SQL 成本...")
        original_cost = get_query_cost(conn, original_sql)
        if original_cost is None:
            print(f"  ⚠️  无法获取原始 SQL 成本，跳过")
            error_ids.append(query_id)
            continue

        # 获取重写 SQL 成本
        print("  📈 获取重写 SQL 成本...")
        rewritten_cost = get_query_cost(conn, rewritten_sql)
        if rewritten_cost is None:
            print(f"  ⚠️  无法获取重写 SQL 成本，跳过")
            error_ids.append(query_id)
            continue

        # 计算成本降低率
        if original_cost > 0:
            costs_reduction_rate = (original_cost - rewritten_cost) / original_cost
        else:
            costs_reduction_rate = 0.0

        print(f"  ✅ 原始成本: {original_cost:.2f}, 重写成本: {rewritten_cost:.2f}, "
              f"降低率: {costs_reduction_rate:.4f}")

        csv_rows.append({
            "id": query_id,
            "original_costs": f"{original_cost:.2f}",
            "rewrite_costs": f"{rewritten_cost:.2f}",
            "costs_reduction_rate": f"{costs_reduction_rate:.6f}"
        })

    # 关闭数据库连接
    conn.close()
    print("\n" + "=" * 60)
    print(f"✅ 处理完成，共处理 {processed} 条查询")

    # 输出处理失败的 ID
    print("\n📋 处理失败的查询 ID 汇总：")
    if error_ids:
        unique_error_ids = sorted(set(error_ids), key=lambda x: int(x) if x.isdigit() else x)
        print(f"  共 {len(unique_error_ids)} 条查询处理失败：")
        print("  IDs:", ", ".join(unique_error_ids))
    else:
        print("  ✅ 所有查询均成功处理，没有失败的 ID")

    # 写入 CSV 文件
    print(f"\n💾 写入 CSV 文件: {output_file}")
    fieldnames = ["id", "original_costs", "rewrite_costs", "costs_reduction_rate"]

    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"✅ CSV 文件写入成功: {output_file}")
        print(f"📊 共写入 {len(csv_rows)} 条记录")
    except Exception as e:
        print(f"❌ 写入 CSV 文件失败: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="从 JSON 文件中提取 SQL 查询，执行 EXPLAIN 获取成本，并输出到 CSV"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="输入 JSON 文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="输出 CSV 文件路径（默认：输入文件同目录下的 query_costs_comparison.csv）"
    )
    parser.add_argument(
        "--database", "-d",
        type=str,
        default=None,
        help="数据库名称（默认：使用环境变量 DB_NAME，或 'tpch'）"
    )

    args = parser.parse_args()

    # 检查输入文件是否存在
    input_file = args.input
    if not input_file.exists():
        print(f"❌ 输入文件不存在: {input_file}")
        sys.exit(1)

    # 确定输出文件路径
    if args.output:
        output_file = args.output
    else:
        # 默认输出到输入文件同目录
        output_file = input_file.parent / "query_costs_comparison.csv"

    # 处理文件
    process_json_file(input_file, output_file, args.database)


if __name__ == "__main__":
    main()

