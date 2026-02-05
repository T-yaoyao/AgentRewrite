import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import psycopg2
import psycopg2.errors

from src.utils.postgres_connector import PostgresConnector


def create_connector() -> PostgresConnector:
    """
    创建 PostgreSQL 连接器，使用默认配置。
    默认配置：
      - host: 172.17.0.3
      - port: 5432
      - database: dsb
      - user: postgres
      - password: 123456
    """
    host = "127.0.0.1"
    port = 5432
    database = "tpch"
    user = "postgres"
    password = "123456"

    return PostgresConnector(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
    )


def measure_query_time_seconds(conn, sql: str, timeout: Optional[float] = None) -> float:
    """
    执行一条 SQL，返回执行时间（秒）。
    为了确保查询真正执行完成，对于有结果集的语句会 fetchall。
    
    Args:
        conn: 数据库连接
        sql: SQL 查询语句
        timeout: 超时时间（秒），如果为 None 则不设置超时
    
    Returns:
        执行时间（秒），如果超时则返回 timeout 值
    """
    start = time.time()
    cur = None
    
    try:
        cur = conn.cursor()
        
        # 如果设置了超时，使用 statement_timeout
        if timeout is not None:
            cur.execute(f"SET statement_timeout = {int(timeout * 1000)}")  # PostgreSQL 使用毫秒
        
        try:
            cur.execute(sql)
            try:
                # 对于 SELECT 等有结果集的语句，拉取所有结果，确保执行完全结束
                cur.fetchall()
            except psycopg2.ProgrammingError:
                # 对于没有结果集的语句（如 INSERT/UPDATE），忽略此错误
                pass
        except (psycopg2.errors.QueryCanceled, psycopg2.OperationalError) as e:
            # 查询被超时取消
            error_msg = str(e).lower()
            if "timeout" in error_msg or "canceling statement" in error_msg:
                end = time.time()
                elapsed = end - start
                # 如果设置了超时且实际执行时间达到或超过超时时间，返回超时值
                if timeout is not None and elapsed >= timeout:
                    # 本次查询已被超时取消，当前事务处于 aborted 状态，需要回滚
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    if cur:
                        cur.close()
                    return timeout
                # 否则返回实际执行时间
                # 不是严格意义上的“达到超时上限”，但也被取消了，同样需要回滚事务
                try:
                    conn.rollback()
                except Exception:
                    pass
                if cur:
                    cur.close()
                return elapsed
            # 其他操作错误，重新抛出
            # 其他操作错误，同样回滚事务再抛出
            try:
                conn.rollback()
            except Exception:
                pass
            if cur:
                cur.close()
            raise
        
        # 恢复默认超时设置
        if timeout is not None:
            cur.execute("RESET statement_timeout")
        
        conn.commit()
        end = time.time()
        elapsed = end - start
        
        if cur:
            cur.close()
        
        return elapsed
    except Exception as e:
        # 如果发生其他错误，计算已用时间
        end = time.time()
        elapsed = end - start
        
        if cur:
            try:
                cur.close()
            except Exception:
                pass

        # 发生异常时回滚事务，避免后续出现 InFailedSqlTransaction
        try:
            conn.rollback()
        except Exception:
            pass
        
        # 如果是超时相关的错误，返回超时值
        error_msg = str(e).lower()
        if timeout is not None and ("timeout" in error_msg or "canceling statement" in error_msg):
            if elapsed >= timeout:
                return timeout
        
        # 其他错误重新抛出
        raise


def average_middle_three(times: List[float]) -> float:
    """
    给定 5 次执行时间，去掉最大和最小，取中间 3 次的平均值。
    如果次数不足 5（理论上不会发生），则直接取全部的平均值。
    """
    if len(times) < 5:
        return sum(times) / len(times)
    sorted_times = sorted(times)
    middle = sorted_times[1:-1]
    return sum(middle) / len(middle)


def process_csv(
    connector: PostgresConnector,
    input_csv: Path,
    sql_column: Optional[str] = None,
    output_csv: Optional[Path] = None,
) -> None:
    """
    读取 CSV 中的 SQL（列名通过参数或表头自动识别），
    对每条 SQL 执行 5 次，去掉最高和最低的时间，剩余 3 次取平均，
    输出到新的 CSV（id, execution_time_s）。
    """
    # 获取数据库连接，增加友好的错误提示，避免直接抛出 psycopg2 超时异常
    try:
        conn = connector._get_connection()
    except psycopg2.OperationalError as e:
        print("❌ 无法连接到 PostgreSQL 数据库（连接超时或不可达）")
        print(f"   连接配置: host={connector.host}, port={connector.port}, db={connector.database}, user={connector.user}")
        print(f"   详细错误: {e}")
        print("💡 请检查：")
        print("   1) 数据库服务是否已启动并监听对应地址和端口")
        print("   2) Docker 容器 / 远程主机 IP 是否正确（当前使用 172.17.0.3）")
        print("   3) 防火墙或网络策略是否阻止了连接")
        sys.exit(1)

    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # 如未显式指定 sql_column，则根据表头自动推断
        if sql_column is None:
            headers = [h.strip() for h in (reader.fieldnames or [])]
            if "original_query" in headers:
                sql_column = "original_query"
            elif "rewritten_query" in headers:
                sql_column = "rewritten_query"
            else:
                raise ValueError(
                    f"无法在 {input_csv} 中找到 'original_query' 或 'rewritten_query' 列，请检查表头。"
                )

        if output_csv is None:
            # 默认输出到同目录：<原文件名>_exec_times.csv
            output_csv = input_csv.with_name(input_csv.stem + "_exec_times.csv")

        rows = list(reader)
        total = len(rows)
        print(f"开始处理文件: {input_csv}，共有 {total} 条记录")

    # 流式写出结果：一条 SQL 完成就立刻写入 CSV，避免中途异常时全部丢失
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["id", "execution_time_s"])
        writer.writeheader()

        for idx, row in enumerate(rows, start=1):
            sql_id = row.get("id")
            sql = row.get(sql_column)

            if not sql_id or not sql:
                print(f"[{idx}/{total}] 跳过无效记录（缺少 id 或 SQL）")
                continue

            print(f"[{idx}/{total}] 执行 id={sql_id} 的 SQL（5 次取中间 3 次平均）...")
            times: List[float] = []

            # 第一次执行，设置 300s 超时
            first_time = measure_query_time_seconds(conn, sql, timeout=300.0)
            times.append(first_time)

            # 如果第一次执行就超过 300s，直接记录 300s 并跳过后续执行
            if first_time >= 300.0:
                print(
                    f"[{idx}/{total}] ⚠️  第一次执行时间超过 300s ({first_time:.2f}s)，中断后续执行，记录为 300s"
                )
                avg_time = 300.0
            else:
                # 继续执行剩余的 4 次（不设置超时，让它们正常执行）
                print(
                    f"[{idx}/{total}] 第一次执行时间: {first_time:.6f} 秒，继续执行剩余 4 次..."
                )
                for run_num in range(2, 6):  # 第 2-5 次
                    t = measure_query_time_seconds(conn, sql, timeout=None)
                    times.append(t)
                    print(
                        f"[{idx}/{total}] 第 {run_num} 次执行时间: {t:.6f} 秒"
                    )

                avg_time = average_middle_three(times)

            print(
                f"[{idx}/{total}] 完成 id={sql_id}，最终执行时间={avg_time:.6f} 秒"
            )

            writer.writerow(
                {
                    "id": sql_id,
                    "execution_time_s": f"{avg_time:.6f}",
                }
            )
            out_f.flush()
    print(f"处理完成，结果已写入: {output_csv}")


def process_json(
    connector: PostgresConnector,
    input_json: Path,
    output_csv: Optional[Path] = None,
) -> None:
    """
    从 JSON 文件中读取查询并执行。

    期望的 JSON 结构（数组）：
      [
        {"id": "1", "query": "select ...;"},
        {"id": "2", "query": "select ...;"},
        ...
      ]

    对每条 query 执行 5 次，去掉最高和最低的时间，剩余 3 次取平均，
    输出到 CSV（id, execution_time_s）。
    """
    if output_csv is None:
        output_csv = input_json.with_suffix("").with_name(
            input_json.stem + "_exec_times.csv"
        )

    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON 文件 {input_json} 顶层结构必须是数组。")

    conn = connector._get_connection()
    total = len(data)

    print(f"开始处理 JSON 文件: {input_json}，共有 {total} 条记录")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["id", "execution_time_s"])
        writer.writeheader()

        for idx, item in enumerate(data, start=1):
            sql_id = str(item.get("id")) if item.get("id") is not None else None
            sql = item.get("query")

            if not sql_id or not sql:
                print(f"[{idx}/{total}] 跳过无效记录（缺少 id 或 query）")
                continue

            print(
                f"[{idx}/{total}] 执行 id={sql_id} 的 SQL（5 次取中间 3 次平均）..."
            )
            times: List[float] = []

            # 第一次执行，设置 300s 超时
            first_time = measure_query_time_seconds(conn, sql, timeout=300.0)
            times.append(first_time)

            # 如果第一次执行就超过 300s，直接记录 300s 并跳过后续执行
            if first_time >= 300.0:
                print(
                    f"[{idx}/{total}] ⚠️  第一次执行时间超过 300s ({first_time:.2f}s)，中断后续执行，记录为 300s"
                )
                avg_time = 300.0
            else:
                # 继续执行剩余的 4 次（不设置超时，让它们正常执行）
                print(
                    f"[{idx}/{total}] 第一次执行时间: {first_time:.6f} 秒，继续执行剩余 4 次..."
                )
                for run_num in range(2, 6):  # 第 2-5 次
                    t = measure_query_time_seconds(conn, sql, timeout=None)
                    times.append(t)
                    print(
                        f"[{idx}/{total}] 第 {run_num} 次执行时间: {t:.6f} 秒"
                    )

                avg_time = average_middle_three(times)

            print(
                f"[{idx}/{total}] 完成 id={sql_id}，最终执行时间={avg_time:.6f} 秒"
            )

            writer.writerow(
                {
                    "id": sql_id,
                    "execution_time_s": f"{avg_time:.6f}",
                }
            )
            out_f.flush()

    print(f"JSON 处理完成，结果已写入: {output_csv}")


def main():
    """
    使用方式：
      python execute_sql_from_csv.py /path/to/input

    支持的输入格式：
      1. CSV：
         - 自动检查表头，优先使用 original_query 列，其次 rewritten_query 列
         - 输出：同目录下 <原文件名>_exec_times.csv
      2. JSON（如 dataset/queries/tpch_queries.json）：
         - 期望结构：[{ "id": "...", "query": "select ..." }, ...]
         - 输出：同目录下 <原文件名>_exec_times.csv

    行为：
      - 每条 SQL 执行 5 次
      - 去掉执行时间中的最小值和最大值
      - 对剩余 3 次取平均（秒），写入输出 CSV
    """
    if len(sys.argv) < 2:
        print(
            "用法: python execute_sql_from_csv.py /absolute/or/relative/path/to/input.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    input_path = Path(sys.argv[1]).expanduser().resolve()

    if not input_path.exists():
        print(f"输入文件不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    connector = create_connector()

    if input_path.suffix.lower() == ".json":
        process_json(connector=connector, input_json=input_path)
    else:
        process_csv(connector=connector, input_csv=input_path)

    connector.close()


if __name__ == "__main__":
    main()


