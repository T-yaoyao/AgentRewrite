#!/usr/bin/env python3
"""
Verify SQL equivalence for Rule_Examples.json against PostgreSQL.

The script creates an isolated schema, populates a compact fixture database that
covers the tables used by all rule examples, executes each original/rewritten
pair, and compares result multisets following evaluation.py's result comparison
style.
"""

import argparse
import collections
import json
import os
import sys
import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import psycopg2
from psycopg2 import sql


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.path_config import PROJECT_ROOT as CONFIGURED_PROJECT_ROOT
from src.utils.path_config import load_project_env, setup_python_path


setup_python_path()
load_project_env()

DEFAULT_EXAMPLES_PATH = (
    CONFIGURED_PROJECT_ROOT
    / "src"
    / "Rewrite_Middleware"
    / "Structured_Knowledge_Base"
    / "preparation"
    / "data"
    / "Rule_Examples.json"
)
DEFAULT_STORAGE_PATH = CONFIGURED_PROJECT_ROOT / "output" / "rule_examples_equivalence_results.json"
DEFAULT_FILTERED_PATH = CONFIGURED_PROJECT_ROOT / "output" / "rule_examples_equivalence_summary.json"


COMMON_TABLE_COLUMNS = """
    id integer,
    k integer,
    val integer,
    ref integer,
    a integer,
    b integer,
    c integer,
    col integer,
    x integer,
    y integer,
    name text,
    status text,
    category text,
    t1_id integer,
    amount numeric,
    tag text
"""


COMMON_TABLES = ("t1", "t2", "t3", "t1_a", "t1_b", "t2_a", "t2_b")


FIXTURE_SQL = f"""
CREATE TABLE t (
    id integer,
    name text,
    a integer,
    b integer,
    c integer,
    description text,
    val integer,
    x integer,
    age integer,
    col integer,
    status text,
    category text
);

CREATE TABLE emp (
    empno integer,
    deptno integer,
    job text,
    salary numeric,
    sal numeric,
    name text
);

CREATE TABLE emp_us (
    deptno integer
);

CREATE TABLE emp_cn (
    deptno integer
);

CREATE TABLE employee (
    name text,
    age integer,
    description text
);

CREATE TABLE part (
    part_id integer,
    part_name text,
    type text
);

CREATE TABLE lineitem (
    order_id integer,
    part_id integer
);

{chr(10).join(f"CREATE TABLE {table_name} ({COMMON_TABLE_COLUMNS});" for table_name in COMMON_TABLES)}

INSERT INTO t (id, name, a, b, c, description, val, x, age, col, status, category) VALUES
    (1, 'alpha', 1, 10, 100, 'first', 10, 1, 25, 30, 'ACTIVE', 'A'),
    (2, 'beta', 2, 20, 200, 'second', 20, 2, 35, 20, 'INACTIVE', 'A'),
    (3, 'gamma', 11, 30, 300, 'third', 30, 3, 45, 10, 'ACTIVE', 'B'),
    (6, 'delta', 12, 40, 400, 'fourth', 40, 2, 55, 40, 'ACTIVE', 'B'),
    (NULL, 'null_id', NULL, 50, 500, 'null row', NULL, NULL, 65, 50, 'ACTIVE', 'C');

INSERT INTO t1 (id, k, val, ref, a, b, c, col, x, name, status, category, t1_id, amount, tag) VALUES
    (1, 10, 100, 7, 1, 10, 100, 30, 1, 't1_a', 'ACTIVE', 'A', NULL, 5, 'red'),
    (2, 20, 200, 8, 2, 20, 200, 20, 2, 't1_b', 'INACTIVE', 'A', NULL, 6, 'blue'),
    (5, 50, 500, 9, 11, 30, 300, 10, 5, 't1_c', 'ACTIVE', 'B', NULL, 7, 'green'),
    (6, 60, 600, 9, 12, 40, 400, 40, 6, 't1_d', 'ACTIVE', 'B', NULL, 8, 'yellow'),
    (11, 110, 700, 10, 13, 50, 500, 50, 11, 't1_e', 'ACTIVE', 'C', NULL, 9, 'orange'),
    (NULL, 999, NULL, NULL, NULL, NULL, NULL, 60, NULL, 't1_null', 'ACTIVE', 'C', NULL, NULL, NULL);

INSERT INTO t2 (id, k, val, ref, a, b, c, col, x, name, status, category, t1_id, amount, tag) VALUES
    (1, 10, 1, 7, 101, 1001, 10001, 1, 100, 't2_a', 'ACTIVE', 'A', 1, 10, 'tag_a'),
    (1, 10, 2, 8, 102, 1002, 10002, 2, 200, 't2_b', 'ACTIVE', 'A', 1, 20, 'tag_b'),
    (2, 20, 3, 9, 103, 1003, 10003, 3, 300, 't2_c', 'ACTIVE', 'A', 2, 30, 'tag_c'),
    (5, 50, 4, 10, 104, 1004, 10004, 4, 400, 't2_d', 'INACTIVE', 'B', 5, 40, 'tag_d'),
    (6, 60, 5, 9, 105, 1005, 10005, 5, 500, 't2_e', 'ACTIVE', 'B', 6, 50, 'tag_e'),
    (11, 110, 6, 10, 106, 1006, 10006, 6, 600, 't2_f', 'ACTIVE', 'C', 11, 60, 'tag_f'),
    (NULL, 999, NULL, NULL, 107, 1007, 10007, 7, NULL, 't2_null', 'ACTIVE', 'C', NULL, NULL, NULL);

INSERT INTO t3 (id, k, val, ref, a, b, c, col, x, name, status, category, t1_id, amount, tag) VALUES
    (1, 10, 1000, 7, 201, 2001, 20001, 1, 1, 't3_a', 'ACTIVE', 'A', NULL, 1, 'tag_a'),
    (2, 20, 2000, 8, 202, 2002, 20002, 2, 3, 't3_b', 'ACTIVE', 'A', NULL, 2, 'tag_b'),
    (5, 50, 3000, 9, 203, 2003, 20003, 3, 5, 't3_c', 'ACTIVE', 'B', NULL, 3, 'tag_c'),
    (6, 60, 4000, 10, 204, 2004, 20004, 4, 5, 't3_d', 'ACTIVE', 'B', NULL, 4, 'tag_d'),
    (11, 110, 5000, 10, 205, 2005, 20005, 5, 6, 't3_e', 'ACTIVE', 'C', NULL, 5, 'tag_e'),
    (NULL, 999, NULL, NULL, 206, 2006, 20006, 6, NULL, 't3_null', 'ACTIVE', 'C', NULL, NULL, NULL);

INSERT INTO t1_a (id, k, val, ref, a, b, c, col, x, name, status, category, t1_id, amount, tag) VALUES
    (1, 10, 101, 7, 1, 10, 100, 30, 1, 't1a_1', 'ACTIVE', 'A', NULL, 1, 'a'),
    (6, 60, 106, 9, 6, 60, 600, 20, 6, 't1a_6', 'ACTIVE', 'B', NULL, 6, 'b');

INSERT INTO t1_b (id, k, val, ref, a, b, c, col, x, name, status, category, t1_id, amount, tag) VALUES
    (2, 20, 202, 8, 2, 20, 200, 10, 2, 't1b_2', 'ACTIVE', 'A', NULL, 2, 'c'),
    (11, 110, 211, 10, 11, 110, 1100, 40, 11, 't1b_11', 'ACTIVE', 'C', NULL, 11, 'd');

INSERT INTO t2_a (id, k, val, ref, a, b, c, col, x, name, status, category, t1_id, amount, tag) VALUES
    (1, 10, 1001, 7, 1, 10, 100, 30, 1, 't2a_1', 'ACTIVE', 'A', 1, 1, 'a'),
    (5, 50, 1005, 9, 5, 50, 500, 20, 5, 't2a_5', 'ACTIVE', 'B', 5, 5, 'b');

INSERT INTO t2_b (id, k, val, ref, a, b, c, col, x, name, status, category, t1_id, amount, tag) VALUES
    (2, 20, 2002, 8, 2, 20, 200, 10, 2, 't2b_2', 'ACTIVE', 'A', 2, 2, 'c'),
    (6, 60, 2006, 10, 6, 60, 600, 40, 6, 't2b_6', 'ACTIVE', 'B', 6, 6, 'd');

INSERT INTO emp (empno, deptno, job, salary, sal, name) VALUES
    (1, 10, 'DEV', 100, 100, 'Alice'),
    (2, 10, 'DEV', 150, 150, 'Bob'),
    (3, 20, 'QA', 200, 200, 'Carol'),
    (4, 30, 'OPS', 200, 200, 'Dave'),
    (5, NULL, NULL, NULL, NULL, 'Null Emp');

INSERT INTO emp_us (deptno) VALUES
    (10),
    (10),
    (20),
    (NULL);

INSERT INTO emp_cn (deptno) VALUES
    (10),
    (30),
    (30),
    (NULL);

INSERT INTO employee (name, age, description) VALUES
    ('Alice', 31, 'senior'),
    ('Bob', 28, 'junior'),
    ('Carol', 45, 'lead');

INSERT INTO part (part_id, part_name, type) VALUES
    (1, 'rare_part_1', 'RARE'),
    (2, 'common_part_2', 'COMMON'),
    (3, 'rare_part_3', 'RARE');

INSERT INTO lineitem (order_id, part_id) VALUES
    (1001, 1),
    (1002, 2),
    (1003, 3),
    (1004, 99);
"""


def connect_to_database(retries: int = 5, wait_time: int = 3):
    conn_params = {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
    }
    print(
        "Connecting to PostgreSQL: "
        f"DB_NAME={conn_params['dbname']}, DB_USER={conn_params['user']}, "
        f"DB_HOST={conn_params['host']}, DB_PORT={conn_params['port']}"
    )

    for attempt in range(retries):
        try:
            conn = psycopg2.connect(**conn_params)
            conn.autocommit = True
            return conn
        except psycopg2.Error as exc:
            print(f"Database connection failed (attempt {attempt + 1}/{retries}): {exc}")
            if attempt < retries - 1:
                time.sleep(wait_time)
    return None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_examples(examples_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(examples_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for category, payload in data.items():
        for index, example in enumerate(payload.get("examples", []), start=1):
            rows.append(
                {
                    "id": f"{category}.{index:03d}.{example.get('id', '')}",
                    "category": category,
                    "rule_id": example.get("id", ""),
                    "original_query": example.get("original_query", ""),
                    "rewritten_query": example.get("rewritten_query", ""),
                    "rule_description": example.get("rule_description", ""),
                }
            )
    return rows


def setup_fixture_schema(conn, schema_name: str) -> None:
    with conn.cursor() as cursor:
        cursor.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema_name)))
        cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name)))
        cursor.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema_name)))
        cursor.execute(FIXTURE_SQL)


def set_search_path(conn, schema_name: str) -> None:
    with conn.cursor() as cursor:
        cursor.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema_name)))


def normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, Decimal):
        return ("num", format(value.normalize(), "f"))
    if isinstance(value, int):
        return ("num", format(Decimal(value).normalize(), "f"))
    if isinstance(value, float):
        return ("num", format(Decimal(str(value)).normalize(), "f"))
    if isinstance(value, (date, datetime)):
        return ("date", value.isoformat())
    if isinstance(value, list):
        return ("array", tuple(normalize_value(item) for item in value))
    if isinstance(value, tuple):
        return tuple(normalize_value(item) for item in value)
    return ("text", str(value))


def normalize_rows(rows: Optional[Iterable[Tuple[Any, ...]]]) -> List[Tuple[Any, ...]]:
    if rows is None:
        return []
    return [tuple(normalize_value(value) for value in row) for row in rows]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value


def execute_query(
    conn, schema_name: str, query: str, timeout: int
) -> Tuple[Optional[float], Optional[List[Tuple[Any, ...]]], List[str], Optional[str]]:
    try:
        set_search_path(conn, schema_name)
        with conn.cursor() as cursor:
            cursor.execute("SET statement_timeout = %s", (timeout * 1000,))
            start = time.time()
            cursor.execute(query)
            elapsed = time.time() - start
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            try:
                result = cursor.fetchall()
            except psycopg2.ProgrammingError:
                result = []
            finally:
                cursor.execute("SET statement_timeout = 0")
            return elapsed, result, columns, None
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        return None, None, [], str(exc)


def compare_results(original_rows: List[Tuple[Any, ...]], rewritten_rows: List[Tuple[Any, ...]]) -> Tuple[bool, Dict[str, Any]]:
    original_normalized = normalize_rows(original_rows)
    rewritten_normalized = normalize_rows(rewritten_rows)
    original_counter = collections.Counter(original_normalized)
    rewritten_counter = collections.Counter(rewritten_normalized)
    equivalent = original_counter == rewritten_counter

    details: Dict[str, Any] = {
        "original_row_count": len(original_normalized),
        "rewritten_row_count": len(rewritten_normalized),
        "ordered_equal": original_normalized == rewritten_normalized,
        "multiset_equal": equivalent,
    }
    if not equivalent:
        details["only_in_original"] = [
            {"row": row, "count": count}
            for row, count in list((original_counter - rewritten_counter).items())[:10]
        ]
        details["only_in_rewritten"] = [
            {"row": row, "count": count}
            for row, count in list((rewritten_counter - original_counter).items())[:10]
        ]
    return equivalent, details


def run_verification(args: argparse.Namespace) -> int:
    examples_path = Path(args.examples_path)
    storage_path = Path(args.storage_path)
    filtered_path = Path(args.filtered_path)
    ensure_parent(storage_path)
    ensure_parent(filtered_path)

    examples = load_examples(examples_path)
    conn = connect_to_database()
    if conn is None:
        print("Unable to connect to PostgreSQL.")
        return 2

    setup_fixture_schema(conn, args.schema)
    print(f"Loaded fixture schema: {args.schema}")
    print(f"Loaded {len(examples)} rule examples from {examples_path}")

    results: List[Dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        print(f"[{index}/{len(examples)}] {example['id']}")
        original_time, original_rows, original_columns, original_error = execute_query(
            conn, args.schema, example["original_query"], args.time_out
        )
        rewritten_time, rewritten_rows, rewritten_columns, rewritten_error = execute_query(
            conn, args.schema, example["rewritten_query"], args.time_out
        )

        equivalent = False
        comparison: Dict[str, Any] = {}
        if original_error is None and rewritten_error is None:
            equivalent, comparison = compare_results(original_rows or [], rewritten_rows or [])

        result_row = {
            **example,
            "equivalence": equivalent,
            "original_execution_time": original_time,
            "rewrite_execution_time": rewritten_time,
            "speed_up": (
                (original_time - rewritten_time) / original_time
                if original_time and rewritten_time is not None
                else None
            ),
            "times_up": (
                original_time / rewritten_time
                if original_time and rewritten_time and rewritten_time > 0
                else None
            ),
            "original_error": original_error,
            "rewritten_error": rewritten_error,
            "original_output": {
                "columns": original_columns,
                "rows": original_rows or [],
            },
            "rewritten_output": {
                "columns": rewritten_columns,
                "rows": rewritten_rows or [],
            },
            "comparison": comparison,
        }
        results.append(result_row)

        status = "PASS" if equivalent else "FAIL"
        print(f"  {status}: original={original_time}, rewritten={rewritten_time}")
        if original_error:
            print(f"  original_error: {original_error.splitlines()[0]}")
        if rewritten_error:
            print(f"  rewritten_error: {rewritten_error.splitlines()[0]}")

        storage_path.write_text(json.dumps(to_jsonable(results), ensure_ascii=False, indent=4), encoding="utf-8")
        if args.stop_on_fail and not equivalent:
            break

    total = len(results)
    equivalent_count = sum(1 for row in results if row.get("equivalence") is True)
    failed_count = total - equivalent_count
    successful_runs = sum(
        1 for row in results if row.get("original_error") is None and row.get("rewritten_error") is None
    )
    stats = {
        "total_examples": total,
        "successful_runs": successful_runs,
        "equivalent_count": equivalent_count,
        "failed_count": failed_count,
        "equivalence_rate": (equivalent_count / total) if total else 0.0,
        "schema": args.schema,
        "examples_path": str(examples_path),
    }
    output_data = [
        {"original_average": average_time(row.get("original_execution_time") for row in results)},
        {"rewritten_average": average_time(row.get("rewrite_execution_time") for row in results)},
        stats,
    ] + results
    filtered_path.write_text(json.dumps(to_jsonable(output_data), ensure_ascii=False, indent=4), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if not args.keep_schema:
        with conn.cursor() as cursor:
            cursor.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(args.schema)))
    conn.close()
    return 0 if failed_count == 0 else 1


def average_time(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if isinstance(value, (int, float))]
    if not valid:
        return None
    return sum(valid) / len(valid)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Rule_Examples.json SQL equivalence on PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python scripts/verify_rule_examples_equivalence.py
  python scripts/verify_rule_examples_equivalence.py -t 30 --schema rule_examples_equiv
  python scripts/verify_rule_examples_equivalence.py --stop-on-fail --keep-schema

Default inputs/outputs:
  examples: {DEFAULT_EXAMPLES_PATH}
  storage:  {DEFAULT_STORAGE_PATH}
  summary:  {DEFAULT_FILTERED_PATH}
        """,
    )
    parser.add_argument("-q", "--examples_path", default=str(DEFAULT_EXAMPLES_PATH), help="Path to Rule_Examples.json")
    parser.add_argument("-s", "--storage_path", default=str(DEFAULT_STORAGE_PATH), help="Detailed result JSON path")
    parser.add_argument("-f", "--filtered_path", default=str(DEFAULT_FILTERED_PATH), help="Summary result JSON path")
    parser.add_argument("-t", "--time_out", type=int, default=30, help="Query timeout in seconds")
    parser.add_argument("--schema", default="rule_examples_equiv", help="Temporary PostgreSQL schema name")
    parser.add_argument("--keep-schema", action="store_true", help="Keep the fixture schema after the run")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop after the first failed equivalence check")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(run_verification(parse_arguments()))
