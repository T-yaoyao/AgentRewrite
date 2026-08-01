import psycopg2
import time
import subprocess
import json
import numpy as np
import statistics
import sys
import os
from pathlib import Path
import collections
import re

# Setup project paths first
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.path_config import PROJECT_ROOT, setup_python_path, load_project_env
setup_python_path()
load_project_env()

from decimal import Decimal
from datetime import date
import argparse

# ============================================================================
# IMPORTANT: Database Restart Warning
# ============================================================================
# This evaluation script uses `systemctl restart postgresql` (or similar) to 
# restart the PostgreSQL database between query executions to ensure fair 
# comparison by clearing database caches.
#
# If you don't have permission to restart PostgreSQL (e.g., shared database, 
# cloud database, or restricted environment), use the --no_restart flag:
#   python evaluation.py ... --no_restart
#
# The --no_restart mode will:
#   - Skip database restart operations
#   - Run each query 5 times instead of 3
#   - Remove the highest and lowest execution times
# ============================================================================

# IDs whose original SQL should be treated as timeout (skip actual execution)
# Keyed by DB_NAME from environment.
SKIP_ORIGINAL_BY_DB = {
    "tpch": {"46", "47", "48", "55", "56", "57"},
    "dsb": {"55", "56", "57", "79", "109", "110", "111", "136", "137", "138"},
    "calcite": {"50", "53", "56"},
}


class Evaluation():
    def __init__(self, evaluation_queries_path, result_storage_path, filtered_path, timeout=300, no_restart=False):
        self.evaluation_queries_path = evaluation_queries_path
        self.result_storage_path = result_storage_path
        self.filtered_path = filtered_path
        self.timeout = timeout
        self.no_restart = no_restart
        self.iteration_count = 5 if no_restart else 3

    @staticmethod
    def _normalize_id(v):
        if v is None:
            return ""
        return str(v).strip()

    @staticmethod
    def _normalize_sql_for_compare(sql_text):
        if sql_text is None:
            return ""
        s = str(sql_text)
        s = s.replace("\\n", " ").replace("\n", " ")
        s = re.sub(r"\s+", "", s)
        return s

    @staticmethod
    def _coerce_sql_text(sql_value):
        if isinstance(sql_value, str):
            return sql_value
        if isinstance(sql_value, dict):
            for key in ("rewritten_query", "rewritten_sql", "original_query", "original_sql", "query", "sql"):
                v = sql_value.get(key)
                if isinstance(v, str) and v.strip():
                    return v
            tpch_block = sql_value.get("tpch")
            if isinstance(tpch_block, list):
                for item in tpch_block:
                    extracted = Evaluation._coerce_sql_text(item)
                    if isinstance(extracted, str) and extracted.strip():
                        return extracted
            for v in sql_value.values():
                extracted = Evaluation._coerce_sql_text(v)
                if isinstance(extracted, str) and extracted.strip():
                    return extracted
            return json.dumps(sql_value, ensure_ascii=False)
        if isinstance(sql_value, list):
            for item in sql_value:
                extracted = Evaluation._coerce_sql_text(item)
                if isinstance(extracted, str) and extracted.strip():
                    return extracted
            return ""
        if sql_value is None:
            return ""
        return str(sql_value)

    def dedupe_results_by_id(self, rows):
        if not isinstance(rows, list):
            return []
        latest = {}
        no_id_rows = []
        order = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            rid = self._normalize_id(row.get("id", ""))
            if not rid:
                no_id_rows.append(row)
                continue
            if rid not in latest:
                order.append(rid)
            latest[rid] = row
        deduped = [latest[rid] for rid in order]
        deduped.extend(no_id_rows)
        return deduped

    def _should_skip_original(self, query_id) -> bool:
        db_name = (os.getenv("DB_NAME") or "").strip().lower()
        if not db_name:
            return False
        skip_ids = SKIP_ORIGINAL_BY_DB.get(db_name, set())
        return self._normalize_id(query_id) in skip_ids

    def connect_to_database(self, retries=5, wait_time=5):
        db_name = os.getenv("DB_NAME")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")

        print(f"Connecting to database: DB_NAME={db_name}, DB_USER={db_user}, DB_HOST={db_host}, DB_PORT={db_port}")

        conn_params = {
            'dbname': db_name,
            'user': db_user,
            'password': db_password,
            'host': db_host,
            'port': db_port
        }

        for attempt in range(retries):
            try:
                conn = psycopg2.connect(**conn_params)
                print("✅ Database connection successful")
                return conn
            except psycopg2.Error as e:
                print(f"❌ Database connection failed (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Exiting.")
                    return None

    def convert_to_serializable(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_to_serializable(item) for item in obj)
        elif isinstance(obj, set):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        return obj

    def restart_postgresql(self):
        if self.no_restart:
            print("⚠️  Database restart skipped (--no_restart mode)")
            return

        print("🔄 Restarting PostgreSQL to clear caches...")
        commands = [
            ["systemctl", "restart", "postgresql"],
            ["service", "postgresql", "restart"],
            ["brew", "services", "restart", "postgresql"],
        ]

        for cmd in commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print(f"✅ PostgreSQL restarted using: {' '.join(cmd)}")
                    time.sleep(3)
                    return
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

        print("⚠️  Could not restart PostgreSQL. Continuing without restart.")

    def execute_query(self, conn, cursor, query, timeout, return_flag=False):
        try:
            cursor.execute(f"SET statement_timeout = {timeout * 1000};")
            start_time = time.time()
            cursor.execute(query)
            end_time = time.time()

            execution_result = None
            if return_flag:
                try:
                    execution_result = cursor.fetchall()
                except psycopg2.ProgrammingError:
                    pass

            return (end_time - start_time, execution_result) if return_flag else end_time - start_time

        except psycopg2.OperationalError as e:
            try:
                conn.rollback()
            except:
                pass
            if "canceling statement due to statement timeout" in str(e):
                print(f"⏱️ Query exceeded {timeout}s and was terminated")
                return (-2, None) if return_flag else -2
            print(f"Operational error: {e}")
            return (None, None) if return_flag else None

        except psycopg2.Error as e:
            try:
                conn.rollback()
            except:
                pass
            print(f"Query error: {e}")
            return (None, None) if return_flag else None

        except Exception as e:
            try:
                conn.rollback()
            except:
                pass
            raise e

        finally:
            try:
                cursor.execute("SET statement_timeout = 0;")
            except:
                pass

    def _calculate_average_time(self, times_list):
        if not times_list:
            return 0.0
        if self.no_restart and len(times_list) >= 3:
            sorted_times = sorted(times_list)
            trimmed = sorted_times[1:-1]
            print(f"   📊 No-restart: removed min={sorted_times[0]:.4f}s, max={sorted_times[-1]:.4f}s")
            print(f"   📊 Averaging {len(trimmed)} runs: {sum(trimmed)/len(trimmed):.4f}s")
            return sum(trimmed) / len(trimmed)
        return sum(times_list) / len(times_list)

    def compare_rewritten(self, original_query, rewritten_query, iteration=None, query_id=None):
        iteration = iteration or self.iteration_count

        original_times = []
        rewritten_times = []
        original_result = None
        rewritten_result = None

        print(f"📊 Running {iteration} iterations" + (" (no-restart mode)" if self.no_restart else ""))

        # ---------- Original Query ----------
        if self._should_skip_original(query_id):
            print(f"⏭️  Skipping original SQL execution for query_id={query_id}")
            total_original_time = None
            original_result = None
        else:
            conn = self.connect_to_database()
            if conn is None:
                return None, None, None, None, None, None
            cursor = conn.cursor()

            for i in range(iteration + 1):
                if i == 0:
                    print("🔹 Warm-up run (original)")
                    t, original_result = self.execute_query(
                        conn, cursor, original_query, self.timeout, return_flag=True
                    )
                    print(f"Warm-up original: {t}")
                    if t == -2:
                        print("⏱️ Original query timed out (>300s), capped to 300s")
                        total_original_time = self.timeout
                        original_result = None
                        break
                    continue

                t = self.execute_query(conn, cursor, original_query, self.timeout)
                print(f"Original run {i}: {t}")
                if t and t > 0:
                    original_times.append(t)

            if original_times:
                total_original_time = self._calculate_average_time(original_times)
            elif 'total_original_time' not in locals():
                total_original_time = None

            cursor.close()
            conn.close()

        # ---------- Restart before rewritten ----------
        self.restart_postgresql()

        # ---------- Rewritten Query ----------
        conn = self.connect_to_database()
        if conn is None:
            return total_original_time, None, None, None, original_result, None
        cursor = conn.cursor()

        for i in range(iteration + 1):
            if i == 0:
                print("🔹 Warm-up run (rewritten)")
                t, rewritten_result = self.execute_query(
                    conn, cursor, rewritten_query, self.timeout, return_flag=True
                )
                print(f"Warm-up rewritten: {t}")
                if t == -2:
                    print("⏱️ Rewritten query timed out (>300s), capped to 300s")
                    total_rewrite_time = self.timeout
                    rewritten_result = None
                    break
                continue

            t = self.execute_query(conn, cursor, rewritten_query, self.timeout)
            print(f"Rewritten run {i}: {t}")
            if t and t > 0:
                rewritten_times.append(t)

        if rewritten_times:
            total_rewrite_time = self._calculate_average_time(rewritten_times)
        elif 'total_rewrite_time' not in locals():
            total_rewrite_time = None

        cursor.close()
        conn.close()

        # ---------- Speed-up calculation ----------
        speed_up = None
        times_up = None
        if total_original_time not in (None, -1) and total_rewrite_time not in (None, -1):
            speed_up = (total_original_time - total_rewrite_time) / total_original_time
            times_up = total_original_time / total_rewrite_time
            print(f"🚀 speed_up={speed_up:.4f}, times_up={times_up:.2f}x")

        return (
            total_original_time,
            total_rewrite_time,
            speed_up,
            times_up,
            original_result,
            rewritten_result,
        )

    def evaluate(self):
        with open(self.evaluation_queries_path, 'r') as f:
            data = json.load(f)

        existing_results = []
        if os.path.exists(self.result_storage_path):
            with open(self.result_storage_path, 'r') as f:
                try:
                    existing_results = json.load(f)
                    if isinstance(existing_results, dict):
                        existing_results = [existing_results]
                    elif not isinstance(existing_results, list):
                        existing_results = []
                except json.JSONDecodeError:
                    existing_results = []

        for query_pair in data:
            query_id = query_pair.get('id', '')
            original_query = query_pair.get('original_sql', '')
            rewritten_query = query_pair.get('rewritten_sql', '')

            if not original_query or not rewritten_query:
                continue

            print(f"\n=== Evaluating query_id={query_id} ===")

            total_original_time, total_rewrite_time, speed_up, times_up, original_result, rewritten_result = \
                self.compare_rewritten(original_query, rewritten_query, iteration=self.iteration_count, query_id=query_id)

            skipped_original = self._should_skip_original(query_id)

            if skipped_original:
                equivalence = None
            elif original_result is None or rewritten_result is None:
                equivalence = None
            else:
                equivalence = (
                    collections.Counter(original_result) ==
                    collections.Counter(rewritten_result)
                )

            result_data = {
                "id": query_id,
                "equivalence": equivalence,
                "original_query": original_query,
                "rewritten_query": rewritten_query,
                "original_execution_time": total_original_time,
                "rewrite_execution_time": total_rewrite_time,
                "speed_up": speed_up,
                "times_up": times_up
            }

            existing_results.append(result_data)

        existing_results = sorted(existing_results, key=lambda x: int(x.get("id", 0)))
        with open(self.result_storage_path, 'w') as f:
            json.dump(existing_results, f, indent=4)

        print(f"\n✅ Results written to {self.result_storage_path}")

    def cal(self, equivalence_threshold=0.01, min_abs_time_improvement=0.1):
        with open(self.result_storage_path, "r") as f:
            data = json.load(f)

        data = self.dedupe_results_by_id(data if isinstance(data, list) else [])

        valid_original = []
        valid_rewrite = []
        all_results = []

        for info in data:
            if not isinstance(info, dict):
                continue

            ori_t = info.get("original_execution_time")
            rew_t = info.get("rewrite_execution_time")

            all_results.append(info)

            if isinstance(ori_t, (int, float)) and isinstance(rew_t, (int, float)):
                valid_original.append(ori_t)
                valid_rewrite.append(rew_t)

        if not valid_original:
            print("⚠️ No valid timing data available.")
            return

        ori_stats = {
            "original_average": statistics.mean(valid_original),
            "original_median": statistics.median(valid_original),
            "original_75th_percentile": np.percentile(valid_original, 75),
            "original_95th_percentile": np.percentile(valid_original, 95),
        }

        rew_stats = {
            "rewritten_average": statistics.mean(valid_rewrite),
            "rewritten_median": statistics.median(valid_rewrite),
            "rewritten_75th_percentile": np.percentile(valid_rewrite, 75),
            "rewritten_95th_percentile": np.percentile(valid_rewrite, 95),
        }

        total = len(all_results)
        equiv = sum(1 for x in all_results if x.get("equivalence") is True)
        improved = sum(
            1 for x in all_results
            if isinstance(x.get("original_execution_time"), (int, float)) and
               isinstance(x.get("rewrite_execution_time"), (int, float)) and
               (x["original_execution_time"] - x["rewrite_execution_time"]) > min_abs_time_improvement
        )

        summary = {
            "total_queries": total,
            "equivalent_count": equiv,
            "equivalence_rate": equiv / total if total else 0.0,
            "improved_count": improved,
            "improvement_rate": improved / total if total else 0.0,
        }

        with open(self.filtered_path, "w") as f:
            json.dump([ori_stats, rew_stats, summary] + all_results, f, indent=4)

        print("✅ Metrics calculation completed.")


# ============================================================================
# Helper functions (unchanged)
# ============================================================================

def ensure_file_exists(path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump([], f)


def extract_query_entries(raw_data):
    entries = []

    def _walk(node):
        if isinstance(node, dict):
            if (
                node.get("original_query")
                or node.get("query")
                or node.get("original_sql")
            ):
                entries.append(node)
            return
        if isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(raw_data)
    return entries


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="QUITE Evaluation Script - Evaluate query rewrite performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation.py -q queries.json -s results.json -f filtered.json -t 300
  python evaluation.py -q queries.json -s results.json -f filtered.json -t 300 --no_restart
        """
    )
    parser.add_argument("-q", "--queries_path", type=str, required=True)
    parser.add_argument("-s", "--storage_path", type=str, required=True)
    parser.add_argument("-f", "--filtered_path", type=str, required=True)
    parser.add_argument("-t", "--time_out", type=int, required=True)
    parser.add_argument("--no_restart", action="store_true", default=False)
    parser.add_argument("--equivalence_threshold", type=float, default=0.01)
    parser.add_argument("--min_abs_time_improvement", type=float, default=0.1)
    parser.add_argument("--time_only", action="store_true", default=False)
    parser.add_argument("--watch_queries", action="store_true", default=False)
    parser.add_argument("--watch_interval", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    ensure_file_exists(args.storage_path)
    ensure_file_exists(args.filtered_path)

    model = Evaluation(
        args.queries_path,
        args.storage_path,
        args.filtered_path,
        args.time_out,
        no_restart=args.no_restart,
    )

    # Resume logic
    completed_ids = set()
    result = []
    if os.path.exists(args.storage_path):
        try:
            with open(args.storage_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                existing = [existing]
            result = model.dedupe_results_by_id(existing if isinstance(existing, list) else [])
            for row in result:
                if isinstance(row, dict):
                    rid = model._normalize_id(row.get("id", ""))
                    if rid:
                        completed_ids.add(rid)
            if completed_ids:
                print(f"🔁 Resume: loaded {len(completed_ids)} completed IDs")
        except Exception as e:
            print(f"⚠️ Failed to load existing results: {e}")
            result = []

    stats = {
        "iteration": 0,
        "equiv_number": 0,
        "sucess_run_number": 0,
    }

    def load_query_entries():
        with open(args.queries_path, 'r') as f:
            return extract_query_entries(json.loads(f.read()))

    def evaluate_pending_once(data):
        processed_now = 0
        for query_info in data:
            if not isinstance(query_info, dict):
                continue
            query_id = query_info.get("id", "")
            query_id_str = model._normalize_id(query_id)
            if query_id_str in completed_ids:
                continue

            stats["iteration"] += 1
            processed_now += 1

            print(f"\n=== Iteration {stats['iteration']} | query_id={query_id} ===")

            original_query = model._coerce_sql_text(
                query_info.get("original_query") or query_info.get("query")
            )
            rewritten_query = model._coerce_sql_text(
                query_info.get("rewritten_query", query_info.get("rewritten_sql", ""))
            )

            ori_t, rew_t, speed_up, times_up, ori_res, rew_res = model.compare_rewritten(
                original_query, rewritten_query, query_id=query_id
            )

            skipped_original = model._should_skip_original(query_id)

            if skipped_original:
                equivalence = None
            elif ori_res is None or rew_res is None:
                equivalence = None
            else:
                equivalence = (
                    collections.Counter(ori_res) == collections.Counter(rew_res)
                )

            if equivalence is True:
                stats["equiv_number"] += 1
            if ori_t is not None and rew_t is not None:
                stats["sucess_run_number"] += 1

            result_data = {
                "id": query_id,
                "equivalence": equivalence,
                "original_query": original_query,
                "rewritten_query": rewritten_query,
                "original_execution_time": ori_t,
                "rewrite_execution_time": rew_t,
                "speed_up": speed_up,
                "times_up": times_up
            }

            result.append(result_data)
            if query_id_str:
                completed_ids.add(query_id_str)

            deduped = model.dedupe_results_by_id(result)
            with open(args.storage_path, 'w') as f:
                json.dump([model.convert_to_serializable(x) for x in deduped], f, indent=4)
            result[:] = deduped

            model.cal(
                equivalence_threshold=args.equivalence_threshold,
                min_abs_time_improvement=args.min_abs_time_improvement,
            )

        return processed_now

    data = load_query_entries()
    if not data:
        raise ValueError("No valid query entries found in queries_path")

    print("✅ Queries loaded successfully")

    if args.watch_queries:
        print(f"👀 Watch mode enabled (interval={args.watch_interval}s)")
        while True:
            try:
                data = load_query_entries()
                if evaluate_pending_once(data) == 0:
                    time.sleep(args.watch_interval)
            except KeyboardInterrupt:
                print("\n🛑 Watch mode stopped by user")
                break
    else:
        evaluate_pending_once(data)

    result = model.dedupe_results_by_id(result)
    with open(args.storage_path, 'w') as f:
        json.dump([model.convert_to_serializable(x) for x in result], f, indent=4)

    print(f"\n✅ Experiment completed")
    print(f"Equivalent queries: {stats['equiv_number']}")
    print(f"Successful runs: {stats['sucess_run_number']}/{stats['iteration']}")

    model.cal(
        equivalence_threshold=args.equivalence_threshold,
        min_abs_time_improvement=args.min_abs_time_improvement,
    )