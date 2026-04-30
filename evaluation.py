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
SKIP_ORIGINAL_FIXED_TIMEOUT = 300.0

class Evaluation():
    def __init__(self, evaluation_queries_path, result_storage_path, filtered_path, timeout=300, no_restart=False):
        self.evaluation_queries_path = evaluation_queries_path
        self.result_storage_path = result_storage_path
        self.filtered_path = filtered_path
        self.timeout = timeout
        self.no_restart = no_restart
        
        # Set iteration count based on restart mode
        # no_restart mode: 5 iterations, remove max/min, average remaining 3
        # normal mode: 3 iterations, average all
        self.iteration_count = 5 if no_restart else 3

    @staticmethod
    def _normalize_id(v):
        """Normalize query id for resume/dedup."""
        if v is None:
            return ""
        return str(v).strip()

    @staticmethod
    def _normalize_sql_for_compare(sql_text):
        """
        Normalize SQL for semantic-equality-like text comparison in metrics:
        - Replace literal \\n and real newlines with spaces
        - Remove all whitespace
        """
        if sql_text is None:
            return ""
        s = str(sql_text)
        s = s.replace("\\n", " ").replace("\n", " ")
        s = re.sub(r"\s+", "", s)
        return s

    @staticmethod
    def _coerce_sql_text(sql_value):
        """
        Convert possible structured SQL payloads into executable SQL text.
        Accepts plain strings or dict-like entries that contain rewritten SQL fields.
        """
        if isinstance(sql_value, str):
            return sql_value
        if isinstance(sql_value, dict):
            # Direct field lookup first.
            for key in ("rewritten_query", "rewritten_sql", "original_query", "original_sql", "query", "sql"):
                v = sql_value.get(key)
                if isinstance(v, str) and v.strip():
                    return v
            # Common QUITE output wrapper: {"tpch":[{"rewritten_query":"..."}]}
            tpch_block = sql_value.get("tpch")
            if isinstance(tpch_block, list):
                for item in tpch_block:
                    extracted = Evaluation._coerce_sql_text(item)
                    if isinstance(extracted, str) and extracted.strip():
                        return extracted
            # Recursive fallback for nested dict/list payloads.
            for v in sql_value.values():
                extracted = Evaluation._coerce_sql_text(v)
                if isinstance(extracted, str) and extracted.strip():
                    return extracted
            # Last-resort: stable JSON text (won't execute, but avoids crashes and is debuggable)
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
        """
        Deduplicate result rows by id, keeping the latest row for each id.
        Rows without id are kept as-is.
        """
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
        """Whether original SQL execution should be skipped for this query id."""
        db_name = (os.getenv("DB_NAME") or "").strip().lower()
        if not db_name:
            return False
        skip_ids = SKIP_ORIGINAL_BY_DB.get(db_name, set())
        return self._normalize_id(query_id) in skip_ids

    def connect_to_database(self, retries=5, wait_time=5):
        db_name = os.getenv("DB_NAME")
        db_user= os.getenv("DB_USER")
        db_password= os.getenv("DB_PASSWORD")
        db_host= os.getenv("DB_HOST")
        db_port= os.getenv("DB_PORT")
        print("Connecting to database with the following parameters:")
        print(f"DB_NAME: {db_name}, DB_USER: {db_user}, DB_HOST: {db_host}, DB_PORT: {db_port}")
        conn_params = {
            'dbname': db_name,
            'user': db_user,
            'password': db_password,
            'host': db_host,
            'port': db_port
        }
        print("Connecting to database... DB_NAME: {}, DB_USER: {}, DB_HOST: {}, DB_PORT: {}".format(db_name, db_user, db_host, db_port))
        for attempt in range(retries):
            try:
                conn = psycopg2.connect(**conn_params)
                print("Database connection successful")
                return conn
            except psycopg2.Error as e:
                print(f"Database connection failed (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(wait_time)  
                else:
                    print("Max retries reached. Exiting.")
                    return None

    def convert_to_serializable(self,obj):
        """Convert non-serializable types like Decimal and date in data structures to serializable types."""
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
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        return obj
    
    def restart_postgresql(self):
        """
        Restart PostgreSQL to clear database caches.
        
        This function requires appropriate system permissions:
        - On Linux: sudo systemctl restart postgresql
        - On macOS: brew services restart postgresql
        - On Windows: net stop postgresql && net start postgresql
        
        If you don't have restart permissions, use --no_restart flag.
        """
        if self.no_restart:
            print("⚠️  Database restart skipped (--no_restart mode)")
            return
            
        print("🔄 Restarting PostgreSQL to clear caches...")
        # Try different restart commands based on the system
        commands = [
            ["systemctl", "restart", "postgresql"],      # Linux with systemd
            ["service", "postgresql", "restart"],         # Linux with init.d
            ["brew", "services", "restart", "postgresql"], # macOS with Homebrew
        ]
        
        for cmd in commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print(f"✅ PostgreSQL restarted successfully using: {' '.join(cmd)}")
                    time.sleep(3)  # wait for PostgreSQL to be fully up
                    return
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        print("⚠️  Warning: Could not restart PostgreSQL. Consider using --no_restart mode.")
        print("   Manual restart commands:")
        print("   - Linux: sudo systemctl restart postgresql")
        print("   - macOS: brew services restart postgresql")

    def execute_query(self, conn, cursor, query, timeout,return_flag=False):
        try:
            # Set PostgreSQL statement timeout
            cursor.execute(f"SET statement_timeout = {timeout * 1000};")
            start_time = time.time()
            
            # Execute the query
            cursor.execute(query)
            end_time = time.time()

            execution_result = None
            try:
                execution_result = cursor.fetchall()  # Fetch all results
            except psycopg2.ProgrammingError as e:
                pass            

            if return_flag == False:
                return end_time - start_time
            else:
                return end_time - start_time, execution_result
        
        except psycopg2.OperationalError as e:
            if "canceling statement due to statement timeout" in str(e):
                try:
                    conn.rollback()
                except:
                    pass
                print(f"Query execution exceeded {timeout} seconds and was terminated.")
                # self.restart_postgresql()
                return -2, -2
            else:
                try:
                    conn.rollback()
                except:
                    pass
                print(f"Error exe cuting query: {e}")
                # self.restart_postgresql()
                return (None, None) if return_flag else None
        
        except psycopg2.Error as e:
            try:
                conn.rollback()  # rollback transaction in case of error
            except:
                pass
            print(f"Error executing query: {e}")
            if return_flag == False:
                return -1  
            else:
                return None, None
            
        except Exception as e:
            try:
                conn.rollback()  # Rollback transaction in case of error
            except:
                pass
            raise e
        
        finally:
            try:
                cursor.execute("SET statement_timeout = 0;")
            except:
                pass



    def _calculate_average_time(self, times_list):
        """
        Calculate average execution time.
        
        In no_restart mode: Remove highest and lowest, average remaining.
        In normal mode: Average all times.
        """
        if not times_list:
            return 0.0
            
        if self.no_restart and len(times_list) >= 3:
            # Remove highest and lowest, average remaining
            sorted_times = sorted(times_list)
            trimmed_times = sorted_times[1:-1]  # Remove first (min) and last (max)
            avg_time = sum(trimmed_times) / len(trimmed_times)
            print(f"   📊 No-restart mode: Removed min({sorted_times[0]:.4f}s) and max({sorted_times[-1]:.4f}s)")
            print(f"   📊 Averaged {len(trimmed_times)} runs: {avg_time:.4f}s")
            return avg_time
        else:
            # Normal average
            return sum(times_list) / len(times_list)

    def compare_rewritten(self, original_query, rewritten_query, iteration=None, query_id=None):
        # Use instance iteration count if not specified
        if iteration is None:
            iteration = self.iteration_count
            
        conn = self.connect_to_database()
        cursor = conn.cursor()
        timeout = self.timeout
        original_times = []
        rewritten_times = []
        original_result = None
        rewritten_result = None
        speed_up = None
        times_up = None
        
        print(f"📊 Running {iteration} iterations" + (" (no-restart mode)" if self.no_restart else ""))

        if self._should_skip_original(query_id):
            print(
                f"⏭️  Skip original SQL execution for query_id={query_id}; "
                f"set original_execution_time={SKIP_ORIGINAL_FIXED_TIMEOUT}s"
            )
            total_original_time = SKIP_ORIGINAL_FIXED_TIMEOUT
            # Keep timeout marker behavior consistent with execute_query timeout branch.
            original_result = -2
        else:
            # execute original query
            for i in range(iteration + 1):
                ORIGINAL_TIME_OUT = False
                if conn is None:
                    return None, None, None, None, None, None  
                
                if i == 0:
                    print("start init hot database execution")
                    original_time, original_result = self.execute_query(conn, cursor, original_query, timeout, return_flag=True)
                    print(f"this is the init: original query excute time: {original_time}")
                    if original_time == -2:
                        ORIGINAL_TIME_OUT = True
                        break
                    elif original_time == -1:
                        break
                    else:
                        continue

                original_time = self.execute_query(conn, cursor, original_query, timeout)
                print(f"the {i}-th/{iteration} iteration original query excute time: {original_time}")
                if original_time and original_time > 0:
                    original_times.append(original_time)
            
            if ORIGINAL_TIME_OUT:
                total_original_time = timeout
            elif original_times:
                total_original_time = self._calculate_average_time(original_times)
            else:
                print("Original Query Execution Failed")
                return None, None, None, None, None, None 
        
        if total_original_time is not None:
            print(f"Original Query Execution Time: {total_original_time:.6f} seconds")
        else:
            print("Original Query Execution Failed")

        self.restart_postgresql()
        if conn is None:
            return None, None, None, None, None, None 
        
        conn = self.connect_to_database()
        cursor = conn.cursor()

        try:
            for i in range(iteration + 1):
                REWRITTEN_TIME_OUT = False 
                if conn is None:
                    return None, None, None, None, None, None 
                
                if i == 0:
                    print("start init hot database execution")
                    try:
                        rewritten_time, rewritten_result = self.execute_query(conn, cursor, rewritten_query, timeout, return_flag=True)
                    except Exception as e:
                        if "connection" in str(e).lower() or "cursor" in str(e).lower():
                            print(f"Connection error, retrying after restart: {e}")
                            try:
                                cursor.close()
                                conn.close()
                            except:
                                pass
                            self.restart_postgresql()
                            conn = self.connect_to_database()
                            if conn is None:
                                return None, None, None, None, None, None
                            cursor = conn.cursor()
                            rewritten_time, rewritten_result = self.execute_query(conn, cursor, rewritten_query, timeout, return_flag=True)
                        else:
                            raise e
                    print(f"this is the init: rewrite query excute time: {rewritten_time}")
                    if rewritten_time == -2:
                        REWRITTEN_TIME_OUT = True
                        break
                    elif rewritten_time == -1:
                        break
                    else:
                        continue
                rewritten_time = self.execute_query(conn, cursor, rewritten_query, timeout)
                print(f"the {i}-th/{iteration} iteration rewrite query excute time: {rewritten_time}")
                if rewritten_time and rewritten_time > 0:
                    rewritten_times.append(rewritten_time)

            if REWRITTEN_TIME_OUT:
                total_rewrite_time = timeout
            elif rewritten_times:
                total_rewrite_time = self._calculate_average_time(rewritten_times)
            else:
                print("Rewritten Query Execution Failed")
                return total_original_time, None, None, None, original_result, None
            
            if total_rewrite_time is not None:
                print(f"Rewritten Query Execution Time: {total_rewrite_time:.6f} seconds")


        except psycopg2.errors.SyntaxError as e:
            print(f"Rewritten query execution failed due to syntax error: {e}")
            return total_original_time, None, None, None, original_result, None
        except Exception as e:
            print(f"Rewritten query execution failed due to unexpected error: {e}")
            return total_original_time, None, None, None, original_result, None

        
        if total_original_time is not None and total_rewrite_time is not None:
            speed_up = (total_original_time - total_rewrite_time) / total_original_time
            times_up = total_original_time / total_rewrite_time
            print(f"speed_up : {speed_up}")
            print(f"times up: {times_up}x")
        

        return total_original_time, total_rewrite_time, speed_up, times_up, original_result, rewritten_result
            

    def evaluate(self):
        with open(self.evaluation_queries_path, 'r') as file:
            data = json.load(file)

        existing_results = []

        if os.path.exists(self.result_storage_path):
            with open(self.result_storage_path, 'r') as result_file:
                try:
                    existing_results = json.load(result_file)
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
            
            if original_query and rewritten_query:
                print(f"Running queries for pair: {original_query[:30]}... and rewritten query.")
                
                original_execution_time, rewrite_execution_time, speed_up, times_up,original_result,rewritten_result = self.compare_rewritten(
                    original_query, rewritten_query, iteration=3, query_id=query_id
                )
                skipped_original = self._should_skip_original(query_id)
                if skipped_original:
                    equivalance = True
                elif original_result != None and rewritten_result != None and original_execution_time != self.timeout and rewrite_execution_time != self.timeout:
                    original_counts = collections.Counter(original_result)
                    rewritten_counts = collections.Counter(rewritten_result)
                    equivalance = (original_counts == rewritten_counts)
                else:
                    equivalance = False
                result_data = {
                    "id": query_id,
                    "equivalence": equivalance,
                    "original_query": original_query,
                    "rewritten_query": rewritten_query,
                    "original_execution_time": original_execution_time,
                    "rewrite_execution_time": rewrite_execution_time,
                    "speed_up": speed_up,
                    "times_up": times_up              
                }

                existing_results.append(result_data)


        existing_results = sorted(existing_results, key=lambda x: int(x.get("id", 0)))  
        with open(self.result_storage_path, 'w') as result_file:
            json.dump(existing_results, result_file, indent=4)

        print(f"Experiment results appended to {self.result_storage_path}")
    
    def cal(self, equivalence_threshold=0.01, min_abs_time_improvement=0.1):
        Metric = []
        result_data = []
        all_result_data = []
        original_execution_times = []
        rewritten_execution_times = []
        with open(self.result_storage_path, "r") as file:
            data = json.load(file)
            data = self.dedupe_results_by_id(data if isinstance(data, list) else [])
            for info in data:
                if not isinstance(info, dict):
                    continue
                insert_data = {
                    "id": info.get("id"),
                    "equivalence": info.get("equivalence"),
                    "original_query": info.get("original_query"),
                    "rewritten_query": info.get("rewritten_query"),
                    "original_execution_time": info.get("original_execution_time"),
                    "rewrite_execution_time": info.get("rewrite_execution_time"),
                    "speed_up": info.get("speed_up"),
                    "times_up": info.get("times_up")
                }
                all_result_data.append(insert_data)

                original_t = info.get('original_execution_time')
                rewrite_t = info.get('rewrite_execution_time')
                if (
                    original_t != -1 and original_t is not None and
                    rewrite_t != -1 and rewrite_t is not None
                ):
                    original_execution_times.append(info['original_execution_time'])
                    rewritten_execution_times.append(info['rewrite_execution_time'])
                    result_data.append(insert_data)

        if not original_execution_times or not rewritten_execution_times:
            print("No valid rows left for metrics after filtering timeouts.")
            return

        # calculate mean
        ori_avg = statistics.mean(original_execution_times)
        print(f"The average execution time is: {ori_avg}")
        # calculate median
        ori_median = statistics.median(original_execution_times)
        print(f"The median execution time is: {ori_median}")
        # calculate 75th percentile
        ori_percentile_75 = np.percentile(original_execution_times, 75)
        print(f"The 75th percentile execution time is: {ori_percentile_75}")
        # calculate 95th percentile
        ori_percentile_95 = np.percentile(original_execution_times, 95)
        print(f"The 95th percentile execution time is: {ori_percentile_95}")

        re_avg = statistics.mean(rewritten_execution_times)
        print(f"The average execution time is: {re_avg}")
        # calculate median
        re_median = statistics.median(rewritten_execution_times)
        print(f"The median execution time is: {re_median}")
        # calculate 75th percentile
        re_percentile_75 = np.percentile(rewritten_execution_times, 75)
        print(f"The 75th percentile execution time is: {re_percentile_75}")
        # calculate 95th percentile
        re_percentile_95 = np.percentile(rewritten_execution_times, 95)
        print(f"The 95th percentile execution time is: {re_percentile_95}")
        ori_result = {
            "original_average": ori_avg,
            "original_median": ori_median,
            "original_75th_percentile": ori_percentile_75,
            "original_95th_percentile": ori_percentile_95
        }
        re_result = {
            "rewritten_average": re_avg,
            "rewritten_median": re_median,
            "rewritten_75th_percentile": re_percentile_75,
            "rewritten_95th_percentile": re_percentile_95
        }

        total_queries = len(all_result_data)
        equivalent_count = sum(1 for x in all_result_data if x.get("equivalence") is True)
        improved_count = sum(
            1 for x in all_result_data
            if (
                isinstance(x.get("original_execution_time"), (int, float))
                and isinstance(x.get("rewrite_execution_time"), (int, float))
                and (
                    x.get("original_execution_time") - x.get("rewrite_execution_time")
                    > min_abs_time_improvement
                )
                and self._normalize_sql_for_compare(x.get("original_query", "")) !=
                self._normalize_sql_for_compare(x.get("rewritten_query", ""))
            )
        )
        stats_result = {
            "total_queries": total_queries,
            "equivalent_count": equivalent_count,
            "equivalence_rate": (equivalent_count / total_queries) if total_queries else 0.0,
            "improved_count": improved_count,
            "improvement_rate": (improved_count / total_queries) if total_queries else 0.0,
            "equivalence_threshold": equivalence_threshold,
            "min_abs_time_improvement": min_abs_time_improvement,
        }

        output_data = [ori_result, re_result, stats_result] + all_result_data
        with open(self.filtered_path,"w") as file:
            json.dump(output_data, file, indent=4)
        print("Calculate Metrics Done!") # Debug line to check if the script has finished running

def ensure_file_exists(path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump([], f)  # Create an empty JSON array if the file does not exist


def extract_query_entries(raw_data):
    """
    Flatten mixed JSON payloads and keep only query dict entries.
    Supports files whose top-level list may contain nested summary blocks.
    """
    entries = []

    def _walk(node):
        if isinstance(node, dict):
            # Keep rows that look like query pairs.
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
            # Normal mode (with database restart between queries)
            python evaluation.py -q queries.json -s results.json -f filtered.json -t 300

            # No-restart mode (for environments without restart permission)
            python evaluation.py -q queries.json -s results.json -f filtered.json -t 300 --no_restart

            Note:
            The --no_restart flag is useful when you don't have permission to restart PostgreSQL.
            In this mode, queries are run 5 times, and the average is calculated after removing
            the highest and lowest times to mitigate cold-start effects.
        """
    )
    parser.add_argument("-q", "--queries_path", type=str, required=True, 
                        help="Path to the queries SQL file (JSON format)")
    parser.add_argument("-s", "--storage_path", type=str, required=True, 
                        help="Path to the storage SQL file (JSON format)")
    parser.add_argument("-f", "--filtered_path", type=str, required=True, 
                        help="Path to the filtered SQL file (JSON format)")
    parser.add_argument("-t", "--time_out", type=int, required=True, 
                        help="Query timeout in seconds")
    parser.add_argument("--no_restart", action="store_true", default=False,
                        help="Disable database restart between queries (runs 5 times, removes min/max)")
    parser.add_argument("--equivalence_threshold", type=float, default=0.01,
                        help="Threshold metadata for equivalence rate reporting (default: 0.01)")
    parser.add_argument("--min_abs_time_improvement", type=float, default=0.1,
                        help="Absolute time improvement threshold in seconds for improvement_rate counting (default: 0.1)")
    parser.add_argument("--time-only", action="store_true", default=False,
                        help="Only measure execution time, skip full rewrite and semantic check")
    parser.add_argument("--watch_queries", action="store_true", default=False,
                        help="Continuously watch queries_path and evaluate newly appended query IDs")
    parser.add_argument("--watch_interval", type=float, default=2.0,
                        help="Polling interval in seconds for --watch_queries mode (default: 2.0)")
    return parser.parse_args()     

if __name__ == "__main__":
    args = parse_arguments()
    queries_path = args.queries_path
    storage_path = args.storage_path
    filtered_path = args.filtered_path
    time_out = args.time_out
    no_restart = args.no_restart
    equivalence_threshold = args.equivalence_threshold
    min_abs_time_improvement = args.min_abs_time_improvement
    time_only = getattr(args, 'time_only', False)
    watch_queries = getattr(args, 'watch_queries', False)
    watch_interval = max(0.5, float(getattr(args, 'watch_interval', 2.0)))
    
    if no_restart:
        print("=" * 60)
        print("⚠️  NO-RESTART MODE ENABLED")
        print("=" * 60)
        print("Database will NOT be restarted between query executions.")
        print("Running 5 iterations per query, removing min/max for average.")
        print("=" * 60)

    if time_only:
        print("=" * 60)
        print("⏱️  TIME-ONLY MODE ENABLED")
        print("Only measuring execution time for dsb_test.json (skipping full rewrite & semantic check)")
        print("=" * 60)

    ensure_file_exists(storage_path)
    ensure_file_exists(filtered_path)
    model = Evaluation(queries_path, storage_path, filtered_path, time_out, no_restart=no_restart)


    # Resume + dedupe existing results
    result = []
    completed_ids = set()
    if os.path.exists(storage_path):
        try:
            with open(storage_path, "r", encoding="utf-8") as f:
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
                print(f"🔁 Resume enabled: loaded {len(completed_ids)} completed IDs from {storage_path}")
        except Exception as e:
            print(f"⚠️ Failed to load existing results, start from scratch: {e}")
            result = []
            completed_ids = set()

    stats = {
        "iteration": 0,
        "equiv_number": 0,
        "sucess_run_number": 0,
    }

    def load_query_entries():
        with open(queries_path, 'r') as file:
            json_content = file.read()
        raw_data = json.loads(json_content)
        return extract_query_entries(raw_data)

    def evaluate_pending_once(data):
        processed_now = 0
        for i, query_info in enumerate(data, start=0):
            if not isinstance(query_info, dict):
                print(f"⏭️  Skip invalid query entry at index={i}: type={type(query_info).__name__}")
                continue
            query_id = query_info.get("id", "")
            query_id_str = model._normalize_id(query_id)
            if query_id_str in completed_ids:
                continue

            stats["iteration"] += 1
            processed_now += 1
            print("this is the {}-th iteration".format(stats["iteration"]))
            print(f"the query id is {query_id}")

            # Support both "query" (in dsb_test.json) and "original_query"
            original_query = model._coerce_sql_text(query_info.get("original_query") or query_info.get("query"))
            if time_only:
                rewritten_query = original_query  # In time-only mode, we only test original for now
                print(f"Original Query (time-only): {original_query[:80]}...")
            else:
                rewritten_query = model._coerce_sql_text(
                    query_info.get("rewritten_query", query_info.get("rewritten_sql", "No rewritten query found"))
                )
                print(f"Original Query: {original_query[:80]}...")
                print(f"Rewritten Query: {rewritten_query[:80]}...")

            total_original_time,total_rewrite_time,speed_up,times_up,original_result,rewritten_result = model.compare_rewritten(
                original_query, rewritten_query, query_id=query_id
            )
            
            equivalance = False
            if total_original_time != None and total_rewrite_time != None and original_result != -1 and rewritten_result != -1:
                stats["sucess_run_number"] += 1
            
            skipped_original = model._should_skip_original(query_id)
            if skipped_original:
                equivalance = True
            elif original_result != None and rewritten_result != None and original_result != -1 and rewritten_result != -1 and total_original_time != time_out and total_rewrite_time != time_out:
                original_counts = collections.Counter(original_result)
                rewritten_counts = collections.Counter(rewritten_result)
                equivalance = (original_counts == rewritten_counts)

            if equivalance:
                stats["equiv_number"] += 1
            result_data = {
                "id": query_id,
                "equivalence": equivalance,
                "original_query": original_query,
                "rewritten_query": rewritten_query,
                "original_execution_time": total_original_time,
                "rewrite_execution_time": total_rewrite_time,
                "speed_up": speed_up,
                "times_up": times_up          
            }
            result.append(result_data)
            if query_id_str:
                completed_ids.add(query_id_str)

            # Persist after each query for true resume capability
            deduped = model.dedupe_results_by_id(result)
            with open(storage_path, 'w') as result_file:
                json.dump([model.convert_to_serializable(item) for item in deduped], result_file, indent=4)
            result[:] = deduped
            print(f"Results saved after query {query_id}")

            # Refresh metrics after each newly evaluated query.
            model.cal(
                equivalence_threshold=equivalence_threshold,
                min_abs_time_improvement=min_abs_time_improvement,
            )
        return processed_now

    data = load_query_entries()
    if not data:
        raise ValueError(
            "No valid query entries found in queries_path. "
            "Expected dict items containing original_query/query/original_sql."
        )
    print("data load sucessfully!")

    if watch_queries:
        print(f"👀 Watch mode enabled: polling every {watch_interval:.1f}s")
        while True:
            try:
                data = load_query_entries()
                processed_now = evaluate_pending_once(data)
                if processed_now == 0:
                    time.sleep(watch_interval)
            except KeyboardInterrupt:
                print("\n🛑 Watch mode stopped by user")
                break
    else:
        evaluate_pending_once(data)

    # Final flush
    result = model.dedupe_results_by_id(result)
    with open(storage_path, 'w') as result_file:
        json.dump([model.convert_to_serializable(item) for item in result], result_file, indent=4)
    print(f"Experiment results appended to {storage_path}")
    print(f"the number of equivalent query is {stats['equiv_number']}")
    print(f"the number of sucess run query is {stats['sucess_run_number']}/ {stats['iteration']}")

    model.cal(
        equivalence_threshold=equivalence_threshold,
        min_abs_time_improvement=min_abs_time_improvement,
    )

