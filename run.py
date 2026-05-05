
import os
import sys
import json
import asyncio
import time
import argparse
from pathlib import Path

# Setup project paths first
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.path_config import setup_python_path, load_project_env
setup_python_path()
load_project_env()

from tqdm import tqdm
from src.utils.data_distribution import get_statistics_list, get_available_databases
from src.utils.get_data_statistics import get_data_statistics, get_index_info, format_index_info_for_prompt
from src.Rewrite_Middleware.middleware import DBMS, DBMS_Syntax_Tool
from src.utils.agent_template import MessageContent, Message, MemoryWindow, MessageQueue
from src.Query_Rewriter.langgraph_rewriter import LangGraphQueryRewriter
from src.utils.llm_client import GPT

COST_EXPLOSION_PCT_GUARD = 1000.0

def parse_arguments():
    """Parse parameters from command line or use default values"""
    parser = argparse.ArgumentParser(description="QUITE: Query Rewrite System (LLM agents + LangGraph)")
    
    # basic configuration
    parser.add_argument("--input_path", type=str, 
                       default=None,
                       help="Input JSON file path containing queries")
    parser.add_argument("--output_dir", type=str, 
                       default="/root/syy/QUITE/output",
                       help="Output directory for results")
    parser.add_argument("--schema_file", type=str, 
                       default=None,
                       help="Path to the schema file for the database")
    parser.add_argument("--max_iterations", type=int, default=2,
                       help="Maximum iteration loops for query rewriting")

    
    # Query Rewriter configuration
    parser.add_argument("--enable_rewriter", action="store_true", default=False,
                       help="Enable query rewriter")
    parser.add_argument("--save_rewriter_logs", action="store_true", default=False,
                       help="Save rewriter terminal logs to txt files")
    parser.add_argument("--rewriter_batch_size", type=int, default=3,
                       help="Batch size for rewriter (default: 3, forced to 1 if saving logs)")
    
    return parser.parse_args()



def setup_directories(output_dir: str, enable_rewriter: bool):
    """Set up output directories for the query rewriter."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    directories = {}
    
    if enable_rewriter:
        # QueryRewriter tmp 
        rewriter_temp_dir = output_path / "rewriter_temp"
        rewriter_temp_dir.mkdir(parents=True, exist_ok=True)
        directories['rewriter_temp'] = rewriter_temp_dir
    
    directories['output'] = output_path
    return directories

async def run_query_rewriter(args, directories, dbms, data_statistics, schema_file, index_info=""):
    """LLM-based Query Rewriter (LangGraph 管线)"""
    print("\n" + "="*60)
    print("🔄 Starting Query Rewriter")
    print("="*60)
    
    mq = MessageQueue(window_size=8)
    rewriter = LangGraphQueryRewriter(
        mq, dbms, data_statistics, schema_file, args.max_iterations, index_info
    )
    
    # laod input data
    with open(args.input_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    count = 0
    all_results = []  # Store all results in memory
    result_index = {}  # id -> index in all_results (upsert by id)

    # Load existing results if file exists (append mode)
    temp_dir = directories['rewriter_temp']
    final_output_file = directories['output'] / "rewritten_queries.json"
    if final_output_file.exists():
        try:
            with open(final_output_file, "r", encoding='utf-8') as f:
                existing_results = json.load(f)
                if isinstance(existing_results, list):
                    all_results = existing_results
                    for row in all_results:
                        if isinstance(row, dict):
                            row.pop("agent_trace", None)
                    result_index = {
                        str(row.get("id")): idx
                        for idx, row in enumerate(all_results)
                        if isinstance(row, dict) and row.get("id") is not None
                    }
                    count = len(all_results)
                    print(f"📂 Loaded {count} existing results from {final_output_file}")
        except (json.JSONDecodeError, FileNotFoundError):
            print("⚠️ Could not load existing results, starting fresh")
            all_results = []

    print(f"📊 Processing {len(data)} queries (writing results individually)")
    print(f"📁 Output file: {final_output_file}")
    print(f"📝 Save logs: {'Yes' if args.save_rewriter_logs else 'No'}")
    print(f"📈 Starting from query #{count + 1}")
    print()  

    with tqdm(total=len(data),
              initial=count,
              desc="🔄 Processing Queries",
              position=0,
              leave=True,
              dynamic_ncols=True,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        
        for item in data:
            # update progress bar description
            pbar.set_description(f"🔄 Processing Query {count + 1}/{len(data)} (ID: {item.get('id', 'N/A')})")
            
            initial_sql = item["query"]

            start_time = time.time()

            # Record LLM cost before this query
            usage_before = GPT.get_global_usage()
            cost_before = usage_before.get("total_cost_rmb", 0.0)

            try:
                rewritten_sql = await rewriter.run(initial_sql)
                end_time = time.time()
                rewrite_time = end_time - start_time
                success = True
            except Exception as e:
                print(f"❌ Error processing query {item.get('id', 'N/A')}: {str(e)}")
                import traceback
                traceback.print_exc()

                # Create a fallback result for failed queries
                rewritten_sql = {
                    "tpch": [{
                        "rewritten_query": initial_sql,  # fallback to original
                        "original_costs": 0,
                        "rewrite_costs": 0,
                        "costs_reduction_rate": 0,
                        "rewrite_rules": []
                    }]
                }
                end_time = time.time()
                rewrite_time = end_time - start_time
                success = False

            # LLM cost consumed by this query (all agents)
            usage_after = GPT.get_global_usage()
            cost_after = usage_after.get("total_cost_rmb", 0.0)
            query_llm_cost = max(0.0, cost_after - cost_before)
            query_llm_cost = round(query_llm_cost, 6)

            tpch_entry = {}
            if success and isinstance(rewritten_sql, dict):
                tl = rewritten_sql.get("tpch")
                if isinstance(tl, list) and tl:
                    tpch_entry = tl[0] if isinstance(tl[0], dict) else {}
            suggestion = rewritten_sql.get("rewrite_suggestion", []) if success and isinstance(rewritten_sql, dict) else []
            tmp = {
                "id": item["id"],
                "original_query": item["query"],
                "rewritten_query": tpch_entry.get("rewritten_query", initial_sql),
                "original_costs": tpch_entry.get("original_costs", 0),
                "rewrite_costs": tpch_entry.get("rewrite_costs", 0),
                "costs_reduction_rate": tpch_entry.get("costs_reduction_rate", 0),
                "rewrite_rules": tpch_entry.get("rewrite_rules"),
                "time_cost": rewrite_time,
                "llm_costs": query_llm_cost,
                "rewrite_suggestion": suggestion if success else "Error occurred during processing",
            }

            # Write-time guard: if cost explodes beyond threshold, fallback to original SQL in output JSON.
            try:
                oc = float(tmp.get("original_costs") or 0.0)
                rc = float(tmp.get("rewrite_costs") or 0.0)
                cost_increase_pct = ((rc - oc) / oc * 100.0) if oc > 0 else 0.0
            except (TypeError, ValueError):
                oc, rc, cost_increase_pct = 0.0, 0.0, 0.0
            if oc > 0 and cost_increase_pct > COST_EXPLOSION_PCT_GUARD:
                print(
                    f"🛡️ 写入防护触发：query {item['id']} cost 增幅 {cost_increase_pct:.1f}% "
                    f"(> {COST_EXPLOSION_PCT_GUARD:.0f}%)，回退写入 original SQL。"
                )
                tmp["rewritten_query"] = initial_sql
                tmp["rewrite_costs"] = oc
                tmp["costs_reduction_rate"] = 0
                tmp["rewrite_rules"] = None
                if isinstance(tmp.get("rewrite_suggestion"), list):
                    tmp["rewrite_suggestion"].append(
                        {
                            "group": "保护机制",
                            "produced_suggestion": (
                                f"写入防护触发：重写 cost 增幅 {cost_increase_pct:.1f}% 超过 "
                                f"{COST_EXPLOSION_PCT_GUARD:.0f}%，输出已回退到原始 SQL。"
                            ),
                        }
                    )
            # Upsert by id: overwrite existing entry instead of appending duplicates
            item_id = str(tmp.get("id"))
            existing_idx = result_index.get(item_id)
            if existing_idx is not None:
                all_results[existing_idx] = tmp
            else:
                all_results.append(tmp)
                result_index[item_id] = len(all_results) - 1
            count = len(all_results)

            # Write all results to the final output file
            with open(final_output_file, "w", encoding='utf-8') as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)

            # If saving logs, write individual log files
            if args.save_rewriter_logs:
                log_file_path = temp_dir / f"query_{item['id']}.txt"
                with open(log_file_path, "w", encoding='utf-8') as f:
                    f.write(f"Query ID: {item['id']}\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Original: {tmp['original_query']}\n")
                    f.write(f"Rewritten: {tmp['rewritten_query']}\n")
                    f.write(f"Time Cost: {tmp['time_cost']:.2f}s\n")
                    f.write(f"Suggestion: {tmp['rewrite_suggestion']}\n")
                    f.write((rewritten_sql.get("terminal_output", "") if isinstance(rewritten_sql, dict) else "") or "")
                    f.write("\n" + "-"*40 + "\n\n")

            pbar.update(1)
            pbar.set_postfix({
                'Time': f'{rewrite_time:.2f}s',
                'Status': 'Success' if success else 'Error',
                'Total': count
            })

        # update progress bar when done
        pbar.set_description("🔄 Query Rewriter Completed")
        pbar.set_postfix({'Status': f'Processed {count} queries'})

    print(f"\n✅ Query Rewriter completed! Processed {count} queries")
    print(f"📁 Final output: {final_output_file}")
    return final_output_file

async def main():
    print("🚀 QUITE System Starting...")
    
    try:
        args = parse_arguments()
    except SystemExit:
        # if no arguments are provided, use default values
        print("No command line arguments provided. Using default configuration. Process ends!")
        return
    
    print(f"📂 Input: {args.input_path}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🔄 Rewriter: {'Enabled' if args.enable_rewriter else 'Disabled'}")
    
    if args.save_rewriter_logs:
        print(f"📝 Rewriter logs: Enabled (batch size forced to 1)")
    else:
        print(f"📝 Rewriter logs: Disabled (batch size: {args.rewriter_batch_size})")
    
    # set up output directories
    directories = setup_directories(args.output_dir, args.enable_rewriter)
    print(f"\n📁 Directory structure created:")
    for key, path in directories.items():
        print(f"   {key}: {path}")

    ###########################################################
    # database and schema setup
    ###########################################################
    
    # obtain database name and statistics
    dbms = DBMS()
    DB_NAME = dbms.db_name
    data_statistics = None
    if DB_NAME in get_available_databases():
        data_statistics = get_statistics_list(DB_NAME)
    else:
        data_statistics = get_data_statistics()
    
    print(f"\n📊 Database {DB_NAME} statistics loaded")

    index_info = ""
    try:
        index_info = format_index_info_for_prompt(get_index_info(dbms))
        print("📋 Index info loaded for rewriter agents")
    except Exception as e:
        print(f"⚠️ Could not load index info: {e}")
    
    schema_file = args.schema_file
    with open(schema_file, 'r') as f:
        schema_content = f.read()
        if not schema_content.strip():
            raise ValueError(f"Schema file {schema_file} is empty or not found.")
        print(f"📋 Schema content loaded from {schema_file}")

    ###########################################################
    # QUITE System Execution
    ###########################################################
    rewriter_output_file = None
    
    # run Query Rewriter
    if args.enable_rewriter:
        rewriter_output_file = await run_query_rewriter(
            args, directories, dbms, data_statistics, schema_file, index_info
        )

    ###########################################################
    # summary and output
    ###########################################################
    
    print("\n" + "="*60)
    print("🎉 QUITE System Completed!")
    print("="*60)
    
    if rewriter_output_file:
        print(f"📝 Query Rewriter Output: {rewriter_output_file}")
    
    print(f"📁 Output Directory: {directories['output']}")
    
    # list all output files
    output_files = list(directories['output'].glob("*.json"))
    if output_files:
        print(f"\n📊 Final output files:")
        for file in output_files:
            size = file.stat().st_size
            print(f"   {file.name}: {size:,} bytes")
    print("\n✨ Processing completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())