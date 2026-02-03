
import os
import sys
import json
import asyncio
import time
import argparse
import glob
from pathlib import Path

# Setup project paths first
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.path_config import PROJECT_ROOT, setup_python_path, load_project_env
setup_python_path()
load_project_env()

from tqdm import tqdm
from src.utils.data_distribution import get_statistics_list, get_available_databases
from src.utils.get_data_statistics import get_data_statistics
from src.Rewrite_Middleware.middleware import DBMS, DBMS_Syntax_Tool
from src.utils.agent_template import MessageContent, Message, MemoryWindow, MessageQueue
from src.Query_Rewriter.finite_state_machine import QueryRewriter
from src.Hint_Recommender.injection import Hint_Recommender   
from src.utils.llm_client import GPT

def parse_arguments():
    """Parse parameters from command line or use default values"""
    parser = argparse.ArgumentParser(description="QUITE: Query Rewrite and Hint Recommendation System")
    
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
    
    # Hint Recommender configuration
    parser.add_argument("--enable_recommender", action="store_true", default=False,
                       help="Enable hint recommender")
    parser.add_argument("--recommender_batch_size", type=int, default=3,
                       help="Batch size for recommender")
    
    return parser.parse_args()



def setup_directories(output_dir: str, enable_rewriter: bool, enable_recommender: bool):
    """Set up output directories for rewriter and recommender"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    directories = {}
    
    if enable_rewriter:
        # QueryRewriter tmp 
        rewriter_temp_dir = output_path / "rewriter_temp"
        rewriter_temp_dir.mkdir(exist_ok=True)
        directories['rewriter_temp'] = rewriter_temp_dir
    
    if enable_recommender:
        # HintRecommender tmp
        recommender_temp_dir = output_path / "recommender_temp"
        recommender_temp_dir.mkdir(exist_ok=True)
        directories['recommender_temp'] = recommender_temp_dir
    
    directories['output'] = output_path
    return directories

def merge_batch_files(temp_dir: Path, output_dir: Path, final_filename: str):
    """merge all batch files into a single JSON file"""
    batch_files = sorted(glob.glob(str(temp_dir / "batch_*.json")))
    if not batch_files:
        print(f"No batch files found in {temp_dir}")
        return None
    
    merged_data = []
    for batch_file in batch_files:
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    merged_data.append(data)
        except Exception as e:
            print(f"Error reading {batch_file}: {e}")
    
    if merged_data:
        final_file_path = output_dir / final_filename
        with open(final_file_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=4, ensure_ascii=False)
        print(f"✅ Merged {len(batch_files)} batch files into {final_file_path}")
        print(f"📊 Total queries processed: {len(merged_data)}")
        return final_file_path
    return None

async def run_query_rewriter(args, directories, dbms, data_statistics, schema_file):
    """LLM-based Query Rewriter"""
    print("\n" + "="*60)
    print("🔄 Starting Query Rewriter")
    print("="*60)
    
    mq = MessageQueue(window_size=8)
    rewriter = QueryRewriter(mq, dbms, data_statistics, schema_file, args.max_iterations)
    
    # laod input data
    with open(args.input_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    count = 0
    all_results = []  # Store all results in memory

    # Load existing results if file exists (append mode)
    temp_dir = directories['rewriter_temp']
    final_output_file = directories['output'] / "rewritten_queries.json"
    if final_output_file.exists():
        try:
            with open(final_output_file, "r", encoding='utf-8') as f:
                existing_results = json.load(f)
                if isinstance(existing_results, list):
                    all_results = existing_results
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

            # Reset rewriter state for new query
            rewriter.initial_sql = initial_sql
            rewriter.current_state = "INITIAL_CHECK"  # Ensure correct initial state
            start_time = time.time()

            # Record LLM cost before this query
            usage_before = GPT.get_global_usage()
            cost_before = usage_before.get("total_cost_rmb", 0.0)

            try:
                rewritten_sql = await rewriter.run()
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

            # 在评估完成后、写入 JSON 之前，对“最终版本”的重写 SQL 再做一次语法验证
            if success:
                final_sql = None

                # 从 rewriter.run() 的结果结构中提取最终重写 SQL
                if isinstance(rewritten_sql, dict):
                    tpch_list = rewritten_sql.get("tpch")
                    if isinstance(tpch_list, list) and tpch_list:
                        first_entry = tpch_list[0]
                        if isinstance(first_entry, dict):
                            final_sql = first_entry.get("rewritten_query")

                # 仅当最终 SQL 存在且确实与原始 SQL 不同时才进行额外校验
                if final_sql and isinstance(final_sql, str) and final_sql.strip() and final_sql.strip() != initial_sql.strip():
                    try:
                        print("🧪 对最终重写 SQL 进行语法二次校验...")
                        syntax_check = await DBMS_Syntax_Tool(dbms, final_sql)
                        is_valid = syntax_check.get("flag", syntax_check.get("valid", True))

                        if not is_valid:
                            error_info = syntax_check.get("error", "Unknown error")
                            print("⚠️ 最终 SQL 语法检查失败，尝试交给 RewriteAgent 进行修复")
                            print(f"   错误信息: {error_info}")

                            previous_rewrite = rewriter.current_rewrite_result or {
                                "original_sql": initial_sql,
                                "rewritten_sql": final_sql,
                            }

                            # 调用 RewriteAgent 的迭代修复能力
                            async with rewriter.llm_semaphore:
                                corrected_sql = await rewriter.rewrite_agent.iterative_rewrite(
                                    initial_sql, error_info, previous_rewrite
                                )

                            if corrected_sql and isinstance(corrected_sql, str) and corrected_sql.strip():
                                print("🔍 验证修复后的最终 SQL 语法...")
                                corrected_syntax = await DBMS_Syntax_Tool(dbms, corrected_sql)
                                corrected_is_valid = corrected_syntax.get(
                                    "flag", corrected_syntax.get("valid", True)
                                )

                                if corrected_is_valid:
                                    # 更新最终输出中的 SQL 文本（仅更新字符串，不改动成本等其他字段）
                                    try:
                                        if isinstance(rewritten_sql, dict):
                                            tpch_list = rewritten_sql.get("tpch")
                                            if isinstance(tpch_list, list) and tpch_list:
                                                first_entry = tpch_list[0]
                                                if isinstance(first_entry, dict):
                                                    first_entry["rewritten_query"] = corrected_sql
                                    except Exception as update_err:
                                        print(f"⚠️ 更新最终重写 SQL 时发生错误: {update_err}")
                                    else:
                                        print("✅ 最终 SQL 语法修复成功，已更新写入 JSON 的内容")
                                else:
                                    print("❌ 修复后的最终 SQL 仍存在语法问题，将保留原评估版本写入 JSON")
                            else:
                                print("❌ RewriteAgent 未能生成有效的修复 SQL，将保留原评估版本写入 JSON")
                    except Exception as final_check_err:
                        # 二次校验或修复过程本身出错时，不影响主流程，只记录日志
                        print(f"⚠️ 最终 SQL 语法二次校验/修复过程中发生异常: {final_check_err}")

            # LLM cost consumed by this query (all agents)
            usage_after = GPT.get_global_usage()
            cost_after = usage_after.get("total_cost_rmb", 0.0)
            query_llm_cost = max(0.0, cost_after - cost_before)
            query_llm_cost = round(query_llm_cost, 6)

            tmp = {
                "id": item["id"],
                "original_query": item["query"],
                "rewritten_query": rewritten_sql,
                "time_cost": rewrite_time,
                "llm_costs": query_llm_cost,
                "rewrite_suggestion": rewriter.optimization_advice if success else "Error occurred during processing"
            }
            all_results.append(tmp)
            count += 1

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
                    f.write(rewriter.terminal_output or "")
                    f.write("\n" + "-"*40 + "\n\n")

            pbar.update(1)
            pbar.set_postfix({
                'Time': f'{rewrite_time:.2f}s',
                'Status': 'Success' if success else 'Error',
                'Total': count
            })

            await rewriter.clear()
            await rewriter.clear_log()

        # update progress bar when done
        pbar.set_description("🔄 Query Rewriter Completed")
        pbar.set_postfix({'Status': f'Processed {count} queries'})

    print(f"\n✅ Query Rewriter completed! Processed {count} queries")
    print(f"📁 Final output: {final_output_file}")
    return final_output_file

async def run_hint_recommender(args, directories, dbms, rewriter_output_file):
    """run query hint recommender"""
    print("\n" + "="*60)
    print("💡 Starting Hint Recommender")
    print("="*60)
    
    if not rewriter_output_file or not rewriter_output_file.exists():
        print("❌ No rewriter output file found. Skipping hint recommender.")
        return None
    
    recommender = Hint_Recommender(dbms, dbms.db_name)
    temp_dir = directories['recommender_temp']
    
    # load rewriter output data
    print(f"📖 Loading data from {rewriter_output_file}")
    ori_data = recommender.load_data_from_file(str(rewriter_output_file))
    if not ori_data:
        print("❌ No data loaded from rewriter output. Skipping hint recommender.")
        return None
    
    print(f"📊 Processing {len(ori_data)} queries with batch size {args.recommender_batch_size}")
    print(f"📁 Temp directory: {temp_dir}")
    print()  
    
    batch_count = 0
    result = []
    
    with tqdm(total=len(ori_data), 
              desc="💡 Hint Recommender Progress", 
              position=1, 
              leave=True,
              dynamic_ncols=True,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        
        for i, item in enumerate(ori_data):
            # update progress bar description
            pbar.set_description(f"💡 Processing Hint {i + 1}/{len(ori_data)} (ID: {item.get('id', 'N/A')})")
            
            start_time = time.time()
            
            hint_result = recommender.process_single_query(item["original_query"],item["rewritten_query"],item["id"])
            
            end_time = time.time()
            process_time = end_time - start_time
            
            result.append(hint_result)
            
            # update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'Time': f'{process_time:.2f}s',
                'Batch': batch_count + 1 if (i + 1) % args.recommender_batch_size == 0 else batch_count
            })
            
            # save results in batches
            if (i + 1) % args.recommender_batch_size == 0 or i == len(ori_data) - 1:
                batch_count += 1
                batch_file = temp_dir / f"batch_{batch_count}.json"
                
                with open(batch_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                
                result = []
        
        # update progress bar when done
        pbar.set_description("💡 Hint Recommender Completed")
        pbar.set_postfix({'Status': 'Merging files...'})
    
    print(f"\n✅ Hint Recommender completed! Processed {len(ori_data)} queries in {batch_count} batches")
    
    # merge all batch files into output directory
    final_file = merge_batch_files(temp_dir, directories['output'], "recommended_hints.json")
    return final_file

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
    print(f"💡 Recommender: {'Enabled' if args.enable_recommender else 'Disabled'}")
    
    if args.save_rewriter_logs:
        print(f"📝 Rewriter logs: Enabled (batch size forced to 1)")
    else:
        print(f"📝 Rewriter logs: Disabled (batch size: {args.rewriter_batch_size})")
    
    # set up output directories
    directories = setup_directories(args.output_dir, args.enable_rewriter, args.enable_recommender)
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
    
    schema_file = args.schema_file
    with open(schema_file, 'r') as f:
        schema_content = f.read()
        if not schema_content.strip():
            raise ValueError(f"Schema file {schema_file} is empty or not found.")
        print(f"📋 Schema content loaded from {schema_file}")

    ###########################################################
    # QUITE System Execution
    ###########################################################
    # print(args.enable_rewriter, args.enable_recommender)
    rewriter_output_file = None
    recommender_output_file = None
    
    # run Query Rewriter
    if args.enable_rewriter:
        rewriter_output_file = await run_query_rewriter(args, directories, dbms, data_statistics, schema_file)
    
    # run Hint Recommender
    if args.enable_recommender:
        if not args.enable_rewriter:
            # if rewriter is not enabled, check if rewriter output file exists
            existing_file = directories['output'] / "rewritten_queries.json"
            if existing_file.exists():
                rewriter_output_file = existing_file
                print(f"📖 Using existing rewriter output: {existing_file}")
            else:
                print("❌ No existing rewriter output found. Cannot run recommender without rewriter data.")
                return

        if not rewriter_output_file:
            print("❌ No rewriter output available for recommender.")
            return

        recommender_output_file = await run_hint_recommender(args, directories, dbms, rewriter_output_file)

    ###########################################################
    # summary and output
    ###########################################################
    
    print("\n" + "="*60)
    print("🎉 QUITE System Completed!")
    print("="*60)
    
    if rewriter_output_file:
        print(f"📝 Query Rewriter Output: {rewriter_output_file}")
    
    if recommender_output_file:
        print(f"💡 Hint Recommender Output: {recommender_output_file}")
    
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