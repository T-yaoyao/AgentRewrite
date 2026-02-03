import asyncio
import sys
import threading
import json

# Setup project paths
from src.utils.path_config import setup_python_path
setup_python_path()

from src.Rewrite_Middleware.middleware import DBMS_EXPLAIN_Tool, DBMS_Syntax_Tool, Knowledge_Base_Tool, Equivalence_Check_Tool, DBMS
from src.Rewrite_Middleware.Agent_Memory_Buffer.memory_buffer import AgentMemoryBuffer, OutputCollector, create_memory_buffer
from src.Query_Rewriter.agent_definition import ReasoningAgent, DecisionAgent, RewriteAgent, get_rules_by_groups, get_rule_examples
from src.utils.agent_template import MessageContent, Message, MemoryWindow, MessageQueue
from src.Query_Rewriter.global_memory import GlobalMemoryManager


class QueryRewriter:
    """SQL Rewrite Finite State Machine"""
    def __init__(self, message_queue: MessageQueue, dbms: DBMS, data_statistics, schema_file, MAX_ITERATION_LOOP=2):
        self.current_state = "INITIAL_CHECK"
        self.memory = create_memory_buffer(data_statistics, schema_file)

        self.dbms = dbms
        self.iteration = 0
        self.MAX_ITERATION_LOOP = MAX_ITERATION_LOOP
        self.terminal_output = None
        self.output_collector = OutputCollector()

        # Initialize each agent
        self.reasoning_agent = ReasoningAgent(message_queue)
        self.decision_agent = DecisionAgent(message_queue)
        self.rewrite_agent = RewriteAgent(message_queue)

        # Set up observation relationships
        self.decision_agent.watch(["ReasoningAgent","ExplainAgent"])

        # Multi-round optimization variables
        self.optimization_round = 1
        self.can_optimize = None
        self.optimization_advice = []
        self.rule_library = {}
        self.current_rewrite_result = None
        self.previous_feedback = None

        # Parallel processing related
        self.parallel_threads = 2
        self.parallel_reasoning_results = []
        self.parallel_verification_results = []
        self._stop_event = threading.Event()
        self.llm_semaphore = asyncio.Semaphore(3)  # Control LLM concurrency
        self.db_semaphore = asyncio.Semaphore(5)   # Control database concurrency
        
        # Global memory/knowledge base
        self.global_memory = GlobalMemoryManager()
        
        # Knowledge retrieval results (few-shot examples only, no fast track)
        self.few_shot_examples = []
        self.retrieved_record_id = None  # For updating frequency

    @property
    def data_statistics(self):
        return self.memory.data_statistics
    
    @property
    def initial_sql(self):
        return self.memory.initial_sql
    
    @initial_sql.setter
    def initial_sql(self, value):
        self.memory.initial_sql = value
    
    @property
    def optimization_advice(self):
        return self.memory.optimization_advice
    
    @optimization_advice.setter
    def optimization_advice(self, value):
        self.memory.optimization_advice = value
    
    @property
    def produced_sql(self):
        return self.memory.produced_sql
    
    @produced_sql.setter
    def produced_sql(self, value):
        self.memory.produced_sql = value
    
    @property
    def rewritten_sql(self):
        return self.memory.rewritten_sql
    
    @rewritten_sql.setter
    def rewritten_sql(self, value):
        self.memory.rewritten_sql = value
    
    @property
    def re_explain_result(self):
        return self.memory.re_explain_result
    
    @re_explain_result.setter
    def re_explain_result(self, value):
        self.memory.re_explain_result = value
    
    @property
    def ori_explain_result(self):
        return self.memory.ori_explain_result
    
    @ori_explain_result.setter
    def ori_explain_result(self, value):
        self.memory.ori_explain_result = value
    
    @property
    def imp_explain_result(self):
        return self.memory.imp_explain_result
    
    @imp_explain_result.setter
    def imp_explain_result(self, value):
        self.memory.imp_explain_result = value
    
    @property
    def guide_info(self):
        return self.memory.guide_info
    
    @guide_info.setter
    def guide_info(self, value):
        self.memory.guide_info = value
    
    @property
    def report(self):
        return self.memory.report
    
    @report.setter
    def report(self, value):
        self.memory.report = value
    
    @property
    def schema_file(self):
        return self.memory.schema_file
    
    async def clear(self):
        """Clear all initialization variables - using a simplified memory buffer"""
        # Use memory buffer's clearing method
        self.memory.clear_volatile_memory()

        # Reset FSM state to initial state for new query processing
        self.current_state = "INITIAL_CHECK"
        self.iteration = 0
        self.parallel_reasoning_results = []
        self.parallel_verification_results = []
        self._stop_event.clear()

        # Reset multi-round optimization variables
        self.optimization_round = 1
        self.can_optimize = None
        self.optimization_advice = []
        self.rule_library = {}
        self.current_rewrite_result = None
        self.previous_feedback = None
        
        # Reset knowledge retrieval state (no fast track, only few-shot examples)
        self.few_shot_examples = []
        self.retrieved_record_id = None
        
        print(f"🧹 Memory cleared. Buffer status: {self.memory}")
    
    async def clear_log(self):
        if hasattr(self, 'terminal_output'):
            del self.terminal_output
        
    async def run(self):
        self.output_collector.start_collecting()

        while self.current_state != "TERMINATED":

            if self.current_state == "INITIAL_CHECK":
                await self.state_initial_check()

            elif self.current_state == "RULE_SELECTION":
                await self.state_rule_selection()

            elif self.current_state == "REWRITE":
                await self.state_rewrite()

            elif self.current_state == "EVALUATION":
                await self.state_evaluation()

            await asyncio.sleep(0.1)  # Prevent event loop blocking

        terminal_output = self.output_collector.stop_collecting()
        # Store the output in FSM's attributes
        self.terminal_output = terminal_output

        return self.format_final_output()

    async def state_initial_check(self):
        """Initial optimization feasibility check with knowledge retrieval"""
        print("🔍 开始初始优化可行性检查...")

        try:
            # Step 1: Knowledge retrieval (One-Pass Retrieval)
            print("📚 检索历史优化经验...")
            retrieval_results = self.global_memory.retrieve(self.initial_sql, top_k=3)
            
            # Define thresholds for few-shot examples
            # MIN_SIMILARITY_THRESHOLD: Minimum similarity to use as context
            # Similarity < 0.8 usually indicates low relevance and may mislead the model
            MIN_SIMILARITY_THRESHOLD = 0.80  # Minimum threshold for using as few-shot examples
            HIGH_SIMILARITY_THRESHOLD = 0.90  # High similarity threshold (for logging)
            
            # Reset knowledge retrieval state
            self.few_shot_examples = []
            self.retrieved_record_id = None
            
            # Step 2: Filter retrieval results by similarity threshold
            # Only use results above threshold as few-shot examples
            if not retrieval_results:
                # Case: No matches - cold start
                print("🧊 无相关历史经验，完全从头推理")
                self.few_shot_examples = []
            else:
                # Filter results by similarity threshold
                filtered_results = [
                    result for result in retrieval_results 
                    if result.get('score', 0) >= MIN_SIMILARITY_THRESHOLD
                ]
                
                if filtered_results:
                    # Sort by cost reduction rate (descending) to prioritize best optimizations
                    # If multiple results have same similarity, prefer the one with higher cost reduction
                    filtered_results.sort(
                        key=lambda x: (
                            x.get('cost_reduction_rate', 0),  # Primary: cost reduction rate (higher is better)
                            x.get('score', 0)  # Secondary: similarity score (higher is better)
                        ),
                        reverse=True
                    )
                    
                    # Use top 3 results (sorted by cost reduction rate) as few-shot examples
                    self.few_shot_examples = filtered_results[:3]
                    best_result = filtered_results[0]
                    best_score = best_result.get('score', 0)
                    best_reduction = best_result.get('cost_reduction_rate', 0) * 100
                    
                    # Log based on similarity level
                    if best_score >= HIGH_SIMILARITY_THRESHOLD:
                        print(f"✅ 发现高度相似的历史案例 (相似度: {best_score:.4f}, 成本降低: {best_reduction:.2f}%)，作为参考上下文")
                    else:
                        print(f"🤔 发现相似案例 (相似度: {best_score:.4f}, 成本降低: {best_reduction:.2f}%)，作为参考上下文")
                    
                    # Update frequency for the best match (highest cost reduction)
                    if best_result.get('id'):
                        self.global_memory.update_hit_frequency(best_result['id'])
                else:
                    # All results below threshold - don't use as context
                    best_score = retrieval_results[0].get('score', 0) if retrieval_results else 0
                    print(f"🧊 历史案例相似度较低 (相似度: {best_score:.4f} < {MIN_SIMILARITY_THRESHOLD})，不使用作为上下文，完全从头推理")
                    self.few_shot_examples = []
            
            # Step 3: Normal optimization feasibility check (always proceed)
            async with self.llm_semaphore:
                explain_info = await DBMS_EXPLAIN_Tool(self.dbms, self.initial_sql)

            # Extract and store original cost (needed even if cannot optimize)
            original_cost = self._extract_cost_from_explain(explain_info)
            self.final_original_costs = original_cost

            async with self.llm_semaphore:
                check_result = await self.decision_agent.initial_optimization_check(
                    self.initial_sql, self.data_statistics, explain_info, self.few_shot_examples
                )

            self.can_optimize = check_result.get("can_optimize", False)
            self.optimization_advice = check_result.get("advice", [])

            if self.can_optimize:
                print("✅ SQL可以优化，进入规则选择阶段")
                self.current_state = "RULE_SELECTION"
            else:
                print("❌ SQL无需优化，终止流程")
                # If cannot optimize, rewritten cost equals original cost
                self.final_rewritten_costs = original_cost
                self.current_state = "TERMINATED"

        except Exception as e:
            print(f"初始检查失败: {e}")
            import traceback
            traceback.print_exc()
            self.can_optimize = False
            self.current_state = "TERMINATED"

    async def state_rule_selection(self):
        """Rule selection based on optimization advice"""
        print(f"🎯 第{self.optimization_round}轮规则选择...")

        try:
            # Extract groups from advice
            groups = []
            advice_text = ""
            for advice_item in self.optimization_advice:
                group = advice_item.get("group", "")
                suggestion = advice_item.get("produced_suggestion", "")
                if group:
                    groups.append(group)
                advice_text += f"- {group}: {suggestion}\n"

            # Get rules for these groups
            self.rule_library = get_rules_by_groups(groups)

            # Select rule sequence with few-shot examples
            async with self.llm_semaphore:
                rule_sequence = await self.reasoning_agent.select_rule_sequence(
                    self.initial_sql,
                    self.optimization_advice,
                    self.rule_library,
                    self.data_statistics,
                    "",  # explain_info will be empty for now
                    self.optimization_round,
                    self.previous_feedback,
                    self.few_shot_examples
                )

            self.selected_rules = rule_sequence
            print(f"✅ 选择了 {len(rule_sequence.get('applied_rules', []))} 个规则")
            self.current_state = "REWRITE"

        except Exception as e:
            print(f"规则选择失败: {e}")
            self.current_state = "TERMINATED"

    async def state_rewrite(self):
        """Execute SQL rewriting"""
        print("🔧 开始SQL重写...")

        try:
            # Normal rule selection path (no fast track)
            applied_rules = self.selected_rules.get("applied_rules", [])
            rule_examples = get_rule_examples(applied_rules)
            rule_sequence = self.selected_rules

            # Execute rewriting
            async with self.llm_semaphore:
                rewrite_result = await self.rewrite_agent.rewrite_with_rule_sequence(
                    self.initial_sql,
                    rule_sequence,
                    rule_examples,
                    json.dumps(self.optimization_advice, ensure_ascii=False),
                    self.data_statistics,
                    self.schema_file
                )

            self.current_rewrite_result = rewrite_result

            # Check if there was a parse error
            parse_error = rewrite_result.get("parse_error", False)
            rewritten_sql = rewrite_result.get("rewritten_sql", self.initial_sql)
            
            # If parse error occurred and we got the original SQL (meaning extraction failed)
            if parse_error and rewritten_sql == self.initial_sql:
                print("⚠️ 重写结果解析错误且无法提取SQL，尝试使用iterative_rewrite修复...")
                error_info = rewrite_result.get("error_info", "JSON解析错误，无法提取重写SQL")
                print(f"   错误信息: {error_info}")
                
                async with self.llm_semaphore:
                    corrected_sql = await self.rewrite_agent.iterative_rewrite(
                        self.initial_sql, error_info, rewrite_result
                    )
                
                if corrected_sql:
                    rewrite_result["rewritten_sql"] = corrected_sql
                    rewritten_sql = corrected_sql
                    print("✅ 通过iterative_rewrite成功生成SQL")
                else:
                    print("❌ iterative_rewrite未能生成SQL，使用原始SQL继续流程")
            
            # Syntax check
            syntax_check = await DBMS_Syntax_Tool(self.dbms, rewritten_sql)

            # DBMS_Syntax_Tool returns {"flag": True/False, "error": ...}
            # Check both "flag" and "valid" for compatibility
            is_valid = syntax_check.get("flag", syntax_check.get("valid", True))
            
            if not is_valid:
                print("⚠️ 语法检查失败，开始迭代修正...")
                MAX_FIX_ATTEMPTS = 3
                current_sql = rewritten_sql
                fix_attempt = 0
                fix_success = False
                
                while fix_attempt < MAX_FIX_ATTEMPTS and not fix_success:
                    fix_attempt += 1
                    print(f"🔄 第 {fix_attempt}/{MAX_FIX_ATTEMPTS} 次修复尝试...")
                    
                    # Get current error info
                    current_syntax_check = await DBMS_Syntax_Tool(self.dbms, current_sql)
                    error_info = current_syntax_check.get("error", "Unknown error")
                    print(f"   错误信息: {error_info}")
                    
                    # Prepare previous rewrite info with current SQL
                    previous_rewrite_info = rewrite_result.copy()
                    previous_rewrite_info["rewritten_sql"] = current_sql
                    previous_rewrite_info["error_info"] = error_info
                    
                    async with self.llm_semaphore:
                        corrected_sql = await self.rewrite_agent.iterative_rewrite(
                            self.initial_sql, error_info, previous_rewrite_info
                        )
                    
                    if corrected_sql:
                        # Verify corrected SQL syntax
                        print("🔍 验证修正后的SQL语法...")
                        corrected_syntax_check = await DBMS_Syntax_Tool(self.dbms, corrected_sql)
                        corrected_is_valid = corrected_syntax_check.get("flag", corrected_syntax_check.get("valid", True))
                        
                        if corrected_is_valid:
                            rewrite_result["rewritten_sql"] = corrected_sql
                            self.current_rewrite_result = rewrite_result
                            print("✅ 语法修正完成")
                            fix_success = True
                        else:
                            print(f"❌ 修正后的SQL仍有语法错误: {corrected_syntax_check.get('error', 'Unknown error')}")
                            current_sql = corrected_sql  # Use corrected SQL for next attempt
                    else:
                        print("❌ 语法修正失败，无法生成修正后的SQL")
                        break  # Exit loop if can't generate SQL
                
                if not fix_success:
                    print(f"⚠️ 经过 {fix_attempt} 次修复尝试后仍无法修复，使用原始SQL继续流程")
                    rewrite_result["rewritten_sql"] = self.initial_sql
                    self.current_rewrite_result = rewrite_result
            else:
                print("✅ 语法检查通过")

            self.current_state = "EVALUATION"

        except Exception as e:
            print(f"重写失败: {e}")
            self.current_state = "TERMINATED"

    async def state_evaluation(self):
        """Evaluate optimization results"""
        print("📊 开始评估优化结果...")

        try:
            # Get costs for both SQLs
            original_cost_result = await DBMS_EXPLAIN_Tool(self.dbms, self.initial_sql)
            rewritten_sql = self.current_rewrite_result.get("rewritten_sql", self.initial_sql)
            rewritten_cost_result = await DBMS_EXPLAIN_Tool(self.dbms, rewritten_sql)

            # Extract costs (simplified extraction)
            original_cost = self._extract_cost_from_explain(original_cost_result)
            rewritten_cost = self._extract_cost_from_explain(rewritten_cost_result)

            # Store final costs for output
            self.final_original_costs = original_cost
            self.final_rewritten_costs = rewritten_cost

            # Get rule sequence (from selected rules)
            groups = self.selected_rules.get("groups", "") if self.selected_rules else ""
            applied_rules = self.selected_rules.get("applied_rules", []) if self.selected_rules else []

            # Prepare evaluation info
            evaluation_info = {
                "original_costs": original_cost,
                "rewritten_costs": rewritten_cost,
                "original_explain_info": json.dumps(original_cost_result, ensure_ascii=False),
                "rewritten_explain_info": json.dumps(rewritten_cost_result, ensure_ascii=False),
                "groups": groups,
                "applied_rules": applied_rules,
                "original_sql": self.initial_sql,
                "rewritten_sql": rewritten_sql,
                "reason": self.previous_feedback.get("reason", "") if self.previous_feedback else ""
            }

            # Evaluate
            async with self.llm_semaphore:
                evaluation_result = await self.decision_agent.evaluate_with_costs(
                    evaluation_info, self.optimization_round
                )

            terminate = evaluation_result.get("terminate", True)
            should_rollback = evaluation_result.get("是否回退SQL", False)

            if terminate:
                print("✅ 优化完成，终止流程")
                if should_rollback:
                    print("⚠️ 回退到原始SQL")
                    self.rewritten_sql = self.initial_sql
                    # Update rewrite result to reflect rollback
                    if self.current_rewrite_result:
                        self.current_rewrite_result["rewritten_sql"] = self.initial_sql
                    # Clear selected rules since we're rolling back
                    self.selected_rules = None
                    # Clear optimization advice since we're rolling back
                    self.optimization_advice = []
                    # Set rewritten cost to original cost (rollback means no improvement)
                    self.final_rewritten_costs = original_cost
                else:
                    # Store successful optimization to knowledge base
                    if rewritten_cost < original_cost:
                        print(f"📊 准备存储优化案例: 原始成本={original_cost:.2f}, 重写成本={rewritten_cost:.2f}, 降低率={((original_cost-rewritten_cost)/original_cost*100):.2f}%")
                        try:
                            # Get rule sequence (from selected rules)
                            groups = self.selected_rules.get("groups", "") if self.selected_rules else ""
                            applied_rules = self.selected_rules.get("applied_rules", []) if self.selected_rules else []
                            
                            if not applied_rules:
                                print(f"⚠️ 警告：规则序列为空，可能影响存储质量")
                            
                            record_id = self.global_memory.store_successful_optimization(
                                original_sql=self.initial_sql,
                                rewritten_sql=rewritten_sql,
                                rule_sequence=applied_rules,
                                groups=groups,
                                original_cost=original_cost,
                                rewritten_cost=rewritten_cost
                            )
                            if record_id:
                                print(f"💾 已存储成功优化案例到知识库 (ID: {record_id})")
                            else:
                                print(f"⚠️ 存储返回None，可能被过滤（检查日志了解原因）")
                        except Exception as e:
                            print(f"⚠️ 存储知识库失败: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"⚠️ 跳过存储：重写成本 ({rewritten_cost:.2f}) >= 原始成本 ({original_cost:.2f})")
                
                self.current_state = "TERMINATED"
            elif self.optimization_round < self.MAX_ITERATION_LOOP:
                print(f"🔄 第{self.optimization_round}轮优化未通过，准备下一轮...")
                self.optimization_round += 1

                # Update feedback for next round
                self.previous_feedback = {
                    "reason": evaluation_result.get("reason", ""),
                    "problematic_rules": evaluation_result.get("可能造成这个结果的规则", [])
                }

                # Rollback SQL if needed
                if should_rollback:
                    print("🔙 回退到原始SQL")
                    self.current_rewrite_result["rewritten_sql"] = self.initial_sql

                self.current_state = "RULE_SELECTION"
            else:
                print("❌ 已达到最大优化轮数，终止流程")
                # Handle rollback decision even when max rounds reached
                if should_rollback:
                    print("⚠️ 回退到原始SQL")
                    if self.current_rewrite_result:
                        self.current_rewrite_result["rewritten_sql"] = self.initial_sql
                    # Clear selected rules since we're rolling back
                    self.selected_rules = None
                    # Set rewritten cost to original cost (rollback means no improvement)
                    self.final_rewritten_costs = original_cost
                else:
                    print(f"✅ 保留当前重写SQL（未回退）")
                self.current_state = "TERMINATED"

        except Exception as e:
            print(f"评估失败: {e}")
            self.current_state = "TERMINATED"

    def _extract_cost_from_explain(self, explain_result):
        """Extract total cost from explain result"""
        try:
            # If explain_result is a string (JSON), parse it first
            if isinstance(explain_result, str):
                try:
                    explain_result = json.loads(explain_result)
                except json.JSONDecodeError:
                    return 0

            if isinstance(explain_result, dict):
                # Try different possible structures
                if "total_cost" in explain_result:
                    return explain_result["total_cost"]
                elif "Plan" in explain_result and "Total Cost" in explain_result["Plan"]:
                    return explain_result["Plan"]["Total Cost"]
                elif "cost_analysis" in explain_result:
                    return explain_result["cost_analysis"].get("total_cost", 0)
            elif isinstance(explain_result, list) and len(explain_result) > 0:
                plan = explain_result[0]
                if "Plan" in plan and "Total Cost" in plan["Plan"]:
                    return plan["Plan"]["Total Cost"]
        except:
            pass
        return 0

    def format_final_output(self):
        """Format final output according to requirements"""
        # Get final costs (these should be stored during evaluation or initial check)
        original_costs = getattr(self, 'final_original_costs', 0)
        rewritten_costs = getattr(self, 'final_rewritten_costs', 0)
        
        if not self.can_optimize:
            # Cannot optimize case - first round decision
            # Use actual original cost (already extracted in state_initial_check)
            return {
                "tpch": [{
                    "rewritten_query": self.initial_sql,
                    "original_costs": original_costs,
                    "rewrite_costs": original_costs,  # If cannot optimize, rewritten = original
                    "costs_reduction_rate": 0,
                    "rewrite_rules": None
                }]
            }

        # Can optimize case
        final_sql = self.current_rewrite_result.get("rewritten_sql", self.initial_sql) if self.current_rewrite_result else self.initial_sql
        
        # Check if this is a rollback case (final_sql equals original_sql after optimization attempt)
        is_rollback = (final_sql == self.initial_sql)
        
        # Get applied rules (from selected rules)
        # If rolled back, selected_rules should be None (set in state_evaluation)
        applied_rules = None
        if self.selected_rules and not is_rollback:
            applied_rules = self.selected_rules.get("applied_rules", [])
            applied_rules = applied_rules if applied_rules else None
        
        # If rolled back, ensure costs reflect rollback (rewritten = original)
        if is_rollback:
            rewritten_costs = original_costs
            costs_reduction_rate = 0
        else:
            costs_reduction_rate = ((original_costs - rewritten_costs) / original_costs * 100) if original_costs > 0 else 0

        return {
            "tpch": [{
                "rewritten_query": final_sql,
                "original_costs": original_costs,
                "rewrite_costs": rewritten_costs,
                "costs_reduction_rate": round(costs_reduction_rate, 2),
                "rewrite_rules": applied_rules
            }]
        }

    async def parallel_reasoning_worker(self, worker_id: int):
        """Parell reasoning worker"""
        try:
            print(f"\n=== Worker {worker_id}: Start Reasoning Stage ===")
            
            async with self.db_semaphore:
                explain_info = await DBMS_EXPLAIN_Tool(self.dbms, self.initial_sql)
            async with self.llm_semaphore:
                if self.report is None:
                    reasoning_result = await self.reasoning_agent.analyze_sql(
                        self.initial_sql, self.data_statistics, explain_info
                    )
                else:
                    reasoning_result = await self.reasoning_agent.analyze_sql_report(
                        self.initial_sql, self.data_statistics, self.report, explain_info
                    )

            # Extract structured results
            print(f"\n=== Worker {worker_id}: Start Rewrite ===")
            chain = self.decision_agent.retrieve_reasoning_chain()
            async with self.llm_semaphore:
                summary = await self.decision_agent.summarize_chain(
                    chain, self.initial_sql, self.data_statistics
                )
            
            produced_sql = self.decision_agent.extract_sql_candidate_content(summary)
            rewritten_sql = self.decision_agent.extract_rewritten_sql_content(summary)
            optimization_advice = self.decision_agent.extract_advice_content(summary)

            # Check if there is any actual optimization content
            if rewritten_sql and rewritten_sql.strip() and rewritten_sql != self.initial_sql:
                # If there is valid optimized SQL, use it
                print(f"Worker {worker_id}: Found optimized SQL")
                return {
                    "worker_id": worker_id,
                    "status": "success",
                    "produced_sql": produced_sql,
                    "rewritten_sql": rewritten_sql,
                    "optimization_advice": optimization_advice
                }
            elif "TERMINATE" in reasoning_result:
                # Only early stop when there is no optimized SQL and a clear request to terminate
                print(f"Worker {worker_id}: Early stop - no need to optimize")
                return {
                    "worker_id": worker_id,
                    "status": "early_stop",
                    "produced_sql": self.initial_sql,
                    "rewritten_sql": self.initial_sql,
                    "optimization_advice": "No need to optimize"
                }
            else:
                # There are optimization suggestions but no SQL extracted, still considered a success
                print(f"Worker {worker_id}: Using extracted results")
                return {
                    "worker_id": worker_id,
                    "status": "success",
                    "produced_sql": produced_sql or self.initial_sql,
                    "rewritten_sql": rewritten_sql or self.initial_sql,
                    "optimization_advice": optimization_advice or "No specific advice"
                }
            
        except Exception as e:
            print(f"Worker {worker_id}: Reasoning error occurred: {str(e)}")
            return {
                "worker_id": worker_id,
                "status": "error",
                "error": str(e),
                "equivalence_passed": False
            }
        
    async def state_reasoning_parallel(self):
        """Parallel reasoning state"""
        print(f"######################################################################################")
        print(f"\n=== Start Parallel Reasoning (Number of Threads: {self.parallel_threads}) ===")

        # Clear results
        new_reasoning_results = []

        # Create and run parallel tasks
        tasks = []
        for worker_id in range(self.parallel_threads):
            task = asyncio.create_task(self.parallel_reasoning_worker(worker_id))
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if result is not None and not isinstance(result, Exception):
                if result.get("status") in ["success", "early_stop"]:
                    new_reasoning_results.append(result)
                else:
                    # Skip failed workers
                    worker_id = result.get("worker_id", "unknown")
                    print(f"## Worker {worker_id} Reasoning failed. Skipping this worker.##")
        
        self.parallel_reasoning_results = new_reasoning_results
        
        if not self.parallel_reasoning_results:
            print("## All the parallel reasoning workers failed. Use the original SQL.##")
            self.produced_sql = self.initial_sql
            self.rewritten_sql = self.initial_sql
            self.optimization_advice = "No need to optimize"
            self.current_state = "TERMINATED"
        else:
            success_count = len([r for r in self.parallel_reasoning_results if r["status"] == "success"])
            early_stop_count = len([r for r in self.parallel_reasoning_results if r["status"] == "early_stop"])
            print(f"## Parallel reasoning completed, obtained {len(self.parallel_reasoning_results)} Effective results (Suceess: {success_count}, Early Stop: {early_stop_count})##")
            
            self.current_state = "VERIFICATION"

    async def parallel_verification_worker(self, reasoning_result: dict):
        """verification worker"""
        worker_id = reasoning_result["worker_id"]
        rewritten_sql = reasoning_result["rewritten_sql"]
        produced_sql = reasoning_result["produced_sql"]
        optimization_advice = reasoning_result["optimization_advice"]
        
        try:
            print(f"\n=== Worker {worker_id}: Start Verification Stage ===")
            
            # Syntax Check
            async with self.db_semaphore:
                syntax_check = await DBMS_Syntax_Tool(self.dbms, rewritten_sql)
            MAX_CORRECT_TIMES = 0
            MAX_CORRECT_FLAG = False
            
            if not syntax_check["flag"]:
                print(f"## Worker {worker_id}: grammar mistake: {syntax_check['error']}, try to correct it ##")
                CHECK_FLAG = False
                current_error = syntax_check["error"]  # Record the current error to be corrected
                
                while CHECK_FLAG == False and MAX_CORRECT_TIMES < 3:
                    if self._stop_event.is_set():
                        print(f"Worker {worker_id}: interrupted")
                        return None
                        
                    print(f"Worker {worker_id}: ################ Start correct the error ################")
                    print(f"Worker {worker_id}: Current error to fix: {current_error}")
                    async with self.llm_semaphore:
                        checked_sql = await self.rewrite_agent.correct_sql(
                            self.initial_sql, rewritten_sql, current_error  # Always modify the initial version, but use the current error
                        )
                    async with self.db_semaphore:
                        check_result = await DBMS_Syntax_Tool(self.dbms, checked_sql)
                    MAX_CORRECT_TIMES += 1
                    
                    if check_result["flag"]:
                        CHECK_FLAG = True
                        MAX_CORRECT_FLAG = True
                        print(f"-- Worker {worker_id}: ✓ THE GRAMMAR CHECK HAS BEEN PASSED.--")
                        rewritten_sql = checked_sql
                    else:
                        # Update the error information to the current attempted error for reference in the next correction
                        current_error = check_result["error"]
                        print(f"-- Worker {worker_id}: X Grammar check failed again. New error: {current_error}--")
                
                if MAX_CORRECT_FLAG == False:
                    print(f"-- Worker {worker_id}: X The grammar check failed. Please go back and review the reasoning.--")
                    return {
                        "worker_id": worker_id,
                        "status": "syntax_failed",
                        "equivalence_passed": False
                    }
            else:
                print(f"-- Worker {worker_id}: ✓ THE GRAMMAR CHECK HAS BEEN PASSED.--")
            
            # Equivalence Check
            MAX_EQUIV_TIMES = 0
            MAX_EQUIV_FLAG = False
            
            print(f"-- Worker {worker_id}: Perform SQL equivalence check--")
            
            result = await Equivalence_Check_Tool(
                self.initial_sql, rewritten_sql, self.schema_file, timeout=10
            )
            
            if result is not None and "EQ" in result:
                print(f"-- Worker {worker_id}: ✓ Through the optimizer verification, the optimized SQL is equivalent to the original SQL.--")
                MAX_EQUIV_FLAG = True
            else:
                print(f"-- Worker {worker_id}: X Not verified by the optimizer, calling LLM to rewrite process--")
                CHECK_EQUIV_FLAG = False
                tmp_checked_sql = rewritten_sql
                
                while CHECK_EQUIV_FLAG == False and MAX_EQUIV_TIMES < 3:
                    if self._stop_event.is_set():
                        print(f"-- Worker {worker_id}: interrupted--")
                        return None
                        
                    async with self.llm_semaphore:
                        checked_report = await self.decision_agent.check_equivalence(
                            self.initial_sql, tmp_checked_sql, optimization_advice
                        )
                    CHCKED_FLAG = self.decision_agent.extract_equivalence_content(checked_report)
                    MAX_EQUIV_TIMES += 1
                    
                    if "true" in CHCKED_FLAG or "True" in CHCKED_FLAG:
                        CHECK_EQUIV_FLAG = True
                        MAX_EQUIV_FLAG = True
                        rewritten_sql = tmp_checked_sql
                        print(f"-- Worker {worker_id}: ✓ Optimize SQL to be equivalent to the original SQL--")
                    else:
                        print(f"-- Worker {worker_id}: X Optimize SQL not be equivalent to the original SQL (try {MAX_EQUIV_TIMES}/3)--")
                        corrected_sql = self.decision_agent.extract_corrected_sql_content(checked_report)
                        if corrected_sql:
                            tmp_checked_sql = corrected_sql
                            print(f"-- Worker {worker_id}: The revised and optimized SQL query: {tmp_checked_sql}--")
                
                if MAX_EQUIV_FLAG == False:
                    print(f"-- Worker {worker_id}: X The equivalence check failed. The maximum number of attempts has been reached. Using the original SQL.--")
                    rewritten_sql = self.initial_sql
                    self.optimization_advice = "No need to optimize"
            
            print(f"-- Worker {worker_id}: Verification completed")
            return {
                "worker_id": worker_id,
                "status": "success",
                "produced_sql": produced_sql,
                "rewritten_sql": rewritten_sql,
                "optimization_advice": optimization_advice,
                "equivalence_passed": MAX_EQUIV_FLAG
            }
            
        except Exception as e:
            print(f"Worker {worker_id}: Verification errors: {str(e)}")
            return {
                "worker_id": worker_id,
                "status": "error",
                "error": str(e),
                "equivalence_passed": False
            }
    

    async def state_verification_parallel(self):
        """Parallel verification state"""
        print(f"######################################################################################")
        print(f"\n=== Start Parallel Verification ===")

        # Reset results and stop event
        new_verification_results = []
        self._stop_event.clear()

        # Only successful reasoning results are subject to verification
        successful_reasoning = [r for r in self.parallel_reasoning_results
                               if r["status"] == "success"]
        early_stop_reasoning = [r for r in self.parallel_reasoning_results 
                               if r["status"] == "early_stop"]

        # Early stop results can be directly added to the final results
        for early_stop in early_stop_reasoning:
            new_verification_results.append({
                "worker_id": early_stop["worker_id"],
                "status": "success",
                "produced_sql": early_stop["produced_sql"],
                "rewritten_sql": early_stop["rewritten_sql"],
                "optimization_advice": early_stop["optimization_advice"]
            })

        # Create tasks for workers that need verification
        tasks = []
        for reasoning_result in successful_reasoning:
            task = asyncio.create_task(self.parallel_verification_worker(reasoning_result))
            tasks.append(task)

        if tasks:  # If there are tasks that need verification
            # Wait for all verification tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process verification results
            for result in results:
                if result is not None and not isinstance(result, Exception):
                    if result.get("status") == "success":
                        new_verification_results.append(result)
                    else:
                        # Skip failed workers
                        worker_id = result.get("worker_id", "unknown")
                        print(f"## Worker {worker_id} Verification failed. Skipping this worker.##")

        # Determine the next state
        if new_verification_results:
            # If there are successful verification results, proceed to decision 
            print(f"## Parallel verification complished, obtains {len(new_verification_results)} effective results##")
            self.parallel_verification_results = new_verification_results
            self.current_state = "DECISION"
        else:
            # All verification attempts failed, terminate using the original SQL
            print("## All the parallel verification workers failed. Using the original SQL.##")
            self.produced_sql = self.initial_sql
            self.rewritten_sql = self.initial_sql
            self.optimization_advice = "No need to optimize"
            self.current_state = "TERMINATED"



    async def state_decision(self):
        """Decision state - includes selection and decision logic"""
        print(f"######################################################################################")
        print(f"\n=== Start Decision Stage: (Iteration: {self.iteration}/{self.MAX_ITERATION_LOOP}) ===")

        # Step 1: Select the optimal SQL (if there are multiple verification results)
        if len(self.parallel_verification_results) == 1:
            # Only one result, use it directly
            selected_result = self.parallel_verification_results[0]
            print(f"## There is only one valid option, so choose it directly Worker {selected_result['worker_id']}##")
        else:
            # Multiple results, need to select
            print(f"##  {len(self.parallel_verification_results)} results, try to select the best one ##")
            query_pairs = []
            for result in self.parallel_verification_results:
                query_pairs.append({
                    "id": result["worker_id"],
                    "rewritten_sql": result["rewritten_sql"],
                })
            
            async with self.llm_semaphore:
                selected_advice = await self.decision_agent.select_sql(self.initial_sql, query_pairs)
            selected_id = self.decision_agent.extract_selected_id_content(selected_advice)

            # Find the corresponding result
            selected_result = None
            for result in self.parallel_verification_results:
                if result["worker_id"] == selected_id:
                    selected_result = result
                    break
            
            if selected_result is None:
                print(f"## Do not find selected worker {selected_id}, use the first result instead ##")
                selected_result = self.parallel_verification_results[0]
            else:
                print(f"## selected Worker {selected_id}##")

        # Set the selected result
        self.produced_sql = selected_result["produced_sql"]
        self.rewritten_sql = selected_result["rewritten_sql"]
        self.optimization_advice = selected_result["optimization_advice"]

        print(f"## Selected Rewritten SQL: {self.rewritten_sql} ##")
        print(f"## Rewrite Proposals: {self.optimization_advice} ##")


        # Generate report
        print(f"## Generate optimization report... ##")
        async with self.db_semaphore:
            # Execute the two explain tasks in parallel
            ori_explain_task = asyncio.create_task(DBMS_EXPLAIN_Tool(self.dbms, self.initial_sql))
            enhanced_explain_task = asyncio.create_task(DBMS_EXPLAIN_Tool(self.dbms, self.rewritten_sql))
            
            ori_explain_result, enhanced_explain_result = await asyncio.gather(
                ori_explain_task, enhanced_explain_task
            )
        # Store explain results as report (simplified - no detailed report generation needed)
        self.report = {
            "original_explain": ori_explain_result,
            "rewritten_explain": enhanced_explain_result
        }
        print(f"## Report generation completed ##")

        # Step 2: Make a decision
        # Collect equivalence flags from all verification results
        worker_equivalence_flags = []
        for result in self.parallel_verification_results:
            equivalence_passed = result.get("equivalence_passed", True)  # Default to True for backward compatibility
            worker_equivalence_flags.append(equivalence_passed)
        
        async with self.llm_semaphore:
            decision = await self.decision_agent.evaluate(
                self.initial_sql,
                self.rewritten_sql,
                self.report,
                worker_equivalence_flags
            )
        
        if self.iteration < self.MAX_ITERATION_LOOP:
            self.iteration += 1
            if "true" in decision or "True" in decision:
                print(f"## The decision has been made. Optimization  is terminated (iteration: {self.iteration}/{self.MAX_ITERATION_LOOP}) ##")
                self.current_state = "TERMINATED"
            else:
                print(f"## The decision was not approved. Further optimization is required (iteration: {self.iteration}/{self.MAX_ITERATION_LOOP}) ##")
                
                print("## Start a new round of complete iteration ## ")
                rag_knowledge = await Knowledge_Base_Tool(self.rewritten_sql, self.optimization_advice)
                async with self.llm_semaphore:
                    self.optimization_advice = await self.decision_agent.merge_advice(
                        self.rewritten_sql, 
                        self.optimization_advice,
                        rag_knowledge
                    )
                input_report = {
                    "decision": decision,
                    "pre_rewrite_sql": self.rewritten_sql,
                    "corrected_guide_knowledge": rag_knowledge,
                }
                self.guide_info = input_report
                self.report = input_report

                # Restart the entire process and reset all states
                self.parallel_reasoning_results = []
                self.parallel_verification_results = []
                self.current_state = "REASONING"
        else:
            print(f"## Reach the maximum number of iterations ({self.MAX_ITERATION_LOOP}),print Terminate optimization ##")
            self.rewritten_sql = self.initial_sql
            self.current_state = "TERMINATED"
            self.optimization_advice = "No need to optimize"




# Save terminal output to txt file each time a JSON file is saved
def save_terminal_output_to_file(save_file_path, batch, original_stdout, temp_file):
    # Restore standard output
    sys.stdout = original_stdout
    temp_file.close()

    # Copy the contents of the temporary file to the final file
    with open(f"{save_file_path}/batch_{batch}.txt", "w") as f:
        with open("temp_output.txt", "r") as temp_f:
            f.write(temp_f.read())

    # Redirect standard output to temporary file again
    temp_file = open("temp_output.txt", "w")
    sys.stdout = temp_file
    return temp_file


