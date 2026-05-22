"""
LangGraph 编排的查询重写管线：
initial_check → rule_selection → rewrite → syntax_check → semantic_check → evaluation
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.utils.path_config import setup_python_path, load_project_env

setup_python_path()
load_project_env()

from src.Rewrite_Middleware.middleware import (
    DBMS,
    DBMS_EXPLAIN_Tool,
    DBMS_RAW_EXPLAIN_JSON_Tool,
    DBMS_Syntax_Tool,
    Equivalence_Check_Tool,
    _normalize_sql_text,
)
from src.Rewrite_Middleware.plan_structure_compare import compare_explain_plan_structures
from src.Rewrite_Middleware.Agent_Memory_Buffer.memory_buffer import (
    AgentMemoryBuffer,
    OutputCollector,
    create_memory_buffer,
)
from src.Query_Rewriter.agent_definition import (
    DecisionAgent,
    ReasoningAgent,
    RewriteAgent,
    SemanticCheckAgent,
    get_all_rules,
    get_rules_by_groups,
)
from src.Query_Rewriter.schema_context import (
    build_filtered_schema_content,
    filter_data_statistics_for_sql,
)
from src.utils.agent_template import MessageQueue


class RewriteState(TypedDict, total=False):
    """LangGraph 状态：随节点执行逐步写入各 Agent 的结构化输出。
    schema_file：完整 DDL 文件路径（溯源）；schema_content：按 initial_sql 裁剪后的 DDL，供各节点与 Agent 使用。
    """

    initial_sql: str
    current_sql: str
    data_statistics: Any
    index_info: str
    schema_file: str
    schema_content: str
    max_iterations: int

    current_round: int
    can_optimize: bool
    should_terminate: bool

    optimization_advice: List[Dict]
    selected_rules: Optional[Dict]
    current_rewrite_result: Optional[Dict]

    # 与 can_optimize 同粒度：各步 Agent 的可读字段（非整包 dict）
    initial_optimization_reason: str
    semantic_check_note: str
    semantic_equivalent: Optional[bool]
    semantic_equivalence_message: str
    semantic_equivalence_differences: List[Dict]
    initial_explain_info: str
    evaluation_reason: str
    evaluation_terminate: bool
    evaluation_no_further_optimization: bool
    evaluation_next_step_advice: str

    original_cost: float
    current_cost: float
    best_cost: Optional[float]
    best_sql: Optional[str]
    best_rules: Optional[List]
    best_groups: str

    previous_feedback: Optional[Dict]

    few_shot_examples: List
    retrieved_record_id: Optional[str]

    final_original_costs: float
    final_rewritten_costs: float
    rewritten_sql: str

    agent_trace: List[Dict[str, Any]]

    memory: Any
    output_collector: Any


def _stats_str(stats: Any) -> str:
    if isinstance(stats, list):
        return json.dumps(stats, ensure_ascii=False)
    return stats if isinstance(stats, str) else json.dumps(stats, ensure_ascii=False)


def _state_sql(sql: Any) -> str:
    """与 DB 工具一致的 SQL 文本清洗；写入 LangGraph 状态中的 SQL 均应经此处理。"""
    return _normalize_sql_text(sql)


class LangGraphQueryRewriter:
    MIN_SIMILARITY = 0.80
    HIGH_SIMILARITY = 0.90
    MAX_SYNTAX_FIX = 3
    MAX_SEMANTIC_FIX = 3
    MAX_SYNTAX_AFTER_SEMANTIC = 3
    # 估计代价变差超过该比例时回退到 best/原 SQL；结构分析若显示明显改进可保留（见 plan_structure_compare）。
    COST_ROLLBACK_PCT = 50.0
    # If estimated cost drops by more than this fraction vs original, skip LLM evaluation and exit.
    EARLY_TERMINATE_COST_REDUCTION_RATIO = 0.40
    # Plan-structure metadata weights for evaluation bookkeeping.
    REWARD_ROWS_WEIGHT = 0.40
    REWARD_DEPTH_WEIGHT = 0.30
    # Keep rewrite when cost/plan are near-equivalent to avoid over-rollback.
    NEAR_EQUAL_COST_PCT = 0.10
    # Treat rewrite as strong degradation when estimated cost explodes beyond this percentage.
    COST_EXPLOSION_PCT = 1000.0

    def __init__(
        self,
        message_queue: MessageQueue,
        dbms: DBMS,
        data_statistics: Any,
        schema_file: str,
        MAX_ITERATION_LOOP: int = 2,
        index_info: str = "",
    ):
        self.message_queue = message_queue
        self.dbms = dbms
        self.data_statistics = data_statistics
        self.index_info = index_info or ""
        self.schema_file = schema_file
        self.MAX_ITERATION_LOOP = MAX_ITERATION_LOOP

        self.reasoning_agent = ReasoningAgent(message_queue)
        self.decision_agent = DecisionAgent(message_queue)
        self.rewrite_agent = RewriteAgent(message_queue)
        self.semantic_check_agent = SemanticCheckAgent(message_queue)
        self.decision_agent.watch(["ReasoningAgent", "ExplainAgent"])

        self.llm_semaphore = asyncio.Semaphore(3)
        self.db_semaphore = asyncio.Semaphore(5)

        # Global memory is intentionally disabled for this ablation variant.
        # Keep the attribute so existing branches remain simple, but do not
        # retrieve from or write to the non-parametric memory store.
        self.global_memory = None
        print("🧪 All memory disabled (no Track1 / no Track2 ablation)")

        self.graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(RewriteState)
        g.add_node("initial_check", self._initial_check_node)
        g.add_node("rule_selection", self._rule_selection_node)
        g.add_node("rewrite", self._rewrite_node)
        g.add_node("syntax_check", self._syntax_check_node)
        g.add_node("semantic_check", self._semantic_check_node)
        g.add_node("evaluation", self._evaluation_node)

        g.set_entry_point("initial_check")
        g.add_conditional_edges(
            "initial_check",
            self._route_after_initial,
            {"continue": "rule_selection", "end": END},
        )
        g.add_edge("rule_selection", "rewrite")
        g.add_edge("rewrite", "syntax_check")
        g.add_edge("syntax_check", "semantic_check")
        g.add_edge("semantic_check", "evaluation")
        g.add_conditional_edges(
            "evaluation",
            self._route_after_eval,
            {"again": "initial_check", "end": END},
        )
        return g.compile()

    def _route_after_initial(self, state: RewriteState) -> str:
        return "continue" if state.get("can_optimize") else "end"

    def _route_after_eval(self, state: RewriteState) -> str:
        if not state.get("should_terminate", True):
            return "again"
        return "end"

    def _trace(self, state: RewriteState, node: str, payload: Dict[str, Any]) -> List[Dict]:
        tr = list(state.get("agent_trace") or [])
        tr.append({"node": node, "output": payload})
        return tr

    def _extract_cost_from_explain(self, explain_result) -> float:
        try:
            if isinstance(explain_result, str):
                m = re.search(r"执行计划总代价:\s*([0-9,]+\.?[0-9]*)", explain_result)
                if m:
                    return float(m.group(1).replace(",", ""))
                try:
                    explain_result = json.loads(explain_result)
                except json.JSONDecodeError:
                    return 0.0
            if isinstance(explain_result, dict):
                if "total_cost" in explain_result:
                    return float(explain_result["total_cost"])
                if "Plan" in explain_result and "Total Cost" in explain_result["Plan"]:
                    return float(explain_result["Plan"]["Total Cost"])
            if isinstance(explain_result, list) and explain_result:
                p = explain_result[0]
                if "Plan" in p and "Total Cost" in p["Plan"]:
                    return float(p["Plan"]["Total Cost"])
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _parse_equivalence_tool_result(raw_result: Any) -> Dict[str, str]:
        """
        Normalize sqlsolver/Equivalence_Check_Tool output.

        Returns status in {"equivalent", "not_equivalent", "unknown"} and keeps
        the raw tool output as the reason so rewrite repair can use it.
        """
        if raw_result is None:
            return {"status": "unknown", "reason": "Equivalence_Check_Tool returned no output."}

        text = str(raw_result).strip()
        if not text:
            return {"status": "unknown", "reason": "Equivalence_Check_Tool returned empty output."}

        lowered = text.lower()
        compact = re.sub(r"[\s_\-]+", "", lowered)

        if "unknown" in lowered or "timeout" in lowered:
            return {"status": "unknown", "reason": text}

        negative_patterns = (
            r"\bneq\b",
            r"\bnot\s+equiv(?:alent)?\b",
            r"\bnon[-\s]?equiv(?:alent)?\b",
            r"\bnot\s+equal\b",
            r"\bfalse\b",
            r"\bcounter\s*example\b",
            r"\bcounterexample\b",
        )
        if compact in {"neq", "noteq", "notequivalent", "false"} or any(
            re.search(pattern, lowered) for pattern in negative_patterns
        ):
            return {"status": "not_equivalent", "reason": text}

        positive_patterns = (
            r"\beq\b",
            r"\bequiv(?:alent)?\b",
            r"\btrue\b",
        )
        if compact in {"eq", "equiv", "equivalent", "true"} or any(
            re.search(pattern, lowered) for pattern in positive_patterns
        ):
            return {"status": "equivalent", "reason": text}

        return {"status": "unknown", "reason": text}

    @staticmethod
    def _count_sql_subquery_markers(sql_text: str) -> int:
        if not isinstance(sql_text, str) or not sql_text.strip():
            return 0
        return len(re.findall(r"\(\s*select\b", sql_text, flags=re.IGNORECASE))

    @classmethod
    def _count_plan_subplans(cls, explain_payload: Any) -> int:
        roots = cls._extract_plan_roots(explain_payload)
        count = 0

        def _walk(node: Dict[str, Any]) -> None:
            nonlocal count
            parent_rel = str(node.get("Parent Relationship") or "")
            if parent_rel in {"SubPlan", "InitPlan"} or node.get("Subplan Name"):
                count += 1
            for child in node.get("Plans") or []:
                if isinstance(child, dict):
                    _walk(child)

        for root in roots:
            _walk(root)
        return count

    @classmethod
    def _analyze_reuse_positive_signals(
        cls,
        original_sql: str,
        rewritten_sql: str,
        original_explain: Any,
        rewritten_explain: Any,
    ) -> Dict[str, Any]:
        """
        Detect strong positive heuristics that are often underestimated by cost:
        decorrelation, subplan elimination, and common-result reuse via CTE.
        """
        orig_subqueries = cls._count_sql_subquery_markers(original_sql)
        rew_subqueries = cls._count_sql_subquery_markers(rewritten_sql)
        orig_subplans = cls._count_plan_subplans(original_explain)
        rew_subplans = cls._count_plan_subplans(rewritten_explain)
        rewrite_has_with = bool(re.search(r"^\s*with\b", rewritten_sql or "", re.IGNORECASE))
        original_has_with = bool(re.search(r"^\s*with\b", original_sql or "", re.IGNORECASE))
        rewrite_cte_count = len(re.findall(r"(?i)\bwith\b|\),\s*[a-zA-Z_][a-zA-Z0-9_]*\s+as\s*\(", rewritten_sql or ""))
        original_cte_count = len(re.findall(r"(?i)\bwith\b|\),\s*[a-zA-Z_][a-zA-Z0-9_]*\s+as\s*\(", original_sql or ""))
        rewrite_has_derived_union = bool(
            re.search(r"join\s*\(\s*select[\s\S]*?\bunion\s+all\b", rewritten_sql or "", re.IGNORECASE)
        )
        rewrite_has_preagg_cte = bool(
            re.search(
                r"\bas\s*\(\s*select[\s\S]*?\b(sum|avg|min|max|count)\s*\(",
                rewritten_sql or "",
                re.IGNORECASE,
            )
        )
        rewrite_has_group_by = bool(re.search(r"\bgroup\s+by\b", rewritten_sql or "", re.IGNORECASE))
        original_has_group_by = bool(re.search(r"\bgroup\s+by\b", original_sql or "", re.IGNORECASE))
        rewrite_has_cross_join_cte = bool(
            re.search(r"\bcross\s+join\s+[a-zA-Z_][a-zA-Z0-9_]*\b", rewritten_sql or "", re.IGNORECASE)
        )

        reasons: List[str] = []
        positive = False

        if orig_subplans > rew_subplans and orig_subplans >= 1:
            positive = True
            reasons.append(
                f"检测到子计划/初始化计划减少: {orig_subplans}->{rew_subplans}，"
                "说明重写可能消除了重复执行或去相关。"
            )

        if rewrite_has_with and orig_subqueries > rew_subqueries:
            positive = True
            reasons.append(
                f"检测到嵌套子查询标记减少: {orig_subqueries}->{rew_subqueries}，"
                "且重写使用CTE，说明可能将相关子查询/重复计算改写为一次性公共结果复用。"
            )

        if rewrite_has_with and orig_subqueries >= 1 and rew_subqueries == 0:
            positive = True
            reasons.append(
                "原SQL含子查询而重写SQL改为CTE/连接且不再包含嵌套子查询标记，"
                "这通常意味着去相关或消除重复执行。"
            )

        if rewrite_has_with and not original_has_with and rewrite_cte_count >= 1:
            positive = True
            reasons.append(
                "重写SQL显式引入CTE组织公共中间结果，说明存在一次性计算后复用的意图。"
            )

        if rewrite_has_derived_union:
            positive = True
            reasons.append(
                "重写SQL将 UNION ALL 置于派生表/JOIN 路径中，常对应分支过滤下推与公共结果复用。"
            )

        if rewrite_has_preagg_cte and (not original_has_group_by or rewrite_cte_count > original_cte_count):
            positive = True
            reasons.append(
                "检测到预聚合CTE/子查询（含 SUM/AVG/MIN/MAX/COUNT），"
                "这通常用于先压缩中间结果再参与 JOIN，可视为公共中间结果复用的强信号。"
            )

        if rewrite_has_cross_join_cte:
            positive = True
            reasons.append(
                "检测到通过 CROSS JOIN/显式连接方式复用CTE结果，说明边界或参数值被一次性计算并复用。"
            )

        return {
            "positive": positive,
            "original_subqueries": orig_subqueries,
            "rewritten_subqueries": rew_subqueries,
            "original_subplans": orig_subplans,
            "rewritten_subplans": rew_subplans,
            "rewrite_has_with": rewrite_has_with,
            "original_cte_count": original_cte_count,
            "rewrite_cte_count": rewrite_cte_count,
            "rewrite_has_derived_union": rewrite_has_derived_union,
            "rewrite_has_preagg_cte": rewrite_has_preagg_cte,
            "rewrite_has_cross_join_cte": rewrite_has_cross_join_cte,
            "reasons": reasons,
        }

    @staticmethod
    def _extract_plan_roots(explain_payload: Any) -> List[Dict[str, Any]]:
        """Best-effort parse EXPLAIN payload into root Plan dict list."""
        if explain_payload is None:
            return []
        payload = explain_payload
        if isinstance(payload, str):
            s = payload.strip()
            if not s:
                return []
            try:
                payload = json.loads(s)
            except json.JSONDecodeError:
                return []
        if isinstance(payload, dict):
            plan = payload.get("Plan")
            if isinstance(plan, dict):
                return [plan]
            return []
        if isinstance(payload, list):
            roots: List[Dict[str, Any]] = []
            for item in payload:
                if isinstance(item, dict) and isinstance(item.get("Plan"), dict):
                    roots.append(item["Plan"])
            return roots
        return []

    @classmethod
    def _collect_rows_and_depth(cls, node: Dict[str, Any], depth: int = 1) -> Dict[str, float]:
        """Collect heuristic structure metrics: sum(plan rows) and max tree depth."""
        sum_rows = 0.0
        max_depth = float(depth)
        pr = node.get("Plan Rows")
        if pr is not None:
            try:
                sum_rows += float(pr)
            except (TypeError, ValueError):
                pass
        for child in node.get("Plans") or []:
            if not isinstance(child, dict):
                continue
            child_m = cls._collect_rows_and_depth(child, depth + 1)
            sum_rows += float(child_m["sum_rows"])
            max_depth = max(max_depth, float(child_m["max_depth"]))
        return {"sum_rows": sum_rows, "max_depth": max_depth}

    @classmethod
    def _compute_structure_reward(cls, original_explain: Any, rewritten_explain: Any) -> Dict[str, float]:
        """
        Structure reward in [-1, 1], based on:
        - row estimate improvement (sum of Plan Rows, weak positive-only hint)
        - plan tree depth improvement (shallower tree preferred)
        """
        o_roots = cls._extract_plan_roots(original_explain)
        r_roots = cls._extract_plan_roots(rewritten_explain)
        if not o_roots or not r_roots:
            return {
                "structure_reward": 0.0,
                "rows_score": 0.0,
                "depth_score": 0.0,
                "o_rows": 0.0,
                "r_rows": 0.0,
                "o_depth": 0.0,
                "r_depth": 0.0,
            }

        def _merge(roots: List[Dict[str, Any]]) -> Dict[str, float]:
            rows = 0.0
            depth = 0.0
            for root in roots:
                m = cls._collect_rows_and_depth(root, depth=1)
                rows += float(m["sum_rows"])
                depth = max(depth, float(m["max_depth"]))
            return {"rows": rows, "depth": depth}

        om = _merge(o_roots)
        rm = _merge(r_roots)
        o_rows, r_rows = om["rows"], rm["rows"]
        o_depth, r_depth = om["depth"], rm["depth"]

        # Treat summed Plan Rows as a weak auxiliary hint:
        # reward clear reductions, but do not punish increases because
        # CTE/derived-table/pre-aggregation rewrites often inflate intermediate
        # cardinality estimates while still running much faster in reality.
        if o_rows >= 100.0 and o_rows > 0:
            rows_score = max(0.0, min(1.0, 1.0 - (r_rows / o_rows)))
        else:
            rows_score = 0.0

        # Depth reward: shallower rewritten plan => positive score.
        if o_depth > 0:
            depth_score = max(-1.0, min(1.0, 1.0 - (r_depth / o_depth)))
        else:
            depth_score = 0.0

        structure_reward = (
            cls.REWARD_ROWS_WEIGHT * rows_score
            + cls.REWARD_DEPTH_WEIGHT * depth_score
        )
        structure_reward = max(-1.0, min(1.0, float(structure_reward)))
        return {
            "structure_reward": structure_reward,
            "rows_score": float(rows_score),
            "depth_score": float(depth_score),
            "o_rows": float(o_rows),
            "r_rows": float(r_rows),
            "o_depth": float(o_depth),
            "r_depth": float(r_depth),
        }

    def _is_near_equivalent_plan_and_cost(
        self,
        plan_struct: Optional[Dict[str, Any]],
        original_cost: float,
        rewritten_cost: float,
    ) -> bool:
        """
        Decide whether original/rewrite are close enough to prefer keeping rewrite SQL.
        """
        if original_cost <= 0 or rewritten_cost <= 0:
            return False
        cost_gap = abs(rewritten_cost - original_cost) / max(original_cost, 1e-9)
        if cost_gap > self.NEAR_EQUAL_COST_PCT:
            return False
        if not isinstance(plan_struct, dict) or not plan_struct.get("ok"):
            return True
        om = plan_struct.get("original_metrics") or {}
        rm = plan_struct.get("rewritten_metrics") or {}
        small_deltas = (
            abs(int(om.get("seq_scan", 0)) - int(rm.get("seq_scan", 0))) <= 1
            and abs(int(om.get("index_like", 0)) - int(rm.get("index_like", 0))) <= 1
            and abs(int(om.get("nested_loop", 0)) - int(rm.get("nested_loop", 0))) <= 1
            and abs(int(om.get("join_hash_merge", 0)) - int(rm.get("join_hash_merge", 0))) <= 1
        )
        # Do not use summed Plan Rows as a closeness gate. It is too unstable
        # under decorrelation / CTE / pre-aggregation rewrites and should remain
        # an auxiliary reference instead of a blocking condition.
        return small_deltas

    async def _initial_check_node(self, state: RewriteState) -> Dict[str, Any]:
        print("🔍 开始初始优化可行性检查...")
        trace = list(state.get("agent_trace") or [])
        few_shot: List = []
        retrieved_id = None
        try:
            base_sql = _state_sql(state.get("current_sql") or state["initial_sql"])
            # Non-parametric global memory retrieval is disabled in this no-memory ablation.
            async with self.db_semaphore:
                explain_info = await DBMS_EXPLAIN_Tool(self.dbms, base_sql)
            base_cost = self._extract_cost_from_explain(explain_info)
            stats = _stats_str(state["data_statistics"])
            idx = state.get("index_info") or self.index_info
            async with self.llm_semaphore:
                check = await self.decision_agent.initial_optimization_check(
                    base_sql,
                    stats,
                    explain_info,
                    few_shot,
                    index_info=idx,
                    previous_feedback=state.get("previous_feedback"),
                )
            trace.append({"node": "initial_check", "output": check})
            can = check.get("can_optimize", False)
            advice = check.get("advice", [])
            if can:
                print("✅ SQL 可以优化，进入规则选择")
            else:
                print("❌ SQL 无需优化，终止流程")
            return {
                "can_optimize": can,
                "optimization_advice": advice,
                "initial_optimization_reason": check.get("reason", "") or "",
                "initial_explain_info": (
                    json.dumps(explain_info, ensure_ascii=False)
                    if not isinstance(explain_info, str)
                    else explain_info
                ),
                "original_cost": state.get("original_cost", base_cost) if state.get("original_cost") else base_cost,
                "current_cost": base_cost,
                "final_original_costs": state.get("final_original_costs", base_cost) if state.get("final_original_costs") else base_cost,
                "few_shot_examples": few_shot,
                "retrieved_record_id": retrieved_id,
                "agent_trace": trace,
                "should_terminate": not can,
                "current_sql": base_sql,
                "rewritten_sql": base_sql,
                "final_rewritten_costs": base_cost if not can else state.get("final_rewritten_costs", 0.0),
            }
        except Exception as e:
            print(f"初始检查失败: {e}")
            import traceback

            traceback.print_exc()
            base_sql = _state_sql(state.get("current_sql") or state["initial_sql"])
            oc = state.get("original_cost", 0.0)
            return {
                "can_optimize": False,
                "should_terminate": True,
                "initial_optimization_reason": "",
                "initial_explain_info": "",
                "agent_trace": trace + [{"node": "initial_check", "error": str(e)}],
                "final_original_costs": oc,
                "final_rewritten_costs": state.get("current_cost", oc),
                "current_sql": base_sql,
                "rewritten_sql": base_sql,
            }

    async def _rule_selection_node(self, state: RewriteState) -> Dict[str, Any]:
        rnd = state.get("current_round", 1)
        print(f"🎯 第{rnd}轮规则选择...")
        trace = list(state.get("agent_trace") or [])
        try:
            base_sql = _state_sql(state.get("current_sql") or state["initial_sql"])
            groups: List[str] = []
            for advice in state.get("optimization_advice") or []:
                if not isinstance(advice, dict):
                    continue
                group = str(advice.get("group") or "").strip()
                if group and group not in groups:
                    groups.append(group)

            lib = get_rules_by_groups(groups) if groups else {}
            if not lib:
                lib = get_all_rules()
                group_note = "未命中优化组，回退到全量规则库"
            else:
                group_note = "按优化组筛选: " + ", ".join(groups)
            stats = _stats_str(state["data_statistics"])
            idx = state.get("index_info") or self.index_info
            explain_info = state.get("initial_explain_info", "")
            if not explain_info:
                async with self.db_semaphore:
                    explain = await DBMS_EXPLAIN_Tool(self.dbms, base_sql)
                explain_info = (
                    json.dumps(explain, ensure_ascii=False)
                    if not isinstance(explain, str)
                    else explain
                )

            rule_count = sum(len(rules or {}) for rules in lib.values())
            print(f"📚 加载规则库 {rule_count} 条规则（{group_note}；记忆已关闭）")

            async with self.llm_semaphore:
                seq = await self.reasoning_agent.select_rule_sequence(
                    base_sql,
                    state["optimization_advice"],
                    lib,
                    stats,
                    explain_info,
                    rnd,
                    state.get("previous_feedback"),
                    state.get("few_shot_examples") or [],
                    index_info=idx,
                )

            trace.append({"node": "rule_selection", "output": seq})
            print(f"✅ 选择了 {len(seq.get('applied_rules', []))} 个规则")
            return {"selected_rules": seq, "agent_trace": trace}
        except Exception as e:
            print(f"规则选择失败: {e}")
            return {
                "should_terminate": True,
                "agent_trace": trace + [{"node": "rule_selection", "error": str(e)}],
            }

    async def _rewrite_node(self, state: RewriteState) -> Dict[str, Any]:
        print("🔧 开始SQL重写...")
        trace = list(state.get("agent_trace") or [])
        try:
            base_sql = _state_sql(state.get("current_sql") or state["initial_sql"])
            sel = state.get("selected_rules") or {"applied_rules": [], "groups": ""}
            stats = _stats_str(state["data_statistics"])
            idx = state.get("index_info") or self.index_info
            async with self.llm_semaphore:
                rr = await self.rewrite_agent.rewrite_with_rule_sequence(
                    base_sql,
                    sel,
                    json.dumps(state.get("optimization_advice") or [], ensure_ascii=False),
                    stats,
                    schema_content=state.get("schema_content") or "",
                    index_info=idx,
                    previous_feedback=state.get("previous_feedback"),
                )
            rr = dict(rr or {})
            rr["rewritten_sql"] = _state_sql(rr.get("rewritten_sql", base_sql))
            trace.append({"node": "rewrite", "output": dict(rr)})
            if rr.get("parse_error") and rr.get("rewritten_sql") == base_sql:
                async with self.llm_semaphore:
                    fixed = await self.rewrite_agent.iterative_rewrite(
                        state["initial_sql"],
                        rr.get("error_info", "JSON 解析错误"),
                        rr,
                        data_statistics=stats,
                        index_info=idx,
                    )
                if fixed:
                    rr["rewritten_sql"] = _state_sql(fixed)
                    rr.pop("parse_error", None)
            raw_g = rr.get("semantic_correctness_guarantee")
            raw_s = rr.get("semantic_check")
            g = raw_g.strip() if isinstance(raw_g, str) else ""
            s = raw_s.strip() if isinstance(raw_s, str) else ""
            note_s = g or s
            return {
                "current_rewrite_result": rr,
                "semantic_check_note": note_s,
                "agent_trace": trace,
            }
        except Exception as e:
            print(f"重写失败: {e}")
            return {
                "should_terminate": True,
                "agent_trace": trace + [{"node": "rewrite", "error": str(e)}],
            }

    async def _syntax_check_node(self, state: RewriteState) -> Dict[str, Any]:
        print("🔍 开始语法检查...")
        trace = list(state.get("agent_trace") or [])
        rr = dict(state.get("current_rewrite_result") or {})
        base_sql = _state_sql(state.get("current_sql") or state["initial_sql"])
        sql = _state_sql(rr.get("rewritten_sql", base_sql))
        rr["rewritten_sql"] = sql
        stats = _stats_str(state["data_statistics"])
        idx = state.get("index_info") or self.index_info
        try:
            sc = await DBMS_Syntax_Tool(self.dbms, sql)
            ok = sc.get("flag", sc.get("valid", True))
            if ok:
                print("✅ 语法检查通过")
                trace.append({"node": "syntax_check", "output": {"valid": True}})
                return {"current_rewrite_result": rr, "agent_trace": trace}
            cur = sql
            for attempt in range(self.MAX_SYNTAX_FIX):
                sc2 = await DBMS_Syntax_Tool(self.dbms, cur)
                err = sc2.get("error", "Unknown")
                prev = dict(rr)
                prev["rewritten_sql"] = cur
                prev["error_info"] = err
                async with self.llm_semaphore:
                    cur = _state_sql(
                        await self.rewrite_agent.iterative_rewrite(
                            state["initial_sql"],
                            err,
                            prev,
                            data_statistics=stats,
                            index_info=idx,
                        )
                        or cur
                    )
                if cur and (await DBMS_Syntax_Tool(self.dbms, cur)).get("flag", True):
                    rr["rewritten_sql"] = cur
                    trace.append({"node": "syntax_check", "output": {"valid": True, "attempts": attempt + 1}})
                    return {"current_rewrite_result": rr, "agent_trace": trace}
            rr["rewritten_sql"] = _state_sql(state["initial_sql"])
            trace.append({"node": "syntax_check", "output": {"valid": False, "rolled_back": True}})
            return {"current_rewrite_result": rr, "agent_trace": trace}
        except Exception as e:
            print(f"语法检查异常: {e}")
            rr["rewritten_sql"] = _state_sql(state["initial_sql"])
            return {
                "current_rewrite_result": rr,
                "agent_trace": trace + [{"node": "syntax_check", "error": str(e)}],
            }

    async def _semantic_check_node(self, state: RewriteState) -> Dict[str, Any]:
        print("🧠 开始语义等价检查...")
        trace = list(state.get("agent_trace") or [])
        rr = dict(state.get("current_rewrite_result") or {})
        base_sql = _state_sql(state.get("current_sql") or state["initial_sql"])
        cur = _state_sql(rr.get("rewritten_sql", base_sql))
        rr["rewritten_sql"] = cur
        rules = rr.get("applied_rules") or (state.get("selected_rules") or {}).get("applied_rules", [])
        schema = (state.get("schema_content") or "").strip()
        stats = _stats_str(state["data_statistics"])
        idx = state.get("index_info") or self.index_info

        note_top = (state.get("semantic_check_note") or "").strip()
        note_g = rr.get("semantic_correctness_guarantee")
        note_g_s = note_g.strip() if isinstance(note_g, str) else ""
        note_rr = rr.get("semantic_check")
        note_rr_s = note_rr.strip() if isinstance(note_rr, str) else ""
        rewriter_guarantee = (note_top or note_g_s or note_rr_s) or None
        last_chk: Optional[Dict[str, Any]] = None
        last_rejected: Optional[str] = None
        last_reject_msg: Optional[str] = None
        last_reject_diffs: Optional[List[Dict[str, Any]]] = None

        async def _syntax_repair_after_semantic_fix(candidate_sql: str) -> Optional[str]:
            candidate = _state_sql(candidate_sql)
            for _s in range(self.MAX_SYNTAX_AFTER_SEMANTIC):
                syn = await DBMS_Syntax_Tool(self.dbms, candidate)
                if syn.get("flag", syn.get("valid", True)):
                    return _state_sql(candidate)
                err = syn.get("error", "")
                prev = dict(rr)
                prev["rewritten_sql"] = candidate
                prev["error_info"] = err
                async with self.llm_semaphore:
                    candidate = _state_sql(
                        await self.rewrite_agent.iterative_rewrite(
                            state["initial_sql"],
                            err,
                            prev,
                            data_statistics=stats,
                            index_info=idx,
                        )
                        or candidate
                    )
            return None

        # First use the deterministic equivalence checker. Only fall back to
        # LLM semantic audit when the tool cannot decide (UNKNOWN/timeout/error).
        deterministic_unknown = False
        for attempt_i in range(self.MAX_SEMANTIC_FIX):
            try:
                raw_eq = await Equivalence_Check_Tool(
                    state["initial_sql"],
                    cur,
                    state.get("schema_file", self.schema_file),
                    timeout=10,
                    verbose=False,
                )
            except Exception as ex:
                raw_eq = f"UNKNOWN_EXCEPTION: {ex}"

            parsed_eq = self._parse_equivalence_tool_result(raw_eq)
            status = parsed_eq["status"]
            reason = parsed_eq["reason"]
            tool_out = {
                "checker": "Equivalence_Check_Tool",
                "equivalent": status == "equivalent",
                "status": status,
                "message": reason,
                "attempt": attempt_i + 1,
            }
            trace.append({"node": "semantic_check", "output": tool_out})

            if status == "equivalent":
                rr["rewritten_sql"] = _state_sql(cur)
                print("✅ 确定性等价验证通过")
                return {
                    "current_rewrite_result": rr,
                    "semantic_equivalent": True,
                    "semantic_equivalence_message": reason,
                    "semantic_equivalence_differences": [],
                    "agent_trace": trace,
                }

            if status == "unknown":
                deterministic_unknown = True
                print(f"ℹ️ 确定性等价验证无法判定，转入 LLM 语义检查: {reason}")
                break

            diffs = [
                {
                    "type": "确定性等价验证失败",
                    "description": reason,
                    "location": "Equivalence_Check_Tool",
                }
            ]
            last_chk = {"equivalent": False, "message": reason, "differences": diffs}
            last_rejected = cur
            last_reject_msg = reason
            last_reject_diffs = diffs
            print(f"❌ 确定性等价验证不通过: {reason}")

            async with self.llm_semaphore:
                fixed = await self.rewrite_agent.semantic_fix(
                    state["initial_sql"],
                    cur,
                    rules,
                    diffs,
                )
            if not fixed:
                break
            fixed_sql = await _syntax_repair_after_semantic_fix(fixed)
            if not fixed_sql:
                break
            cur = fixed_sql

        if not deterministic_unknown:
            rr["rewritten_sql"] = _state_sql(state["initial_sql"])
            rollback_out = {
                "equivalent": False,
                "rolled_back": True,
                "checker": "Equivalence_Check_Tool",
            }
            trace.append({"node": "semantic_check", "output": rollback_out})
            return {
                "current_rewrite_result": rr,
                "semantic_equivalent": False,
                "semantic_equivalence_message": (last_chk or {}).get("message", "") or "",
                "semantic_equivalence_differences": (last_chk or {}).get("differences") or [],
                "agent_trace": trace,
            }

        for attempt_i in range(self.MAX_SEMANTIC_FIX):
            async with self.llm_semaphore:
                chk = await self.semantic_check_agent.check_equivalence(
                    state["initial_sql"],
                    cur,
                    rules,
                    semantic_correctness_guarantee=rewriter_guarantee,
                    schema_content=schema,
                    index_info=idx,
                    semantic_check_attempt=attempt_i,
                    last_rejected_rewritten_sql=last_rejected,
                    last_rejection_message=last_reject_msg,
                    last_rejection_differences=last_reject_diffs,
                )
            last_chk = chk
            trace.append({"node": "semantic_check", "output": chk})
            if chk.get("equivalent"):
                rr["rewritten_sql"] = _state_sql(cur)
                print("✅ 语义等价")
                return {
                    "current_rewrite_result": rr,
                    "semantic_equivalent": True,
                    "semantic_equivalence_message": chk.get("message", "") or "",
                    "semantic_equivalence_differences": chk.get("differences") or [],
                    "agent_trace": trace,
                }
            print(f"❌ 语义不等价: {chk.get('message', '')}")
            last_rejected = cur
            last_reject_msg = chk.get("message") or None
            last_reject_diffs = chk.get("differences") or None
            async with self.llm_semaphore:
                fixed = await self.rewrite_agent.semantic_fix(
                    state["initial_sql"],
                    cur,
                    rules,
                    chk.get("differences") or [],
                )
            if not fixed:
                break
            fixed_sql = await _syntax_repair_after_semantic_fix(fixed)
            if not fixed_sql:
                break
            cur = fixed_sql

        rr["rewritten_sql"] = _state_sql(state["initial_sql"])
        rollback_out = {"equivalent": False, "rolled_back": True}
        trace.append({"node": "semantic_check", "output": rollback_out})
        return {
            "current_rewrite_result": rr,
            "semantic_equivalent": False,
            "semantic_equivalence_message": (last_chk or {}).get("message", "") or "",
            "semantic_equivalence_differences": (last_chk or {}).get("differences") or [],
            "agent_trace": trace,
        }

    async def _evaluation_node(self, state: RewriteState) -> Dict[str, Any]:
        print("📊 开始评估优化结果...")
        trace = list(state.get("agent_trace") or [])
        if not state.get("can_optimize"):
            return {"should_terminate": True, "agent_trace": trace}

        init_sql = _state_sql(state["initial_sql"])
        rr = dict(state.get("current_rewrite_result") or {})
        rw = _state_sql(rr.get("rewritten_sql", init_sql))
        rr["rewritten_sql"] = rw
        sel = state.get("selected_rules") or {}
        groups = sel.get("groups", "")
        applied = sel.get("applied_rules", [])

        try:
            async def _raw_exp(sql: str):
                async with self.db_semaphore:
                    return await DBMS_RAW_EXPLAIN_JSON_Tool(self.dbms, sql)

            o_exp, r_exp = await asyncio.gather(_raw_exp(init_sql), _raw_exp(rw))
            rnd = state.get("current_round", 1)
            oc = self._extract_cost_from_explain(o_exp)
            rc = self._extract_cost_from_explain(r_exp)

            try:
                o_parsed = json.loads(o_exp) if isinstance(o_exp, str) else o_exp
                r_parsed = json.loads(r_exp) if isinstance(r_exp, str) else r_exp
            except (json.JSONDecodeError, TypeError):
                o_parsed, r_parsed = o_exp, r_exp
            plan_struct = compare_explain_plan_structures(o_parsed, r_parsed)
            reuse_signal = self._analyze_reuse_positive_signals(
                init_sql,
                rw,
                o_parsed,
                r_parsed,
            )
            if reuse_signal.get("positive"):
                plan_struct["reuse_positive"] = True
                plan_struct["reuse_signal"] = reuse_signal
                plan_struct["reasons"] = list(plan_struct.get("reasons") or []) + list(
                    reuse_signal.get("reasons") or []
                )
            preserve_despite_higher_cost = bool(
                plan_struct.get("preserve_despite_higher_cost")
            )
            if reuse_signal.get("positive"):
                preserve_despite_higher_cost = True
                plan_struct["preserve_despite_higher_cost"] = True
            rows_drastically_reduced = bool(
                plan_struct.get("rows_drastically_reduced")
            )
            if plan_struct.get("ok") and plan_struct.get("reasons"):
                print("📐 计划结构: " + " | ".join(plan_struct["reasons"]))

            cost_reduction_ratio = (oc - rc) / oc if oc > 0 else 0.0
            cost_increase_pct = ((rc - oc) / oc * 100.0) if oc > 0 else 0.0
            cost_exploded = bool(
                rw != init_sql
                and oc > 0
                and cost_increase_pct > self.COST_EXPLOSION_PCT
            )
            if cost_exploded:
                if rnd < self.MAX_ITERATION_LOOP:
                    print(
                        f"⚠️ 检测到成本爆炸：重写 cost 增幅 {cost_increase_pct:.1f}% "
                        f"(> {self.COST_EXPLOSION_PCT:.0f}%)，判定当前重写性能更差，进入下一轮优化。"
                    )
                    ev = {
                        "terminate": False,
                        "reason": (
                            f"重写 SQL 的估计 cost 增幅达到 {cost_increase_pct:.1f}%（超过 {self.COST_EXPLOSION_PCT:.0f}% 阈值），"
                            "按强负规则判定当前重写性能更差，需要继续优化。"
                        ),
                        "no_further_optimization": False,
                        "next_step_advice": (
                            "下一轮请回退激进改写，优先恢复关键过滤与连接路径，"
                            "避免引入导致中间结果爆炸的 CTE/JOIN 结构。"
                        ),
                    }
                else:
                    print(
                        f"⚠️ 检测到成本爆炸：重写 cost 增幅 {cost_increase_pct:.1f}% "
                        f"(> {self.COST_EXPLOSION_PCT:.0f}%)，达到最大轮次后终止。"
                    )
                    ev = {
                        "terminate": True,
                        "reason": (
                            f"重写 SQL 的估计 cost 增幅达到 {cost_increase_pct:.1f}%（超过 {self.COST_EXPLOSION_PCT:.0f}% 阈值），"
                            "判定当前重写性能显著更差；已达到最大轮次，终止优化。"
                        ),
                        "no_further_optimization": True,
                        "next_step_advice": "无需下一步优化。",
                    }
                trace.append(
                    {
                        "node": "evaluation",
                        "output": ev,
                        "skipped_llm": True,
                        "plan_structure_analysis": plan_struct,
                    }
                )
            elif rows_drastically_reduced and rw != init_sql:
                row_ratio = plan_struct.get("rows_ratio")
                row_pct = (float(row_ratio) * 100.0) if isinstance(row_ratio, (int, float)) else 0.0
                print(
                    f"✅ 行数估计求和大幅下降至原的 {row_pct:.4f}%，"
                    "直接判定优化成功，终止并保留重写 SQL"
                )
                ev = {
                    "terminate": True,
                    "reason": (
                        f"行数估计求和（Plan Rows累加）大幅下降，重写侧约为原的 {row_pct:.4f}%，"
                        "按强规则直接判定为优化成功：终止多轮优化并保留本轮重写 SQL。"
                    ),
                    "no_further_optimization": True,
                    "next_step_advice": "无需下一步优化。",
                }
                trace.append(
                    {
                        "node": "evaluation",
                        "output": ev,
                        "skipped_llm": True,
                        "plan_structure_analysis": plan_struct,
                    }
                )
            elif (
                oc > 0
                and rw != init_sql
                and rc < oc
                and cost_reduction_ratio > self.EARLY_TERMINATE_COST_REDUCTION_RATIO
            ):
                pct = cost_reduction_ratio * 100.0
                thr_pct = self.EARLY_TERMINATE_COST_REDUCTION_RATIO * 100.0
                print(
                    f"✅ 估计代价降幅 {pct:.1f}% 超过 {thr_pct:.0f}%，跳过 LLM 评估，直接终止并保留重写 SQL"
                )
                ev = {
                    "terminate": True,
                    "reason": (
                        f"估计代价降幅 {pct:.1f}% 超过 {thr_pct:.0f}%，"
                        "按规则跳过模型评估：直接终止多轮优化并保留本轮重写 SQL。"
                    ),
                    "no_further_optimization": True,
                    "next_step_advice": "无需下一步优化。",
                }
                trace.append(
                    {
                        "node": "evaluation",
                        "output": ev,
                        "skipped_llm": True,
                        "plan_structure_analysis": plan_struct,
                    }
                )
            else:
                info = {
                    "original_costs": oc,
                    "rewritten_costs": rc,
                    "original_explain_plan_json": o_exp,
                    "rewritten_explain_plan_json": r_exp,
                    "plan_structure_analysis": plan_struct,
                    "groups": groups,
                    "applied_rules": applied,
                    "optimization_advice": state.get("optimization_advice") or [],
                    "original_sql": init_sql,
                    "rewritten_sql": rw,
                    "reason": (state.get("previous_feedback") or {}).get("reason", ""),
                }
                async with self.llm_semaphore:
                    ev = await self.decision_agent.evaluate_with_costs(info, rnd)
                trace.append(
                    {
                        "node": "evaluation",
                        "output": ev,
                        "plan_structure_analysis": plan_struct,
                    }
                )
            terminate = ev.get("terminate", True)
            no_further_optimization = bool(ev.get("no_further_optimization", False))
            eval_reason = str(ev.get("reason", "") or "")
            next_step_advice = str(ev.get("next_step_advice", "") or "").strip()
            uncertain_markers = (
                "不确定",
                "无法高置信",
                "证据冲突",
                "无法判断",
                "uncertain",
            )
            is_uncertain = any(marker in eval_reason for marker in uncertain_markers)
            is_near_equivalent = self._is_near_equivalent_plan_and_cost(plan_struct, oc, rc)
            if is_near_equivalent:
                print("ℹ️ 计划与代价接近：倾向保留当前重写 SQL。")
            reuse_positive = bool(plan_struct.get("reuse_positive"))
            if reuse_positive:
                print("ℹ️ 检测到去相关/消除重复执行/公共结果复用：倾向保留当前重写 SQL。")

            structural_improved = bool(plan_struct.get("structural_improvement"))
            improved_now = bool(
                rw != init_sql and (
                    rc < oc
                    or structural_improved
                    or preserve_despite_higher_cost
                    or reuse_positive
                )
            )
            not_improved_or_degraded = not improved_now
            if not_improved_or_degraded and not no_further_optimization and rnd < self.MAX_ITERATION_LOOP:
                terminate = False
                ev["terminate"] = False
                if not next_step_advice:
                    reasons = plan_struct.get("reasons") if isinstance(plan_struct, dict) else []
                    reasons_text = "；".join(reasons[:3]) if reasons else ""
                    next_step_advice = (
                        "下一轮请优先修复导致未改进/恶化的关键结构："
                        "尽量恢复或增强索引访问路径，减少大范围扫描/高开销算子，"
                        "并针对相关子查询与JOIN顺序做更保守改写。"
                        + (f" 参考当前结构线索：{reasons_text}" if reasons_text else "")
                    )
                ev["next_step_advice"] = next_step_advice
                print("ℹ️ 本轮未改进/恶化：不终止，进入下一轮重写。")

            # Explicitly carry the cost-explosion signal forward for next-round agents.
            if cost_exploded:
                ev["cost_explosion"] = True
                ev["cost_increase_pct"] = round(cost_increase_pct, 2)
                ev["cost_explosion_threshold_pct"] = self.COST_EXPLOSION_PCT

            # 仅当「明确下一轮有望降低执行时间」时才续轮：未保留本轮重写时不允许进入下一轮
            if not terminate:
                print("ℹ️ 评估建议继续下一轮优化（terminate=false）")

            best_cost = state.get("best_cost")
            best_sql = state.get("best_sql")
            if best_sql:
                best_sql = _state_sql(best_sql)
            best_rules = state.get("best_rules")
            best_groups = state.get("best_groups", "")
            if (
                rw != init_sql
                and rc < (best_cost if best_cost is not None else float("inf"))
            ):
                best_cost, best_sql, best_rules, best_groups = rc, rw, applied, groups

            out: Dict[str, Any] = {
                "final_original_costs": oc,
                "final_rewritten_costs": rc,
                "rewritten_sql": rw,
                "best_cost": best_cost,
                "best_sql": best_sql,
                "best_rules": best_rules,
                "best_groups": best_groups,
                "evaluation_reason": ev.get("reason", "") or "",
                "evaluation_terminate": terminate,
                "evaluation_no_further_optimization": no_further_optimization,
                "evaluation_next_step_advice": next_step_advice,
                "agent_trace": trace,
                "current_sql": rw,
            }

            if terminate:
                out["should_terminate"] = True
                if rw != init_sql:
                    if is_uncertain:
                        print("ℹ️ 评估不确定：保留重写 SQL，但记忆已关闭，跳过记忆更新。")
                out["rewritten_sql"] = _state_sql(out.get("rewritten_sql", rw))
                cr = dict(out.get("current_rewrite_result") or rr)
                cr["rewritten_sql"] = out["rewritten_sql"]
                out["current_rewrite_result"] = cr
                return out

            nxt = rnd + 1
            if nxt > self.MAX_ITERATION_LOOP:
                out["should_terminate"] = True
                ratio = ((rc - oc) / oc * 100) if oc > 0 else 0.0
                cost_regress = ratio > self.COST_ROLLBACK_PCT
                if cost_regress and preserve_despite_higher_cost:
                    print(
                        f"ℹ️  估计代价上升 {ratio:.1f}%（> {self.COST_ROLLBACK_PCT}%），"
                        "计划结构分析建议保留当前重写，不回退。"
                    )
                elif cost_regress:
                    print(
                        f"ℹ️  估计代价上升 {ratio:.1f}%（> {self.COST_ROLLBACK_PCT}%），"
                        "但已取消回退逻辑，保留当前重写 SQL 作为最终输出。"
                    )
                if rw != init_sql:
                    if is_uncertain:
                        print("ℹ️ 评估不确定：保留重写 SQL，但记忆已关闭，跳过记忆更新。")
                out["rewritten_sql"] = _state_sql(out.get("rewritten_sql", rw))
                cr = dict(out.get("current_rewrite_result") or rr)
                cr["rewritten_sql"] = out["rewritten_sql"]
                out["current_rewrite_result"] = cr
                return out

            out["should_terminate"] = False
            out["current_round"] = nxt
            out["previous_feedback"] = {
                "reason": ev.get("reason", ""),
                "problematic_rules": applied,
                "next_step_advice": next_step_advice,
                "status": "not_improved_or_degraded" if not_improved_or_degraded else "improved_but_continue",
                "cost_explosion": bool(ev.get("cost_explosion", False)),
                "cost_increase_pct": ev.get("cost_increase_pct"),
                "cost_explosion_threshold_pct": ev.get("cost_explosion_threshold_pct"),
                "original_cost": oc,
                "rewritten_cost": rc,
            }
            out["rewritten_sql"] = _state_sql(out.get("rewritten_sql", rw))
            cr = dict(out.get("current_rewrite_result") or rr)
            cr["rewritten_sql"] = out["rewritten_sql"]
            out["current_rewrite_result"] = cr
            return out
        except Exception as e:
            print(f"评估失败: {e}")
            return {"should_terminate": True, "agent_trace": trace + [{"node": "evaluation", "error": str(e)}]}

    def _format_final_output(self, state: RewriteState) -> Dict[str, Any]:
        oc = state.get("final_original_costs", state.get("original_cost", 0.0))
        if not state.get("can_optimize"):
            final_sql = _state_sql(state.get("current_sql") or state.get("rewritten_sql") or state["initial_sql"])
            rc = state.get("final_rewritten_costs", state.get("current_cost", oc))
            return {
                "tpch": [
                    {
                        "rewritten_query": final_sql,
                        "original_costs": oc,
                        "rewrite_costs": rc,
                        "costs_reduction_rate": round(((oc - rc) / oc * 100) if oc > 0 else 0.0, 2),
                        "rewrite_rules": None,
                    }
                ],
                "rewrite_suggestion": [],
                "agent_trace": state.get("agent_trace", []),
            }
        final_sql = _state_sql(state.get("rewritten_sql") or state["initial_sql"])
        if state.get("current_rewrite_result"):
            final_sql = _state_sql(state["current_rewrite_result"].get("rewritten_sql", final_sql))
        rc = state.get("final_rewritten_costs", oc)
        rate = round(((oc - rc) / oc * 100) if oc > 0 else 0.0, 2)
        sel = state.get("selected_rules") or {}
        rules = sel.get("applied_rules")
        if not rules:
            rules = None
        return {
            "tpch": [
                {
                    "rewritten_query": final_sql,
                    "original_costs": oc,
                    "rewrite_costs": rc,
                    "costs_reduction_rate": rate,
                    "rewrite_rules": rules,
                }
            ],
            "rewrite_suggestion": state.get("optimization_advice") or [],
            "agent_trace": state.get("agent_trace", []),
        }

    async def run(self, initial_sql: str) -> Dict[str, Any]:
        initial_sql = _state_sql(initial_sql)
        schema_content = build_filtered_schema_content(self.schema_file, initial_sql)
        data_stats = filter_data_statistics_for_sql(self.data_statistics, initial_sql)
        memory = create_memory_buffer(
            data_stats, self.schema_file, schema_content=schema_content
        )
        memory.initial_sql = initial_sql
        collector = OutputCollector()
        collector.start_collecting()
        st: RewriteState = {
            "initial_sql": initial_sql,
            "current_sql": initial_sql,
            "data_statistics": data_stats,
            "index_info": self.index_info,
            "schema_file": self.schema_file,
            "schema_content": schema_content,
            "max_iterations": self.MAX_ITERATION_LOOP,
            "current_round": 1,
            "can_optimize": False,
            "should_terminate": False,
            "optimization_advice": [],
            "selected_rules": None,
            "current_rewrite_result": None,
            "initial_optimization_reason": "",
            "semantic_check_note": "",
            "semantic_equivalent": None,
            "semantic_equivalence_message": "",
            "semantic_equivalence_differences": [],
            "initial_explain_info": "",
            "evaluation_reason": "",
            "evaluation_terminate": True,
            "evaluation_no_further_optimization": False,
            "evaluation_next_step_advice": "",
            "original_cost": 0.0,
            "current_cost": 0.0,
            "best_cost": None,
            "best_sql": None,
            "best_rules": None,
            "best_groups": "",
            "previous_feedback": None,
            "few_shot_examples": [],
            "retrieved_record_id": None,
            "final_original_costs": 0.0,
            "final_rewritten_costs": 0.0,
            "rewritten_sql": initial_sql,
            "agent_trace": [],
            "memory": memory,
            "output_collector": collector,
        }
        try:
            final = await self.graph.ainvoke(st)
            final["terminal_output"] = collector.stop_collecting()
            fo = self._format_final_output(final)
            if isinstance(fo, dict):
                fo["terminal_output"] = final.get("terminal_output", "")
            return fo
        except Exception as e:
            print(f"LangGraph 执行失败: {e}")
            import traceback

            traceback.print_exc()
            collector.stop_collecting()
            oc = st.get("final_original_costs", 0.0)
            return {
                "tpch": [
                    {
                        "rewritten_query": _state_sql(st.get("current_sql") or initial_sql),
                        "original_costs": oc,
                        "rewrite_costs": st.get("final_rewritten_costs", oc),
                        "costs_reduction_rate": 0,
                        "rewrite_rules": None,
                    }
                ],
                "rewrite_suggestion": [],
            }
