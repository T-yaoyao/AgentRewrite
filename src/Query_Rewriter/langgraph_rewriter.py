"""
LangGraph 编排的查询重写管线：
initial_check（PlanAnalyzer）→ rule_selection → rewrite → syntax_check → semantic_check → evaluation
→（有条件）uct_learning → END。UCT 学习更新使用纯数学公式（基于代价降低率）。
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np

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
    EvaluationAgent,
    ReasoningAgent,
    RewriteAgent,
    SemanticCheckAgent,
    get_all_rules,
)
from src.Query_Rewriter.global_memory import GlobalMemoryManager
from src.Query_Rewriter.schema_context import (
    build_filtered_schema_content,
    filter_data_statistics_for_sql,
)
from src.Query_Rewriter.rule_bandit import get_rule_bandit
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
    evaluation_keep: bool
    evaluation_next_step_advice: str

    original_cost: float
    current_cost: float
    best_cost: Optional[float]
    best_sql: Optional[str]
    best_rules: Optional[List]

    previous_feedback: Optional[Dict]

    few_shot_examples: List
    retrieved_record_id: Optional[str]

    # 评估节点写入，供 uct_learning 节点做数学更新；须显式声明否则 LangGraph 可能丢弃
    _uct_update_info: Optional[Dict[str, Any]]

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


def _sql_identity(sql: Any) -> str:
    """比较两条 SQL 是否为同一文本，忽略空白和末尾分号。"""
    return re.sub(r"\s+", " ", _state_sql(sql)).strip().rstrip(";").strip()


def align_step_sqls(
    applied_rules: List[str],
    rewritten_sql: str,
    raw_steps: Any,
) -> Optional[List[str]]:
    """逐步 SQL 必须与规则序列等长，且最后一项就是最终重写。"""
    if not applied_rules:
        return []
    if not isinstance(raw_steps, list) or len(raw_steps) != len(applied_rules):
        return None
    steps: List[str] = []
    for item in raw_steps:
        if not isinstance(item, str) or not item.strip():
            return None
        steps.append(_state_sql(item))
    if _sql_identity(steps[-1]) != _sql_identity(rewritten_sql):
        return None
    steps[-1] = _state_sql(rewritten_sql)
    return steps


def marginal_cost_rewards(costs: List[float]) -> Optional[List[float]]:
    """r_i = clip((C_{i-1} - C_i) / C_0, -1, 1)。C_0 是本轮规则作用前的代价。"""
    if len(costs) < 2 or costs[0] <= 0:
        return None
    rewards: List[float] = []
    for prev, cur in zip(costs, costs[1:]):
        raw = (float(prev) - float(cur)) / float(costs[0])
        rewards.append(float(max(-1.0, min(1.0, raw))))
    return rewards


def equal_cost_rewards(original_cost: float, rewritten_cost: float, n: int) -> Optional[List[float]]:
    """逐步代价不可用时，把序列代价收益均分给 n 条规则。"""
    if n <= 0 or original_cost <= 0:
        return None
    total = (float(original_cost) - float(rewritten_cost)) / float(original_cost)
    share = float(max(-1.0, min(1.0, total)) / n)
    return [share] * n


def _bind_step_chain(rr: Dict[str, Any], final_sql: str) -> None:
    """保留已有逐步 SQL，并把最后一步改成实际留下的 SQL。"""
    final_sql = _state_sql(final_sql)
    rr["rewritten_sql"] = final_sql
    steps = rr.get("step_sqls")
    if isinstance(steps, list) and steps and all(isinstance(s, str) and s.strip() for s in steps):
        bound = [_state_sql(s) for s in steps]
        bound[-1] = final_sql
        rr["step_sqls"] = bound
    else:
        rr["step_sqls"] = None


def _rollback_unfixed_rewrite(rr: Dict[str, Any], original_sql: str) -> None:
    """语法或语义修不好时退回原始 SQL，并禁止本轮写入 LinUCB。"""
    rr["rewritten_sql"] = _state_sql(original_sql)
    rr["step_sqls"] = None
    rr["skip_linucb"] = True


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
    # Reward fusion: combine optimizer cost signal with plan-structure signal.
    REWARD_COST_WEIGHT = 0.60
    REWARD_ROWS_WEIGHT = 0.20
    REWARD_DEPTH_WEIGHT = 0.20
    # Keep rewrite when cost/plan are near-equivalent to avoid over-rollback.
    NEAR_EQUAL_COST_PCT = 0.10
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
        self.evaluation_agent = EvaluationAgent(message_queue)
        self.rewrite_agent = RewriteAgent(message_queue)
        self.semantic_check_agent = SemanticCheckAgent(message_queue)
        self.evaluation_agent.watch(["ReasoningAgent", "ExplainAgent"])

        self.llm_semaphore = asyncio.Semaphore(3)
        self.db_semaphore = asyncio.Semaphore(5)

        try:
            self.global_memory = GlobalMemoryManager()
            print("✅ Global memory manager initialized")
        except Exception as e:
            print(f"⚠️ Global memory unavailable: {e}")
            self.global_memory = None
        
        # Initialize UCT bandit components
        self.bandit = get_rule_bandit()
        print("✅ UCT Bandit initialized (math reward mode)")

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
        g.add_node("uct_learning", self._uct_learning_node)
        g.add_conditional_edges(
            "evaluation",
            self._route_after_eval,
            {"again": "initial_check", "uct": "uct_learning", "end": END},
        )
        g.add_edge("uct_learning", END)
        return g.compile()

    def _route_after_initial(self, state: RewriteState) -> str:
        return "continue" if state.get("can_optimize") else "end"

    def _route_after_eval(self, state: RewriteState) -> str:
        if not state.get("should_terminate", True):
            return "again"
        if state.get("_uct_update_info"):
            return "uct"
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

    def _uct_update_payload(
        self,
        sel: Dict[str, Any],
        applied: List[str],
        init_sql: str,
        rw: str,
        o_exp: Any,
        r_exp: Any,
        oc: float,
        rc: float,
        rule_step_rewards: List[float],
    ) -> Dict[str, Any]:
        return {
            "applied_rules": applied,
            "chosen_rule_prior": sel.get("chosen_rule_prior") or [],
            "rule_step_rewards": list(rule_step_rewards),
            "context_vector": sel.get("_context_vector"),
            "original_cost": oc,
            "rewritten_cost": rc,
            "original_sql": init_sql,
            "rewritten_sql": rw,
            "original_explain": o_exp,
            "rewritten_explain": r_exp,
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

    def _plan_analyzer_report(self, explain_payload: Any) -> str:
        """PlanAnalyzer text for every node. Raw EXPLAIN is not included."""
        from src.Rewrite_Middleware.plan_analyzer import PlanAnalyzer

        if explain_payload is None:
            return "计划分析失败：没有执行计划。"
        if isinstance(explain_payload, str):
            text = explain_payload
        else:
            text = json.dumps(explain_payload, ensure_ascii=False)
        try:
            db_name = self.dbms.db_name if hasattr(self.dbms, "db_name") else None
            analyzer = PlanAnalyzer(explain_json_str=text, db_name=db_name)
            return analyzer.format_all_nodes_report()
        except Exception as e:
            return f"计划分析失败：{e}"

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

    def _paper_sequence_reward(self, oc: float, rc: float, original_explain: Any, rewritten_explain: Any) -> float:
        """融合收益 R = 0.6 R_cost + 0.2 R_rows + 0.2 R_depth，三项均为对称 clip。"""
        cost_reward = self.bandit.compute_sequence_reward(oc, rc)
        detail = self._compute_structure_reward(original_explain, rewritten_explain)
        o_rows = float(detail.get("o_rows", 0.0))
        r_rows = float(detail.get("r_rows", 0.0))
        o_depth = float(detail.get("o_depth", 0.0))
        r_depth = float(detail.get("r_depth", 0.0))
        if o_rows > 0:
            rows_reward = max(-1.0, min(1.0, (o_rows - r_rows) / o_rows))
        else:
            rows_reward = 0.0
        if o_depth > 0:
            depth_reward = max(-1.0, min(1.0, (o_depth - r_depth) / o_depth))
        else:
            depth_reward = 0.0
        reward = (
            self.REWARD_COST_WEIGHT * cost_reward
            + self.REWARD_ROWS_WEIGHT * rows_reward
            + self.REWARD_DEPTH_WEIGHT * depth_reward
        )
        return float(max(-1.0, min(1.0, reward)))

    def _store_positive_memory(
        self,
        original_sql: str,
        rewritten_sql: str,
        rule_sequence: List[str],
        original_cost: float,
        rewritten_cost: float,
        original_explain: Any,
        rewritten_explain: Any,
    ) -> None:
        """论文融合收益 R>0，或估计代价下降时，写入案例向量库。"""
        if not self.global_memory:
            return
        reward = self._paper_sequence_reward(
            original_cost, rewritten_cost, original_explain, rewritten_explain
        )
        cost_decreased = (
            isinstance(original_cost, (int, float))
            and not isinstance(original_cost, bool)
            and isinstance(rewritten_cost, (int, float))
            and not isinstance(rewritten_cost, bool)
            and rewritten_cost < original_cost
        )
        if reward <= 0.0 and not cost_decreased:
            print(f"ℹ️ 记忆跳过：论文收益 R={reward:+.4f}，且估计代价未下降")
            return
        try:
            self.global_memory.store_successful_optimization(
                original_sql=original_sql,
                rewritten_sql=rewritten_sql,
                rule_sequence=rule_sequence,
                original_cost=original_cost,
                rewritten_cost=rewritten_cost,
            )
        except Exception as ex:
            print(f"⚠️ 知识库存储失败: {ex}")

    def _should_write_uct_sample(
        self,
        oc: float,
        rc: float,
        rows_score: float,
        depth_score: float,
    ) -> bool:
        """
        Write learning sample iff cost/rows/depth are not all negative.
        Negative means strictly < 0.
        """
        rcost = self.bandit.compute_sequence_reward(oc, rc)
        rrows = float(rows_score)
        rdepth = float(depth_score)
        return not (rcost < 0.0 and rrows < 0.0 and rdepth < 0.0)

    def _explain_has_cost(self, explain_result: Any) -> bool:
        if isinstance(explain_result, str):
            if re.search(r"执行计划总代价:\s*([0-9,]+\.?[0-9]*)", explain_result):
                return True
            try:
                explain_result = json.loads(explain_result)
            except json.JSONDecodeError:
                return False
        if isinstance(explain_result, dict):
            if "total_cost" in explain_result:
                return True
            plan = explain_result.get("Plan")
            return isinstance(plan, dict) and "Total Cost" in plan
        if isinstance(explain_result, list) and explain_result:
            plan = explain_result[0].get("Plan") if isinstance(explain_result[0], dict) else None
            return isinstance(plan, dict) and "Total Cost" in plan
        return False

    async def _measure_step_cost_rewards(
        self,
        base_sql: str,
        step_sqls: List[str],
        known_explains: Dict[str, Any],
    ) -> Optional[List[float]]:
        """对规则作用前的 SQL 和每一步 SQL 取 EXPLAIN，得到逐步代价差。"""
        chain = [_state_sql(base_sql)] + [_state_sql(sql) for sql in step_sqls]
        explains: List[Any] = [None] * len(chain)
        pending: List[int] = []
        for i, sql in enumerate(chain):
            cached = known_explains.get(_sql_identity(sql))
            if cached is not None:
                explains[i] = cached
            else:
                pending.append(i)

        if pending:
            async def _one(sql: str):
                async with self.db_semaphore:
                    return await DBMS_RAW_EXPLAIN_JSON_Tool(self.dbms, sql)

            fetched = await asyncio.gather(*[_one(chain[i]) for i in pending])
            for i, exp in zip(pending, fetched):
                explains[i] = exp

        costs: List[float] = []
        for exp in explains:
            if not self._explain_has_cost(exp):
                return None
            costs.append(self._extract_cost_from_explain(exp))
        return marginal_cost_rewards(costs)

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

    async def _apply_uct_bandit_update(self, state: RewriteState) -> None:
        """消费 evaluation 写入的 _uct_update_info，完成纯数学 LinUCB 更新。"""
        initial_sql = state["initial_sql"]
        uct_info = state.get("_uct_update_info")
        if not uct_info or not uct_info.get("applied_rules"):
            return
        uct_info = dict(uct_info)
        if not uct_info.get("context_vector"):
            expl = state.get("initial_explain_info") or ""
            ctx = self.bandit.extract_context(
                initial_sql, expl, state.get("data_statistics")
            )
            uct_info["context_vector"] = ctx.tolist()
        try:
            context = np.array(uct_info["context_vector"], dtype=np.float32)
            oc = uct_info["original_cost"]
            rc = uct_info["rewritten_cost"]
            applied = uct_info["applied_rules"]
            chosen_prior_list = uct_info.get("chosen_rule_prior") or []
            chosen_prior_map: Dict[str, float] = {}
            for item in chosen_prior_list:
                if not isinstance(item, dict):
                    continue
                rid = item.get("rule_id")
                pr = item.get("prior")
                if rid and isinstance(pr, (int, float)):
                    chosen_prior_map[str(rid)] = float(pr)
            step_rewards = uct_info.get("rule_step_rewards") or []
            if not isinstance(step_rewards, list) or len(step_rewards) != len(applied):
                print("ℹ️ 缺少逐步代价奖励，跳过 LinUCB 更新")
                return
            cost_reward = self.bandit.compute_sequence_reward(oc, rc)
            struct_detail = self._compute_structure_reward(
                uct_info.get("original_explain"),
                uct_info.get("rewritten_explain"),
            )
            rows_score = float(struct_detail.get("rows_score", 0.0))
            depth_score = float(struct_detail.get("depth_score", 0.0))
            print(
                f"\n🎲 UCT Math Update: {len(applied)} rules, "
                f"step_sum={sum(float(r) for r in step_rewards):+.4f} "
                f"(cost={cost_reward:+.4f}, rows={rows_score:+.4f}, depth={depth_score:+.4f}, "
                f"oc={oc:.4f}, rc={rc:.4f})"
            )
            print(
                "   结构分细节: "
                f"rows {struct_detail.get('o_rows', 0.0):.1f}->{struct_detail.get('r_rows', 0.0):.1f} "
                f"(score={struct_detail.get('rows_score', 0.0):+.3f}), "
                f"depth {struct_detail.get('o_depth', 0.0):.1f}->{struct_detail.get('r_depth', 0.0):.1f} "
                f"(score={struct_detail.get('depth_score', 0.0):+.3f})"
            )
            rule_rewards = self.bandit.update_with_rule_rewards(
                applied,
                context,
                [float(r) for r in step_rewards],
            )
            print("✅ UCT Update Complete:")
            for rule_id, reward in rule_rewards.items():
                stats = self.bandit.get_rule_stats(rule_id)
                if stats:
                    prior_str = (
                        f"{chosen_prior_map[rule_id]:.3f}"
                        if rule_id in chosen_prior_map
                        else "N/A"
                    )
                    print(
                        f"   → {rule_id}: P(s,a)={prior_str}, "
                        f"reward={reward:+.3f}, "
                        f"count={stats['count']}, avg={stats['avg_reward']:+.3f}"
                    )
        except Exception as be:
            print(f"⚠️ UCT Bandit update failed: {be}")

    async def _uct_learning_node(self, state: RewriteState) -> Dict[str, Any]:
        await self._apply_uct_bandit_update(state)
        return {}

    async def _initial_check_node(self, state: RewriteState) -> Dict[str, Any]:
        print("🔍 PlanAnalyzer 分析当前 SQL...")
        trace = list(state.get("agent_trace") or [])
        few_shot: List = []
        retrieved_id = None
        try:
            base_sql = _state_sql(state.get("current_sql") or state["initial_sql"])
            if self.global_memory:
                raw = self.global_memory.retrieve(state["initial_sql"], top_k=3)
                filtered = [r for r in raw if r.get("score", 0) >= self.MIN_SIMILARITY]
                if filtered:
                    filtered.sort(key=lambda x: x.get("score", 0), reverse=True)
                    few_shot = filtered[:3]
                    best = filtered[0]
                    if best.get("id"):
                        retrieved_id = best["id"]
                else:
                    print(
                        f"🧊 历史案例相似度较低，不使用 few-shot"
                    )
            async with self.db_semaphore:
                explain_info = await DBMS_EXPLAIN_Tool(self.dbms, base_sql)
            base_cost = self._extract_cost_from_explain(explain_info)
            explain_text = (
                json.dumps(explain_info, ensure_ascii=False)
                if not isinstance(explain_info, str)
                else explain_info
            )
            trace.append({"node": "initial_check", "output": {"plan_analyzer": True}})
            print("✅ 计划分析完成，进入规则选择")
            return {
                "can_optimize": True,
                "optimization_advice": [],
                "initial_optimization_reason": "",
                "initial_explain_info": explain_text,
                "original_cost": state.get("original_cost", base_cost) if state.get("original_cost") else base_cost,
                "current_cost": base_cost,
                "final_original_costs": state.get("final_original_costs", base_cost) if state.get("final_original_costs") else base_cost,
                "few_shot_examples": few_shot,
                "retrieved_record_id": retrieved_id,
                "agent_trace": trace,
                "should_terminate": False,
                "current_sql": base_sql,
                "rewritten_sql": base_sql,
                "final_rewritten_costs": state.get("final_rewritten_costs", 0.0),
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
        print(f"🎯 第{rnd}轮规则选择 (UCT-guided)...")
        trace = list(state.get("agent_trace") or [])
        try:
            base_sql = _state_sql(state.get("current_sql") or state["initial_sql"])
            lib = get_all_rules()
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
            
            # ====== Bandit scoring (single-pass generation + bandit sorting) ======
            context = self.bandit.extract_context(base_sql, explain_info, state.get("data_statistics"))
            scored_rules = self.bandit.score_rules(lib, context)
            
            # Build UCT-scored rule library for LLM
            uct_scored_lib = {
                rule_id: (desc, score) for rule_id, score, desc in scored_rules
            }
            
            print(f"📊 UCT scored {len(scored_rules)} rules")
            top_3 = scored_rules[:3]
            for rid, score, _ in top_3:
                print(f"   → {rid}: UCT={score:.3f}")
            
            async with self.llm_semaphore:
                seq = await self.reasoning_agent.select_rule_sequence(
                    base_sql,
                    uct_scored_lib,
                    stats,
                    explain_info,
                    rnd,
                    state.get("previous_feedback"),
                    state.get("few_shot_examples") or [],
                    index_info=idx,
                )
            
            # Store context for later bandit update
            seq["_context_vector"] = context.tolist()
            
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
            sel = state.get("selected_rules") or {"applied_rules": []}
            stats = _stats_str(state["data_statistics"])
            idx = state.get("index_info") or self.index_info
            async with self.llm_semaphore:
                rr = await self.rewrite_agent.rewrite_with_rule_sequence(
                    base_sql,
                    sel,
                    schema_content=state.get("schema_content") or "",
                    previous_feedback=state.get("previous_feedback"),
                )
            rr = dict(rr or {})
            rr["rewritten_sql"] = _state_sql(rr.get("rewritten_sql", base_sql))
            rr["rewrite_base_sql"] = base_sql
            rr["step_sqls"] = align_step_sqls(
                sel.get("applied_rules") or rr.get("applied_rules") or [],
                rr["rewritten_sql"],
                rr.get("step_sqls"),
            )
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
                    rr["step_sqls"] = None
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
                _bind_step_chain(rr, sql)
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
                    _bind_step_chain(rr, cur)
                    trace.append({"node": "syntax_check", "output": {"valid": True, "attempts": attempt + 1}})
                    return {"current_rewrite_result": rr, "agent_trace": trace}
            print("ℹ️ 语法修复失败，退回原始 SQL，不记 LinUCB")
            _rollback_unfixed_rewrite(rr, state["initial_sql"])
            trace.append({"node": "syntax_check", "output": {"valid": False, "rolled_back": True, "skip_linucb": True}})
            return {"current_rewrite_result": rr, "agent_trace": trace}
        except Exception as e:
            print(f"语法检查异常: {e}")
            _rollback_unfixed_rewrite(rr, state["initial_sql"])
            return {
                "current_rewrite_result": rr,
                "agent_trace": trace + [{"node": "syntax_check", "error": str(e)}],
            }

    async def _semantic_check_node(self, state: RewriteState) -> Dict[str, Any]:
        print("🧠 开始语义等价检查...")
        trace = list(state.get("agent_trace") or [])
        rr = dict(state.get("current_rewrite_result") or {})
        if rr.get("skip_linucb"):
            print("ℹ️ 语法修复已退回原始 SQL，不再做语义修复")
            return {
                "current_rewrite_result": rr,
                "semantic_equivalent": False,
                "semantic_equivalence_message": "语法修复失败，已退回原始 SQL",
                "semantic_equivalence_differences": [],
                "agent_trace": trace,
            }
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
                _bind_step_chain(rr, cur)
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
        else:
            # The loop repairs after each failed check, so the last repair has not been checked.
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
            if parsed_eq["status"] == "equivalent":
                _bind_step_chain(rr, cur)
                print("✅ 确定性等价验证通过")
                return {
                    "current_rewrite_result": rr,
                    "semantic_equivalent": True,
                    "semantic_equivalence_message": parsed_eq["reason"],
                    "semantic_equivalence_differences": [],
                    "agent_trace": trace,
                }
            if parsed_eq["status"] == "unknown":
                deterministic_unknown = True
                print(f"ℹ️ 确定性等价验证无法判定，转入 LLM 语义检查: {parsed_eq['reason']}")
            else:
                last_chk = {
                    "equivalent": False,
                    "message": parsed_eq["reason"],
                    "differences": [
                        {
                            "type": "确定性等价验证失败",
                            "description": parsed_eq["reason"],
                            "location": "Equivalence_Check_Tool",
                        }
                    ],
                }

        if not deterministic_unknown:
            print("ℹ️ 语义修复失败，退回原始 SQL，不记 LinUCB")
            _rollback_unfixed_rewrite(rr, state["initial_sql"])
            rollback_out = {
                "equivalent": False,
                "rolled_back": True,
                "skip_linucb": True,
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
                _bind_step_chain(rr, cur)
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
        else:
            async with self.llm_semaphore:
                chk = await self.semantic_check_agent.check_equivalence(
                    state["initial_sql"],
                    cur,
                    rules,
                    semantic_correctness_guarantee=rewriter_guarantee,
                    schema_content=schema,
                    index_info=idx,
                    semantic_check_attempt=self.MAX_SEMANTIC_FIX,
                    last_rejected_rewritten_sql=last_rejected,
                    last_rejection_message=last_reject_msg,
                    last_rejection_differences=last_reject_diffs,
                )
            last_chk = chk
            trace.append({"node": "semantic_check", "output": chk})
            if chk.get("equivalent"):
                _bind_step_chain(rr, cur)
                print("✅ 语义等价")
                return {
                    "current_rewrite_result": rr,
                    "semantic_equivalent": True,
                    "semantic_equivalence_message": chk.get("message", "") or "",
                    "semantic_equivalence_differences": chk.get("differences") or [],
                    "agent_trace": trace,
                }

        print("ℹ️ 语义修复失败，退回原始 SQL，不记 LinUCB")
        _rollback_unfixed_rewrite(rr, state["initial_sql"])
        rollback_out = {"equivalent": False, "rolled_back": True, "skip_linucb": True}
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
            if plan_struct.get("ok") and plan_struct.get("reasons"):
                print("📐 计划结构: " + " | ".join(plan_struct["reasons"]))

            cost_reduction_ratio = (oc - rc) / oc if oc > 0 else 0.0
            if (
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
                    "keep": True,
                    "terminate": True,
                    "reason": (
                        f"估计代价降幅 {pct:.1f}% 超过 {thr_pct:.0f}%，"
                        "按规则跳过模型评估：直接终止多轮优化并保留本轮重写 SQL。"
                    ),
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
                struct_for_prompt = self._compute_structure_reward(o_exp, r_exp)
                info = {
                    "original_costs": oc,
                    "rewritten_costs": rc,
                    "original_plan_analysis": self._plan_analyzer_report(o_exp),
                    "rewritten_plan_analysis": self._plan_analyzer_report(r_exp),
                    "original_scan_rows": f"{struct_for_prompt.get('o_rows', 0.0):.1f}",
                    "rewritten_scan_rows": f"{struct_for_prompt.get('r_rows', 0.0):.1f}",
                    "original_plan_depth": f"{struct_for_prompt.get('o_depth', 0.0):.1f}",
                    "rewritten_plan_depth": f"{struct_for_prompt.get('r_depth', 0.0):.1f}",
                    "applied_rules": applied,
                    "original_sql": init_sql,
                    "rewritten_sql": rw,
                    "reason": (state.get("previous_feedback") or {}).get("reason", ""),
                }
                async with self.llm_semaphore:
                    ev = await self.evaluation_agent.evaluate_with_costs(info, rnd)
                trace.append(
                    {
                        "node": "evaluation",
                        "output": ev,
                        "plan_structure_analysis": plan_struct,
                    }
                )
            terminate = bool(ev.get("terminate", True))
            keep = bool(ev.get("keep", False))
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
            if not terminate:
                print("ℹ️ 评估建议继续下一轮优化（terminate=false）")

            rule_step_rewards: Optional[List[float]] = None
            step_sqls = rr.get("step_sqls")
            will_record = terminate or (rnd + 1 > self.MAX_ITERATION_LOOP)
            can_attribute = (
                will_record
                and not rr.get("skip_linucb")
                and not is_uncertain
                and bool(applied)
                and rw != init_sql
                and isinstance(step_sqls, list)
                and len(step_sqls) == len(applied)
            )
            if can_attribute:
                base_sql = _state_sql(
                    rr.get("rewrite_base_sql") or state.get("current_sql") or init_sql
                )
                known_explains = {
                    _sql_identity(init_sql): o_exp,
                    _sql_identity(rw): r_exp,
                }
                rule_step_rewards = await self._measure_step_cost_rewards(
                    base_sql,
                    step_sqls,
                    known_explains,
                )
                if not rule_step_rewards or len(rule_step_rewards) != len(applied):
                    rule_step_rewards = equal_cost_rewards(oc, rc, len(applied))
                    if rule_step_rewards:
                        print(
                            "ℹ️ 逐步 EXPLAIN 代价不可用，LinUCB 改为均分 "
                            f"{rule_step_rewards[0]:+.4f} × {len(applied)}"
                        )
                    else:
                        print("ℹ️ 逐步 EXPLAIN 代价不可用，且总代价无法均分，跳过 LinUCB 更新")
            elif will_record and rr.get("skip_linucb"):
                print("ℹ️ 语法或语义修复失败，跳过 LinUCB 更新")
            elif will_record and applied and rw != init_sql and not is_uncertain:
                rule_step_rewards = equal_cost_rewards(oc, rc, len(applied))
                if rule_step_rewards:
                    print(
                        "ℹ️ 逐步 SQL 与规则序列不一致，LinUCB 改为均分 "
                        f"{rule_step_rewards[0]:+.4f} × {len(applied)}"
                    )
                else:
                    print("ℹ️ 逐步 SQL 与规则序列不一致，且总代价无法均分，跳过 LinUCB 更新")

            best_cost = state.get("best_cost")
            best_sql = state.get("best_sql")
            if best_sql:
                best_sql = _state_sql(best_sql)
            best_rules = state.get("best_rules")
            if (
                rw != init_sql
                and rc < (best_cost if best_cost is not None else float("inf"))
            ):
                best_cost, best_sql, best_rules = rc, rw, applied

            out: Dict[str, Any] = {
                "final_original_costs": oc,
                "final_rewritten_costs": rc,
                "rewritten_sql": rw,
                "best_cost": best_cost,
                "best_sql": best_sql,
                "best_rules": best_rules,
                "evaluation_reason": ev.get("reason", "") or "",
                "evaluation_terminate": terminate,
                "evaluation_keep": keep,
                "evaluation_next_step_advice": next_step_advice,
                "agent_trace": trace,
                "current_sql": rw,
            }

            if terminate:
                out["should_terminate"] = True
                if keep and rw != init_sql:
                    if is_uncertain:
                        print("ℹ️ 评估不确定：保留重写 SQL，但跳过记忆库与 UCT 更新。")
                    else:
                        self._store_positive_memory(
                            original_sql=init_sql,
                            rewritten_sql=out["rewritten_sql"],
                            rule_sequence=applied,
                            original_cost=oc,
                            rewritten_cost=rc,
                            original_explain=o_exp,
                            rewritten_explain=r_exp,
                        )
                    if not is_uncertain:
                        struct_detail_for_gate = self._compute_structure_reward(o_exp, r_exp)
                        should_write = self._should_write_uct_sample(
                            oc=oc,
                            rc=rc,
                            rows_score=float(struct_detail_for_gate.get("rows_score", 0.0)),
                            depth_score=float(struct_detail_for_gate.get("depth_score", 0.0)),
                        )
                        if applied and should_write and rule_step_rewards:
                            out["_uct_update_info"] = self._uct_update_payload(
                                sel,
                                applied,
                                init_sql,
                                rw,
                                o_exp,
                                r_exp,
                                oc,
                                rc,
                                rule_step_rewards,
                            )
                out["rewritten_sql"] = _state_sql(out.get("rewritten_sql", rw))
                cr = dict(out.get("current_rewrite_result") or rr)
                cr["rewritten_sql"] = out["rewritten_sql"]
                out["current_rewrite_result"] = cr
                return out

            nxt = rnd + 1
            if nxt > self.MAX_ITERATION_LOOP:
                out["should_terminate"] = True
                ratio = ((rc - oc) / oc * 100) if oc > 0 else 0.0
                if ratio > self.COST_ROLLBACK_PCT:
                    print(
                        f"ℹ️  估计代价上升 {ratio:.1f}%（> {self.COST_ROLLBACK_PCT}%）。"
                        "已到最大轮数，最终输出以评估是否保留为准。"
                    )
                if rw != init_sql:
                    if is_uncertain:
                        print("ℹ️ 评估不确定：保留重写 SQL，但跳过记忆库与 UCT 更新。")
                    else:
                        self._store_positive_memory(
                            original_sql=init_sql,
                            rewritten_sql=rw,
                            rule_sequence=applied,
                            original_cost=oc,
                            rewritten_cost=rc,
                            original_explain=o_exp,
                            rewritten_explain=r_exp,
                        )
                    if not is_uncertain:
                        struct_detail_for_gate = self._compute_structure_reward(o_exp, r_exp)
                        should_write = self._should_write_uct_sample(
                            oc=oc,
                            rc=rc,
                            rows_score=float(struct_detail_for_gate.get("rows_score", 0.0)),
                            depth_score=float(struct_detail_for_gate.get("depth_score", 0.0)),
                        )
                        if applied and should_write and rule_step_rewards:
                            out["_uct_update_info"] = self._uct_update_payload(
                                sel,
                                applied,
                                init_sql,
                                rw,
                                o_exp,
                                r_exp,
                                oc,
                                rc,
                                rule_step_rewards,
                            )
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
                "status": "agent_continue",
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
        init_sql = _state_sql(state["initial_sql"])
        kept_sql = _state_sql(state.get("rewritten_sql") or "")
        agent_kept = bool(state.get("evaluation_keep")) and bool(kept_sql) and kept_sql != init_sql
        if agent_kept:
            final_sql = kept_sql
            rc_raw = state.get("final_rewritten_costs", oc)
            rc = float(rc_raw) if isinstance(rc_raw, (int, float)) else 0.0
            sel = state.get("selected_rules") or {}
            rules = sel.get("applied_rules") or None
            print(
                f"✅ 评估保留本轮重写，输出该 SQL，估计代价 {rc:.4f}"
            )
        else:
            final_sql = init_sql
            rc = float(oc) if isinstance(oc, (int, float)) else 0.0
            rules = None
            print("ℹ️ 评估未保留重写，输出原 SQL")
        rate = round(((oc - rc) / oc * 100) if oc > 0 else 0.0, 2)
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
            "evaluation_keep": False,
            "evaluation_next_step_advice": "",
            "original_cost": 0.0,
            "current_cost": 0.0,
            "best_cost": None,
            "best_sql": None,
            "best_rules": None,
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
