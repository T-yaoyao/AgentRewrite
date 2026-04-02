"""
LangGraph 编排的查询重写管线：
initial_check → rule_selection → rewrite → syntax_check → semantic_check → evaluation
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

from src.Rewrite_Middleware.middleware import DBMS, DBMS_EXPLAIN_Tool, DBMS_Syntax_Tool
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
    get_rule_examples,
    get_rules_by_groups,
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
    evaluation_rollback_sql: bool

    original_cost: float
    current_cost: float
    best_cost: Optional[float]
    best_sql: Optional[str]
    best_rules: Optional[List]
    best_groups: str

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


class LangGraphQueryRewriter:
    MIN_SIMILARITY = 0.80
    HIGH_SIMILARITY = 0.90
    MAX_SYNTAX_FIX = 3
    MAX_SEMANTIC_FIX = 3
    MAX_SYNTAX_AFTER_SEMANTIC = 3
    COST_ROLLBACK_PCT = 20.0
    EFFECT_SCORE_LAMBDA_BASE = 0.5

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
            {"again": "rule_selection", "uct": "uct_learning", "end": END},
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
    ) -> Dict[str, Any]:
        return {
            "applied_rules": applied,
            "chosen_rule_prior": sel.get("chosen_rule_prior") or [],
            "rule_effect_scores": sel.get("rule_effect_scores") or {},
            "rule_effect_confidence": sel.get("rule_effect_confidence") or {},
            "context_vector": sel.get("_context_vector"),
            "original_cost": oc,
            "rewritten_cost": rc,
            "original_sql": init_sql,
            "rewritten_sql": rw,
            "original_explain": o_exp,
            "rewritten_explain": r_exp,
        }

    @staticmethod
    def _fallback_position_weights(applied: List[str], position_decay: float = 0.90) -> Dict[str, float]:
        if not applied:
            return {}
        if position_decay <= 0:
            position_decay = 1.0
        ws = np.array([position_decay ** i for i in range(len(applied))], dtype=np.float64)
        s = float(ws.sum()) if float(ws.sum()) > 0 else 1.0
        return {rid: float(ws[i] / s) for i, rid in enumerate(applied)}

    def _mix_llm_effect_with_confidence_gate(
        self,
        applied: List[str],
        effect_scores: Dict[str, float],
        effect_confidence: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Mixed weight per rule:
            w_i = lambda_i * w_i_llm + (1-lambda_i) * w_i_fallback
            lambda_i = lambda_base * confidence_i
        """
        if not applied:
            return {}
        fallback = self._fallback_position_weights(applied, position_decay=0.90)
        if not effect_scores:
            return fallback

        llm = {}
        llm_sum = 0.0
        for rid in applied:
            v = float(max(0.0, effect_scores.get(rid, 0.0)))
            llm[rid] = v
            llm_sum += v
        if llm_sum <= 0:
            return fallback
        llm = {rid: (llm[rid] / llm_sum) for rid in applied}

        mixed = {}
        for rid in applied:
            conf = float(effect_confidence.get(rid, 0.5))
            conf = max(0.0, min(1.0, conf))
            lambda_i = self.EFFECT_SCORE_LAMBDA_BASE * conf
            mixed[rid] = lambda_i * llm[rid] + (1.0 - lambda_i) * fallback[rid]

        total = sum(mixed.values())
        if total <= 0:
            return fallback
        return {rid: (mixed[rid] / total) for rid in applied}

    async def _apply_uct_bandit_update(self, state: RewriteState) -> None:
        """消费 evaluation 写入的 _uct_update_info，完成纯数学 LinUCB 更新。"""
        initial_sql = state["initial_sql"]
        uct_info = state.get("_uct_update_info")
        if not uct_info or not uct_info.get("applied_rules"):
            return
        uct_info = dict(uct_info)
        if not uct_info.get("context_vector"):
            expl = state.get("initial_explain_info") or ""
            adv_groups: List[str] = []
            for a in state.get("optimization_advice") or []:
                g = a.get("group")
                if g:
                    adv_groups.append(g)
            ctx = self.bandit.extract_context(initial_sql, expl, adv_groups)
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
            raw_effect_scores = uct_info.get("rule_effect_scores") or {}
            effect_scores: Dict[str, float] = {}
            if isinstance(raw_effect_scores, dict):
                for rid in applied:
                    v = raw_effect_scores.get(rid)
                    if isinstance(v, (int, float)) and float(v) > 0:
                        effect_scores[rid] = float(v)
            raw_effect_conf = uct_info.get("rule_effect_confidence") or {}
            effect_conf: Dict[str, float] = {}
            if isinstance(raw_effect_conf, dict):
                for rid in applied:
                    v = raw_effect_conf.get(rid)
                    if isinstance(v, (int, float)):
                        effect_conf[rid] = float(max(0.0, min(1.0, float(v))))
            mixed_weights = self._mix_llm_effect_with_confidence_gate(
                applied,
                effect_scores,
                effect_conf,
            )
            sequence_reward = self.bandit.compute_sequence_reward(oc, rc)
            print(
                f"\n🎲 UCT Math Update: {len(applied)} rules, "
                f"reward={sequence_reward:+.4f} (oc={oc:.4f}, rc={rc:.4f})"
            )
            rule_rewards = self.bandit.update_with_sequence_reward(
                applied,
                context,
                sequence_reward,
                rule_weights=mixed_weights,
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
                    effect_str = (
                        f"{effect_scores.get(rule_id, 0.0):.3f}"
                        if effect_scores
                        else "N/A"
                    )
                    conf_str = (
                        f"{effect_conf.get(rule_id, 0.5):.2f}"
                        if effect_conf
                        else "N/A"
                    )
                    gate_lambda = (
                        self.EFFECT_SCORE_LAMBDA_BASE * float(effect_conf.get(rule_id, 0.5))
                        if effect_conf
                        else 0.0
                    )
                    print(
                        f"   → {rule_id}: P(s,a)={prior_str}, effect={effect_str}, conf={conf_str}, "
                        f"lambda={gate_lambda:.2f}, reward={reward:+.3f}, "
                        f"count={stats['count']}, avg={stats['avg_reward']:+.3f}"
                    )
        except Exception as be:
            print(f"⚠️ UCT Bandit update failed: {be}")

    async def _uct_learning_node(self, state: RewriteState) -> Dict[str, Any]:
        await self._apply_uct_bandit_update(state)
        return {}

    async def _initial_check_node(self, state: RewriteState) -> Dict[str, Any]:
        print("🔍 开始初始优化可行性检查...")
        trace = list(state.get("agent_trace") or [])
        few_shot: List = []
        retrieved_id = None
        try:
            if self.global_memory:
                raw = self.global_memory.retrieve(state["initial_sql"], top_k=3)
                filtered = [r for r in raw if r.get("score", 0) >= self.MIN_SIMILARITY]
                if filtered:
                    filtered.sort(
                        key=lambda x: (x.get("cost_reduction_rate", 0), x.get("score", 0)),
                        reverse=True,
                    )
                    few_shot = filtered[:3]
                    best = filtered[0]
                    if best.get("id"):
                        self.global_memory.update_hit_frequency(best["id"])
                        retrieved_id = best["id"]
                else:
                    print(
                        f"🧊 历史案例相似度较低，不使用 few-shot"
                    )
            async with self.db_semaphore:
                explain_info = await DBMS_EXPLAIN_Tool(self.dbms, state["initial_sql"])
            oc = self._extract_cost_from_explain(explain_info)
            stats = _stats_str(state["data_statistics"])
            idx = state.get("index_info") or self.index_info
            async with self.llm_semaphore:
                check = await self.decision_agent.initial_optimization_check(
                    state["initial_sql"],
                    stats,
                    explain_info,
                    few_shot,
                    index_info=idx,
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
                "original_cost": oc,
                "final_original_costs": oc,
                "few_shot_examples": few_shot,
                "retrieved_record_id": retrieved_id,
                "agent_trace": trace,
                "should_terminate": not can,
                "rewritten_sql": state["initial_sql"],
                "final_rewritten_costs": oc if not can else state.get("final_rewritten_costs", 0.0),
            }
        except Exception as e:
            print(f"初始检查失败: {e}")
            import traceback

            traceback.print_exc()
            oc = state.get("original_cost", 0.0)
            return {
                "can_optimize": False,
                "should_terminate": True,
                "initial_optimization_reason": "",
                "initial_explain_info": "",
                "agent_trace": trace + [{"node": "initial_check", "error": str(e)}],
                "final_original_costs": oc,
                "final_rewritten_costs": oc,
                "rewritten_sql": state["initial_sql"],
            }

    async def _rule_selection_node(self, state: RewriteState) -> Dict[str, Any]:
        rnd = state.get("current_round", 1)
        print(f"🎯 第{rnd}轮规则选择 (UCT-guided)...")
        trace = list(state.get("agent_trace") or [])
        try:
            groups: List[str] = []
            for a in state.get("optimization_advice") or []:
                g = a.get("group")
                if g:
                    groups.append(g)
            lib = get_rules_by_groups(groups)
            stats = _stats_str(state["data_statistics"])
            idx = state.get("index_info") or self.index_info
            explain_info = state.get("initial_explain_info", "")
            if not explain_info:
                async with self.db_semaphore:
                    explain = await DBMS_EXPLAIN_Tool(self.dbms, state["initial_sql"])
                explain_info = (
                    json.dumps(explain, ensure_ascii=False)
                    if not isinstance(explain, str)
                    else explain
                )
            
            # ====== Bandit scoring (single-pass generation + bandit sorting) ======
            context = self.bandit.extract_context(state["initial_sql"], explain_info, groups)
            scored_rules = self.bandit.score_rules(lib, context)
            
            # Build UCT-scored rule library for LLM
            uct_scored_lib = {}
            for group, rule_id, score, desc in scored_rules:
                if group not in uct_scored_lib:
                    uct_scored_lib[group] = {}
                uct_scored_lib[group][rule_id] = (desc, score)
            
            print(f"📊 UCT scored {len(scored_rules)} rules")
            top_3 = scored_rules[:3]
            for g, rid, score, _ in top_3:
                print(f"   → {rid}: UCT={score:.3f}")
            
            async with self.llm_semaphore:
                seq = await self.reasoning_agent.select_rule_sequence(
                    state["initial_sql"],
                    state["optimization_advice"],
                    uct_scored_lib,  # Pass UCT-scored library
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
            sel = state.get("selected_rules") or {"applied_rules": [], "groups": ""}
            rules = sel.get("applied_rules", [])
            ex = get_rule_examples(rules)
            stats = _stats_str(state["data_statistics"])
            idx = state.get("index_info") or self.index_info
            async with self.llm_semaphore:
                rr = await self.rewrite_agent.rewrite_with_rule_sequence(
                    state["initial_sql"],
                    sel,
                    ex,
                    json.dumps(state.get("optimization_advice") or [], ensure_ascii=False),
                    stats,
                    schema_content=state.get("schema_content") or "",
                    index_info=idx,
                )
            trace.append({"node": "rewrite", "output": dict(rr)})
            if rr.get("parse_error") and rr.get("rewritten_sql") == state["initial_sql"]:
                async with self.llm_semaphore:
                    fixed = await self.rewrite_agent.iterative_rewrite(
                        state["initial_sql"],
                        rr.get("error_info", "JSON 解析错误"),
                        rr,
                        data_statistics=stats,
                        index_info=idx,
                    )
                if fixed:
                    rr["rewritten_sql"] = fixed
                    rr.pop("parse_error", None)
            raw_note = rr.get("semantic_check")
            note_s = raw_note.strip() if isinstance(raw_note, str) else ""
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
        rr = state.get("current_rewrite_result") or {}
        sql = rr.get("rewritten_sql", state["initial_sql"])
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
                    cur = await self.rewrite_agent.iterative_rewrite(
                        state["initial_sql"],
                        err,
                        prev,
                        data_statistics=stats,
                        index_info=idx,
                    ) or cur
                if cur and (await DBMS_Syntax_Tool(self.dbms, cur)).get("flag", True):
                    rr["rewritten_sql"] = cur
                    trace.append({"node": "syntax_check", "output": {"valid": True, "attempts": attempt + 1}})
                    return {"current_rewrite_result": rr, "agent_trace": trace}
            rr["rewritten_sql"] = state["initial_sql"]
            trace.append({"node": "syntax_check", "output": {"valid": False, "rolled_back": True}})
            return {"current_rewrite_result": rr, "agent_trace": trace}
        except Exception as e:
            print(f"语法检查异常: {e}")
            rr["rewritten_sql"] = state["initial_sql"]
            return {
                "current_rewrite_result": rr,
                "agent_trace": trace + [{"node": "syntax_check", "error": str(e)}],
            }

    async def _semantic_check_node(self, state: RewriteState) -> Dict[str, Any]:
        print("🧠 开始语义等价检查...")
        trace = list(state.get("agent_trace") or [])
        rr = state.get("current_rewrite_result") or {}
        cur = rr.get("rewritten_sql", state["initial_sql"])
        rules = rr.get("applied_rules") or (state.get("selected_rules") or {}).get("applied_rules", [])
        schema = (state.get("schema_content") or "").strip()
        stats = _stats_str(state["data_statistics"])
        idx = state.get("index_info") or self.index_info

        note_top = (state.get("semantic_check_note") or "").strip()
        note_rr = rr.get("semantic_check")
        note_rr_s = note_rr.strip() if isinstance(note_rr, str) else ""
        semantic_note = (note_top or note_rr_s) or None
        last_chk: Optional[Dict[str, Any]] = None
        for _ in range(self.MAX_SEMANTIC_FIX):
            async with self.llm_semaphore:
                chk = await self.semantic_check_agent.check_equivalence(
                    state["initial_sql"],
                    cur,
                    rules,
                    semantic_check=semantic_note,
                    schema_content=schema,
                    index_info=idx,
                )
            last_chk = chk
            trace.append({"node": "semantic_check", "output": chk})
            if chk.get("equivalent"):
                rr["rewritten_sql"] = cur
                print("✅ 语义等价")
                return {
                    "current_rewrite_result": rr,
                    "semantic_equivalent": True,
                    "semantic_equivalence_message": chk.get("message", "") or "",
                    "semantic_equivalence_differences": chk.get("differences") or [],
                    "agent_trace": trace,
                }
            print(f"❌ 语义不等价: {chk.get('message', '')}")
            async with self.llm_semaphore:
                fixed = await self.rewrite_agent.semantic_fix(
                    state["initial_sql"],
                    cur,
                    rules,
                    chk.get("differences") or [],
                )
            if not fixed:
                break
            candidate = fixed
            syntax_ok = False
            for _s in range(self.MAX_SYNTAX_AFTER_SEMANTIC):
                syn = await DBMS_Syntax_Tool(self.dbms, candidate)
                if syn.get("flag", syn.get("valid", True)):
                    cur = candidate
                    syntax_ok = True
                    break
                err = syn.get("error", "")
                prev = dict(rr)
                prev["rewritten_sql"] = candidate
                prev["error_info"] = err
                async with self.llm_semaphore:
                    candidate = await self.rewrite_agent.iterative_rewrite(
                        state["initial_sql"],
                        err,
                        prev,
                        data_statistics=stats,
                        index_info=idx,
                    ) or candidate
            if not syntax_ok:
                break

        rr["rewritten_sql"] = state["initial_sql"]
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

        init_sql = state["initial_sql"]
        rr = state.get("current_rewrite_result") or {}
        rw = rr.get("rewritten_sql", init_sql)
        sel = state.get("selected_rules") or {}
        groups = sel.get("groups", "")
        applied = sel.get("applied_rules", [])

        try:
            async def _exp(sql: str):
                async with self.db_semaphore:
                    return await DBMS_EXPLAIN_Tool(self.dbms, sql)

            o_exp, r_exp = await asyncio.gather(_exp(init_sql), _exp(rw))
            rnd = state.get("current_round", 1)
            oc = self._extract_cost_from_explain(o_exp)
            rc = self._extract_cost_from_explain(r_exp)
            info = {
                "original_costs": oc,
                "rewritten_costs": rc,
                "original_explain_info": json.dumps(o_exp, ensure_ascii=False) if not isinstance(o_exp, str) else o_exp,
                "rewritten_explain_info": json.dumps(r_exp, ensure_ascii=False) if not isinstance(r_exp, str) else r_exp,
                "groups": groups,
                "applied_rules": applied,
                "optimization_advice": state.get("optimization_advice") or [],
                "original_sql": init_sql,
                "rewritten_sql": rw,
                "reason": (state.get("previous_feedback") or {}).get("reason", ""),
            }
            async with self.llm_semaphore:
                ev = await self.decision_agent.evaluate_with_costs(info, rnd)
            trace.append({"node": "evaluation", "output": ev})
            terminate = ev.get("terminate", True)
            keep_rewritten = ev.get("是否保留重写SQL", True)
            rollback = not keep_rewritten

            best_cost = state.get("best_cost")
            best_sql = state.get("best_sql")
            best_rules = state.get("best_rules")
            best_groups = state.get("best_groups", "")
            # Only advance "best kept state" when this round is explicitly kept.
            if (
                keep_rewritten
                and rw != init_sql
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
                "evaluation_terminate": ev.get("terminate", True),
                "evaluation_rollback_sql": rollback,
                "agent_trace": trace,
            }

            reduction = ((oc - rc) / oc * 100) if oc > 0 else 0.0
            if terminate:
                out["should_terminate"] = True
                if rollback:
                    # Roll back to last kept/best rewritten state; fallback to original SQL.
                    fallback_sql = best_sql if best_sql else init_sql
                    fallback_cost = best_cost if (best_sql and best_cost is not None) else oc
                    out["rewritten_sql"] = fallback_sql
                    out["final_rewritten_costs"] = fallback_cost
                    rr = dict(rr)
                    rr["rewritten_sql"] = fallback_sql
                    out["current_rewrite_result"] = rr
                elif rc < oc:
                    # Store to global memory
                    if self.global_memory:
                        try:
                            self.global_memory.store_successful_optimization(
                                original_sql=init_sql,
                                rewritten_sql=out["rewritten_sql"],
                                rule_sequence=applied,
                                groups=groups,
                                original_cost=oc,
                                rewritten_cost=rc,
                            )
                        except Exception as ex:
                            print(f"⚠️ 知识库存储失败: {ex}")
                    
                    out["_uct_update_info"] = self._uct_update_payload(
                        sel, applied, init_sql, rw, o_exp, r_exp, oc, rc
                    )
                return out

            nxt = rnd + 1
            if nxt > self.MAX_ITERATION_LOOP:
                out["should_terminate"] = True
                ratio = ((rc - oc) / oc * 100) if oc > 0 else 0.0
                if ratio > self.COST_ROLLBACK_PCT:
                    # Cost degrades too much: roll back to last kept/best rewritten state,
                    # or original SQL when no kept rewrite exists.
                    fallback_sql = best_sql if best_sql else init_sql
                    fallback_cost = best_cost if (best_sql and best_cost is not None) else oc
                    out["rewritten_sql"] = fallback_sql
                    out["final_rewritten_costs"] = fallback_cost
                    rr = dict(state.get("current_rewrite_result") or {})
                    rr["rewritten_sql"] = fallback_sql
                    out["current_rewrite_result"] = rr
                elif rc < oc and applied:
                    if self.global_memory:
                        try:
                            self.global_memory.store_successful_optimization(
                                original_sql=init_sql,
                                rewritten_sql=rw,
                                rule_sequence=applied,
                                groups=groups,
                                original_cost=oc,
                                rewritten_cost=rc,
                            )
                        except Exception as ex:
                            print(f"⚠️ 知识库存储失败: {ex}")
                    out["_uct_update_info"] = self._uct_update_payload(
                        sel, applied, init_sql, rw, o_exp, r_exp, oc, rc
                    )
                return out

            out["should_terminate"] = False
            out["current_round"] = nxt
            out["previous_feedback"] = {
                "reason": ev.get("reason", ""),
                "problematic_rules": ev.get("应用的规则", applied),
            }
            if rollback:
                fallback_sql = best_sql if best_sql else init_sql
                fallback_cost = best_cost if (best_sql and best_cost is not None) else oc
                out["rewritten_sql"] = fallback_sql
                out["final_rewritten_costs"] = fallback_cost
                rr = dict(state.get("current_rewrite_result") or {})
                rr["rewritten_sql"] = fallback_sql
                out["current_rewrite_result"] = rr
            return out
        except Exception as e:
            print(f"评估失败: {e}")
            return {"should_terminate": True, "agent_trace": trace + [{"node": "evaluation", "error": str(e)}]}

    def _format_final_output(self, state: RewriteState) -> Dict[str, Any]:
        oc = state.get("final_original_costs", state.get("original_cost", 0.0))
        if not state.get("can_optimize"):
            return {
                "tpch": [
                    {
                        "rewritten_query": state["initial_sql"],
                        "original_costs": oc,
                        "rewrite_costs": oc,
                        "costs_reduction_rate": 0,
                        "rewrite_rules": None,
                    }
                ],
                "rewrite_suggestion": [],
                "agent_trace": state.get("agent_trace", []),
            }
        final_sql = state.get("rewritten_sql") or state["initial_sql"]
        if state.get("current_rewrite_result"):
            final_sql = state["current_rewrite_result"].get("rewritten_sql", final_sql)
        rc = state.get("final_rewritten_costs", oc)
        if final_sql == state["initial_sql"]:
            rc = oc
        rate = round(((oc - rc) / oc * 100) if oc > 0 else 0.0, 2)
        sel = state.get("selected_rules") or {}
        rules = sel.get("applied_rules") if final_sql != state["initial_sql"] else None
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
            "evaluation_rollback_sql": False,
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
                        "rewritten_query": initial_sql,
                        "original_costs": oc,
                        "rewrite_costs": oc,
                        "costs_reduction_rate": 0,
                        "rewrite_rules": None,
                    }
                ],
                "rewrite_suggestion": [],
            }
