
import json
import math
import os
import re
import textwrap
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

# Setup project paths
from src.utils.path_config import setup_python_path, load_project_env
setup_python_path()
load_project_env()

from src.utils.agent_template import MessageQueue, Agent
from src.utils.llm_client import GPT
from src.utils.llm_json_utils import parse_llm_json, parse_llm_json_with_default
from src.Query_Rewriter.global_memory.sql_fingerprint import SQLFingerprintGenerator

STRICT_JSON_SCHEMAS = {
    "rule_selection": {
        "name": "rule_selection_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "groups": {"type": "string"},
                "applied_rules": {"type": "array", "items": {"type": "string"}},
                "rule_effect_scores": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
                "rule_effect_confidence": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
            },
            "required": ["groups", "applied_rules", "rule_effect_scores", "rule_effect_confidence"],
            "additionalProperties": False,
        },
    },
    "initial_check": {
        "name": "initial_optimization_check",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "can_optimize": {"type": "boolean"},
                "reason": {"type": "string"},
                "advice": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "group": {"type": "string"},
                            "produced_suggestion": {"type": "string"},
                        },
                        "required": ["group", "produced_suggestion"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["can_optimize", "reason", "advice"],
            "additionalProperties": False,
        },
    },
    "evaluation": {
        "name": "rewrite_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "terminate": {"type": "boolean"},
                "reason": {"type": "string"},
                "no_further_optimization": {"type": "boolean"},
                "next_step_advice": {"type": "string"},
            },
            "required": ["terminate", "reason", "no_further_optimization", "next_step_advice"],
            "additionalProperties": False,
        },
    },
    "rewrite": {
        "name": "rewrite_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "groups": {"type": "string"},
                "applied_rules": {"type": "array", "items": {"type": "string"}},
                "rewritten_sql": {"type": "string"},
                "semantic_correctness_guarantee": {"type": "string"},
            },
            "required": [
                "groups",
                "applied_rules",
                "rewritten_sql",
                "semantic_correctness_guarantee",
            ],
            "additionalProperties": False,
        },
    },
    "semantic_fix": {
        "name": "semantic_fix_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "fixed_sql": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["fixed_sql", "note"],
            "additionalProperties": False,
        },
    },
    "iterative_rewrite": {
        "name": "iterative_rewrite_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "rewritten_sql": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["rewritten_sql", "note"],
            "additionalProperties": False,
        },
    },
    "semantic_equivalence": {
        "name": "semantic_equivalence_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "equivalent": {"type": "boolean"},
                "message": {"type": "string"},
                "differences": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "description": {"type": "string"},
                            "location": {"type": "string"},
                        },
                        "required": ["type", "description", "location"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["equivalent", "message", "differences"],
            "additionalProperties": False,
        },
    },
}


def _build_single_agent_gpt() -> GPT:
    """Single LLM client for architecture ablation (single-agent branch)."""
    return GPT(
        api_key=os.getenv("SINGLE_AGENT_MODEL_API_KEY") or os.getenv("REASONING_MODEL_API_KEY"),
        model=os.getenv("SINGLE_AGENT_MODEL") or os.getenv("REASONING_MODEL"),
        base_url=os.getenv("SINGLE_AGENT_MODEL_URL") or os.getenv("REASONING_MODEL_URL"),
        thinking_type=os.getenv("SINGLE_AGENT_MODEL_THINKING") or os.getenv("REASONING_MODEL_THINKING"),
        reasoning_effort=os.getenv("SINGLE_AGENT_MODEL_REASONING_EFFORT")
        or os.getenv("REASONING_MODEL_REASONING_EFFORT"),
    )


def _merge_rewrite_semantic_fields(parsed: dict) -> None:
    """
    统一 semantic_correctness_guarantee 与旧字段 semantic_check；
    优先非空的 semantic_correctness_guarantee，缺省时用 semantic_check（兼容旧模型输出）。
    """
    g = parsed.get("semantic_correctness_guarantee", "")
    g = g.strip() if isinstance(g, str) else ""
    leg = parsed.get("semantic_check", "")
    leg = leg.strip() if isinstance(leg, str) else ""
    if g:
        parsed["semantic_correctness_guarantee"] = g
    elif leg:
        parsed["semantic_correctness_guarantee"] = leg
    else:
        parsed["semantic_correctness_guarantee"] = ""
    parsed["semantic_check"] = parsed.get("semantic_correctness_guarantee", "")


class ReasoningAgent(Agent):
    """MDP-based Reasoning Agent"""
    def __init__(self, mq: MessageQueue):
        super().__init__("ReasoningAgent", mq, gpt = GPT(
        api_key=os.getenv("REASONING_MODEL_API_KEY"),
        model=os.getenv("REASONING_MODEL"),
        base_url=os.getenv("REASONING_MODEL_URL")
        ))

    @staticmethod
    def _sanitize_rule_effect_scores(applied_rules, raw_scores) -> Dict[str, float]:
        """Keep only positive numeric scores for selected rules and normalize to sum=1."""
        if not isinstance(applied_rules, list) or not applied_rules:
            return {}
        if not isinstance(raw_scores, dict):
            return {}

        cleaned: Dict[str, float] = {}
        for rid in applied_rules:
            v = raw_scores.get(rid)
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                fv = float(v)
                if fv > 0:
                    cleaned[rid] = fv
            elif isinstance(v, str):
                s = v.strip()
                if not s:
                    continue
                try:
                    fv = float(s)
                except ValueError:
                    continue
                if fv > 0:
                    cleaned[rid] = fv

        total = sum(cleaned.values())
        if total <= 0:
            return {}
        return {rid: cleaned[rid] / total for rid in cleaned}

    @staticmethod
    def _sanitize_rule_effect_confidence(applied_rules, raw_confidence) -> Dict[str, float]:
        """Keep confidence values in [0, 1] for selected rules only."""
        if not isinstance(applied_rules, list) or not applied_rules:
            return {}
        if not isinstance(raw_confidence, dict):
            return {}

        cleaned: Dict[str, float] = {}
        for rid in applied_rules:
            v = raw_confidence.get(rid)
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                cleaned[rid] = float(max(0.0, min(1.0, float(v))))
            elif isinstance(v, str):
                s = v.strip()
                if not s:
                    continue
                try:
                    fv = float(s)
                except ValueError:
                    continue
                cleaned[rid] = float(max(0.0, min(1.0, fv)))
        return cleaned

    async def select_rule_sequence(self, sql: str, decision_advice: list, rule_library: dict, data_statistics: str, explain_info: str, iteration_round: int = 1, previous_feedback: dict = None, few_shot_examples: list = None, index_info: str = "") -> dict:
        """Select appropriate rule sequence based on DecisionAgent's advice with UCT scores."""
        advice_text = json.dumps(decision_advice, ensure_ascii=False, indent=2)

        rule_text = ""
        all_groups = []
        all_rules = []

        has_uct_scores = False
        for group, rules in rule_library.items():
            if rules:
                first_val = next(iter(rules.values()))
                if isinstance(first_val, tuple):
                    has_uct_scores = True
                break

        for group, rules in rule_library.items():
            all_groups.append(group)
            rule_text += f"### {group}\n"

            sorted_rules = []
            for rule_id, rule_data in rules.items():
                if has_uct_scores and isinstance(rule_data, tuple):
                    desc, score = rule_data
                    sorted_rules.append((rule_id, desc, score))
                else:
                    sorted_rules.append((rule_id, rule_data, 1.0))

            sorted_rules.sort(key=lambda x: x[2], reverse=True)

            for rule_id, desc, score in sorted_rules:
                all_rules.append(rule_id)
                if has_uct_scores:
                    rule_text += f"- {rule_id} [UCT分数: {score:.3f}]: {desc}\n"
                else:
                    rule_text += f"- {rule_id}: {desc}\n"

        groups_text = ", ".join(all_groups)

        previous_feedback_text = ""
        if previous_feedback:
            previous_feedback_text = f"""
            此SQL已经经过第{iteration_round-1}轮优化，但评估结果为需要继续优化。
            上一轮信息：
            {json.dumps(previous_feedback, ensure_ascii=False, indent=2)}
            请基于上一轮的经验和失败原因重新选择规则序列，避免重复同样的错误。
            """
        
        # Build few-shot examples context
        few_shot_context = ""
        if few_shot_examples and len(few_shot_examples) > 0:
            few_shot_context = "\n\n<相似历史案例>\n"
            few_shot_context += "以下是历史上相似SQL的优化案例，供参考（你可以参考这些案例的思路，但需要根据当前SQL的具体情况选择规则）：\n\n"
            for i, example in enumerate(few_shot_examples[:3], 1):
                few_shot_context += f"案例{i}:\n"
                few_shot_context += f"- 相似度: {example.get('score', 0):.4f}\n"
                few_shot_context += f"- SQL指纹: {example.get('sql_fingerprint', '')[:150]}...\n"
                few_shot_context += f"- 应用的规则序列: {example.get('rule_sequence', [])}\n"
                few_shot_context += f"- 优化组别: {example.get('groups', '')}\n"
                few_shot_context += f"- 命中次数: {example.get('frequency', 0)}\n\n"
            few_shot_context += "</相似历史案例>\n"

        uct_guidance = """
        【UCT分数说明】
        规则后面的 [UCT分数: x.xxx] 表示该规则在历史上的表现评分：
        - 高分规则（>0.7）：历史成功率较高，通常更可靠
        - 中分规则（0.4-0.7）：有一定成功案例，需结合SQL结构判断
        - 低分规则（<0.4）：历史表现一般或为较少尝试的新规则
        可以优先选择 UCT 分数高的规则，但需要结合当前SQL的具体结构做最终判断，不要盲目选择。
        """ if has_uct_scores else ""

        prompt = textwrap.dedent(f"""
        <Mission>
        你是一名经验丰富的 DBA，你的任务是基于 SQL / 执行计划 / 统计信息，从全量规则库中直接挑选合适的优化规则序列。你只负责选择规则，不进行任何重写操作。

        1. 规则选择原则：
           - 只能选择与当前SQL结构相符且能够优化该SQL的规则
           - 当查询语句包含复杂的WHERE/JOIN条件，或存在重复子查询计算时，可考虑使用公共表表达式（CTE）
           - 若查询语句本身结构简单，或使用CTE无法减少冗余计算，则应避免过度使用CTE，多余的CTE可能会增加系统开销
        2. 规则选择逻辑：
           - 先分析输入SQL的结构和执行计划中的瓶颈
           - 逐一考虑规则库中的规则，判断其是否适用
           - 挑选能够解决问题的规则
           - 按照规则依赖关系进行排序
           {few_shot_context}

        {previous_feedback_text}

        {uct_guidance}

        <sql语句（当前轮次基底 SQL；若为第2轮及以后，则是上一轮 rewritten_sql）>
        {sql}

        <decision_advice>
        {advice_text}

        <rule_library>
        {rule_text}

        <统计信息>
        {data_statistics}

        <索引信息>
        {index_info}

        <执行计划分析结果>
        {explain_info}

        3. 输出要求：**只输出一个 JSON 对象**（不要 XML 标签、不要 markdown）。字段：
        - groups: 字符串，填写你判断本次所选规则涉及的类别；可为单个类别或多个类别的逗号分隔字符串，不要求与 decision_advice 中的 group 保持一致
        - applied_rules: 字符串数组，按应用顺序列出规则 ID；必须来自 <rule_library>，禁止编造；若无适用规则则为 []
        - rule_effect_scores: 对象，键为 applied_rules 中的 rule_id，值为该规则预估贡献分（正数）。建议总和约为 1；若无规则则 {{}}
        - rule_effect_confidence: 对象，键为 applied_rules 中的 rule_id，值为该规则效果估计的置信度（0到1之间）

        示例：{{"groups": "连接优化, 谓词简化", "applied_rules": ["RULE_ID_1"], "rule_effect_scores": {{"RULE_ID_1": 1.0}}, "rule_effect_confidence": {{"RULE_ID_1": 0.8}}}}
        """)

        thought_chain = await self.llm.get_LLM_response_async(
            prompt=prompt,
            json_format=True,
            json_schema=STRICT_JSON_SCHEMAS["rule_selection"],
        )

        default = {
            "groups": groups_text,
            "applied_rules": [],
            "chosen_rule_prior": [],
            "rule_effect_scores": {},
            "rule_effect_confidence": {},
        }
        parsed, _ = parse_llm_json(thought_chain, default)
        if parsed and isinstance(parsed.get("applied_rules"), list):
            parsed["chosen_rule_prior"] = []
            parsed["rule_effect_scores"] = ReasoningAgent._sanitize_rule_effect_scores(
                parsed.get("applied_rules", []),
                parsed.get("rule_effect_scores", {}),
            )
            parsed["rule_effect_confidence"] = ReasoningAgent._sanitize_rule_effect_confidence(
                parsed.get("applied_rules", []),
                parsed.get("rule_effect_confidence", {}),
            )
            return parsed
        sequence_match = re.search(r"<rule_sequence>(.*?)</rule_sequence>", thought_chain, re.DOTALL)
        inner = sequence_match.group(1).strip() if sequence_match else thought_chain
        parsed2, _ = parse_llm_json(inner, default)
        if parsed2 and isinstance(parsed2.get("applied_rules"), list):
            parsed2["chosen_rule_prior"] = []
            parsed2["rule_effect_scores"] = ReasoningAgent._sanitize_rule_effect_scores(
                parsed2.get("applied_rules", []),
                parsed2.get("rule_effect_scores", {}),
            )
            parsed2["rule_effect_confidence"] = ReasoningAgent._sanitize_rule_effect_confidence(
                parsed2.get("applied_rules", []),
                parsed2.get("rule_effect_confidence", {}),
            )
            return parsed2
        return default


class DecisionAgent(Agent):
    """Summarize Optimization Suggestions Agent"""
    def __init__(self, mq: MessageQueue):
        super().__init__("DecisionAgent", mq, gpt = GPT(
    api_key=os.getenv("DECISION_MODEL_API_KEY"),
    model=os.getenv("DECISION_MODEL"),
    base_url=os.getenv("DECISION_MODEL_URL"),
    thinking_type=os.getenv("DECISION_MODEL_THINKING"),
    reasoning_effort=os.getenv("DECISION_MODEL_REASONING_EFFORT"),
))
        self.watch(["ReasoningAgent", "ExplainAgent"])

    async def initial_optimization_check(
        self,
        sql: str,
        data_statistics: str,
        explain_info: str,
        few_shot_examples: list = None,
        index_info: str = "",
        previous_feedback: dict = None,
    ) -> dict:
        """Initial check to determine if SQL can be optimized through rewriting"""
        
        # Build few-shot examples context
        few_shot_context = ""
        if few_shot_examples and len(few_shot_examples) > 0:
            few_shot_context = "\n\n<相似历史案例>\n"
            few_shot_context += "以下是历史上相似SQL的优化案例，供参考：\n\n"
            for i, example in enumerate(few_shot_examples[:3], 1):
                few_shot_context += f"案例{i}:\n"
                few_shot_context += f"- 相似度: {example.get('score', 0):.4f}\n"
                few_shot_context += f"- SQL指纹: {example.get('sql_fingerprint', '')[:100]}...\n"
                few_shot_context += f"- 应用的规则序列: {example.get('rule_sequence', [])}\n"
                few_shot_context += f"- 命中次数: {example.get('frequency', 0)}\n\n"
            few_shot_context += "</相似历史案例>\n"
        previous_feedback_text = ""
        if previous_feedback:
            previous_feedback_text = f"""
        <上一轮未改进/恶化反馈>
        {json.dumps(previous_feedback, ensure_ascii=False, indent=2)}
        </上一轮未改进/恶化反馈>
        """
        
        prompt = textwrap.dedent(f"""
        <Mission>
        你是一名经验丰富的 SQL查询重写优化专家，你的任务是判断给定的 SQL 查询是否可以通过查询重写优化来提升执行性能。

        1. 分析输入信息：
           - <sql语句>: 当前轮次的基底 SQL。第一轮它等于原始SQL；若进入下一轮，它就是上一轮的 rewritten_sql
           - <统计信息>: 数据库表统计信息
           - <索引信息>: 表索引（索引名与定义）
           - <执行计划分析结果>: 智能执行计划分析器生成的分析结果
           {few_shot_context}
           {previous_feedback_text}

        若提供了“上一轮未改进/恶化反馈”，必须优先据此重新选择优化方向：
           - 避免重复上一轮失败思路
           - 针对反馈中的未改进原因和 next_step_advice 调整优化组
           - advice 中要体现对失败原因的修正策略（不是泛化建议）

        2. 基于执行计划分析中的高代价算子信息，判断SQL是否可以通过以下优化方式提升性能：
           - 子查询优化：子查询转换为JOIN、相关子查询优化、CTE分解等
           - 连接优化：连接顺序调整、连接条件优化、半连接转换等
           - 谓词简化：过滤条件下推、合并、表达式简化等
           - 常量折叠：常量表达式预计算、常量列处理等
           - 聚合优化：GROUP BY优化、聚合函数简化、聚合下推等
           - 投影优化：列裁剪、多余投影消除等
           - 排序优化：冗余排序消除、排序下推等
           - 集合优化：UNION优化、集合操作转换等

        3. 判断标准：
           [True] - 可以优化：
           - 执行计划中存在明显性能瓶颈（如Seq Scan高代价算子、Nested Loop低效连接等）
           - SQL结构复杂，存在可优化的模式（如子查询、复杂JOIN、多层嵌套等）
           - 统计信息显示存在优化空间（如大表扫描、小表关联等）

           [False] - 不需要优化：
           - SQL已经是最优结构，执行计划合理
           - 执行计划显示已经是最佳执行方式
           - SQL已经不能单纯通过查询重写优化来提升性能，需要结合索引或物理优化方式来提升性能

        4. 如果可以优化，在 advice 中给出建议数组；否则 advice 为 []。advice 每项含 group（八类之一）与 produced_suggestion（字符串）。
           八类：子查询优化、连接优化、谓词简化、常量折叠、聚合优化、投影优化、排序优化、集合优化。

        5. **只输出一个 JSON 对象**（不要 markdown、不要注释）。字段：
           can_optimize（布尔）、reason（字符串）、advice（数组，元素为 {{"group","produced_suggestion"}}）。

        <sql语句>
        {sql}

        <统计信息>
        {data_statistics}

        <索引信息>
        {index_info}

        <执行计划分析结果>
        {explain_info}
        """)

        thought_chain = await self.llm.get_LLM_response_async(
            prompt=prompt,
            json_format=True,
            json_schema=STRICT_JSON_SCHEMAS["initial_check"],
        )

        default_result = {
            "can_optimize": False,
            "reason": "无法解析LLM响应，默认认为无需优化",
            "advice": [],
        }
        parsed, _ = parse_llm_json(thought_chain, default_result)
        if not parsed:
            return default_result
        parsed.setdefault("can_optimize", False)
        parsed.setdefault("advice", [])
        parsed.setdefault("reason", "")
        return parsed

    async def evaluate_with_costs(self, optimization_info: dict, iteration_round: int = 1) -> dict:
        """Evaluate rewrite quality and whether to terminate optimization loop."""
        original_costs = optimization_info.get("original_costs", 0)
        rewritten_costs = optimization_info.get("rewritten_costs", 0)
        groups = optimization_info.get("groups", "")
        applied_rules = optimization_info.get("applied_rules", [])
        optimization_advice = optimization_info.get("optimization_advice", [])
        original_sql = optimization_info.get("original_sql", "")
        rewritten_sql = optimization_info.get("rewritten_sql", "")
        original_explain = optimization_info.get("original_explain_plan_json", "")
        rewritten_explain = optimization_info.get("rewritten_explain_plan_json", "")
        advice_text = json.dumps(optimization_advice, ensure_ascii=False, indent=2) if optimization_advice else "[]"
        plan_struct = optimization_info.get("plan_structure_analysis")
        struct_block = ""
        if plan_struct:
            struct_block = textwrap.dedent(f"""

            <heuristic_plan_structure_summary>
            以下为对两套 EXPLAIN JSON 的轻量结构统计（节点类型计数、估计行数求和等），
            仅供与全文 EXPLAIN 对照；其中“估计行数求和”只是弱参考，若与全文结构信号冲突以全文结构分析为准。
            {json.dumps(plan_struct, ensure_ascii=False, indent=2)}
            </heuristic_plan_structure_summary>
            """)

        prompt = textwrap.dedent(f"""
        你负责评估 SQL 优化是否达标，并判断是否终止优化流程。

        基于 `original_sql` 与 `rewritten_sql` 的 SQL 语句与 EXPLAIN 做综合评估，重点对比：
        1) 执行计划变化（扫描方式、连接顺序/算法、过滤下推、排序聚合、临时表/回表、并行度等）；
        2) 结构性改进/退化信号（是否丢失关键访问路径、是否新增无补偿的高开销排序/物化、是否出现明显不合理的数据膨胀等）；
        3) 重写 SQL 改进点（执行计划层面 + SQL 逻辑层面）；
        4) 成本差异（cost 仅作弱信号，不能单独决定结论；即使 costs 增大明显，也只能作为辅助佐证，不能脱离结构分析单独判为恶化）。

        注意：`original_sql` 和 `enhanced_sql` 的 costs 来自优化器，可能不准确。必须采用以下原则：
        - 结构优先：结构信号优先级高于 cost。
        - cost弱化：cost 只能辅助佐证，禁止仅凭 cost 判定优劣。
        - 若结构与 cost 冲突，优先相信结构信号。
        - DSS/OLAP 特殊原则：对于大表分析型 SQL，Seq Scan / Parallel Seq Scan 不天然是负信号；若重写消除了重复执行、相关子查询、回表、随机探测，或通过 CTE/派生表/预聚合实现“一次扫描 + 复用公共结果”，则即使出现全表扫描，也不得直接据此判为恶化。
        - 访问路径判断原则：索引/位图扫描减少，并不自动等于退化；必须结合扫描次数是否减少、子计划是否消失、过滤是否前移、是否减少重复 I/O、是否增强并行/聚合复用来综合判断。
        - 行数估计弱化：行数估计求和（各节点 Plan Rows 累加）仅作辅助参考，不能单独作为“明显恶化”的依据；尤其在 CTE、派生表、预聚合、去相关、JOIN 重排场景下，该指标可能显著失真。
        - 特殊高优先级规则：若行数估计求和（Plan Rows累加）大幅下降（如降至原10%以下），则优先视为优化成功，不得仅因 Seq Scan 增加或 cost 上升而判为恶化。
        - 另一类高优先级正信号：若检测到去相关（decorrelation）、子计划减少、重复执行被消除、或公共中间结果经 CTE/预聚合/派生表复用，则不得仅因 Seq Scan 增加或 cost 上升而直接判为恶化，应优先倾向保留重写 SQL。
        - 成功终止优先规则：若检测到去相关、子计划减少、重复执行消除、或 CTE/派生表/预聚合复用公共中间结果，且未发现明确且可验证的灾难性结构退化证据，则可直接视为“优秀重写”并给出 terminate=true；不要求同时满足 cost 明显下降。
        - 若已检测到去相关/子计划减少/CTE复用/预聚合等高优先级正信号，除非还能指出“明确且可验证的灾难性结构退化证据”（如关键大表被重复扫描、过滤明显失效导致主干数据量失控、并行完全丢失且无任何补偿、引入新的高开销排序/物化主瓶颈），否则不得给出以“全表扫描”或“索引减少”为主因的负面结论。
        - cost 使用约束：即使 costs 增大超过 50%，也只有在同时存在上述“明确且可验证的灾难性结构退化证据”时，才可将其作为支持 terminate=false 的辅助证据；若结构显示去相关/复用/减少重复执行，则不得仅因高 cost 增幅否定该重写。

        终止条件：
        - `terminate = true`（满足任一即终止）：
          a. 结构上明确改进；或
          b. 检测到去相关、子计划减少、重复执行消除、或 CTE/派生表/预聚合复用等高优先级正信号，且不存在明确灾难性结构退化证据；或
          c. 行数估计求和（Plan Rows累加）大幅度降低（如降至原10%以下），无论cost是否增加，均视为优化成功；或
          d. costs大幅度降低；或
          e. 判断"下一步重写已几乎无改进空间"（no_further_optimization=true）。
        - `terminate = false`：
          结构上明确恶化（如关键大表被重复扫描、并行度显著下降且无补偿、过滤明显失效导致主干数据量失控、明显引入新的高开销排序/物化主瓶颈等）。注意：不得仅因出现全表扫描、索引/位图扫描减少、cost 明显上升、或行数估计求和上升就判为恶化。
        - 重要补充：
          若本轮并非“明确恶化”，而是“已有明显正向结构收益，但仍可能继续微调”，可直接 terminate=true；不要因为“也许还能继续优化”就默认 terminate=false。
          若本轮“未改进或恶化”，且仍存在可执行的下一步优化方向，必须给出 terminate=false，并在 next_step_advice 中写明下一轮应如何基于当前 rewritten_sql 继续改写。
          第一轮重写以 original_sql 为基底；若进入下一轮，则默认是在上一轮 rewritten_sql 的基础上继续改写，而不是回退到 original_sql 重新开始。
          本评估阶段不再决定“是否保留 SQL”；最终输出默认沿用当前 rewritten_sql。

        严格仅输出以下 JSON（不得有任何额外文本）：
        {{
          "terminate": true/false,
          "reason": "简要说明依据：执行计划变化、结构性改进/退化信号、cost（仅辅助）、改进点、最终结论；若存在去相关/子计划减少/公共结果复用等高优先级正信号，应明确说明其为何足以支持成功终止；若引用行数估计求和或高 cost，只能作为辅助信息，不得将其单独作为恶化结论依据；若不确定终止需明确不确定原因。",
          "no_further_optimization": true/false,
          "next_step_advice": "当 terminate=false 时，必须给出下一轮可执行改进建议；terminate=true 时可写“无需下一步优化”。"
        }}

        <original_sql>
        {original_sql}
        </original_sql>

        <enhanced_sql>
        {rewritten_sql}
        </enhanced_sql>

        <original_sql_costs>
        {original_costs}
        </original_sql_costs>

        <enhanced_sql_costs>
        {rewritten_costs}
        </enhanced_sql_costs>

        <original_sql_explain>
        {json.dumps(original_explain, ensure_ascii=False, indent=2) if isinstance(original_explain, (dict, list)) else original_explain}
        </original_sql_explain>

        <enhanced_sql_explain>
        {json.dumps(rewritten_explain, ensure_ascii=False, indent=2) if isinstance(rewritten_explain, (dict, list)) else rewritten_explain}
        </enhanced_sql_explain>
        {struct_block}

        <optimization_context>
        round: {iteration_round}
        groups: {groups}
        optimization_advice:
        {advice_text}
        applied_rules:
        {applied_rules}
        </optimization_context>
        """)

        response = await self.llm.get_LLM_response_async(
            prompt=prompt,
            json_format=True,
            json_schema=STRICT_JSON_SCHEMAS["evaluation"],
        )

        fallback = {
            "terminate": True,
            "reason": "无法解析响应，默认终止优化",
            "no_further_optimization": True,
            "next_step_advice": "无需下一步优化。",
        }
        result = parse_llm_json_with_default(response, fallback)
        if not isinstance(result, dict):
            return fallback
        result.setdefault("terminate", True)
        result.setdefault("reason", "")
        result.setdefault("no_further_optimization", False)
        result.setdefault("next_step_advice", "")
        return result


# Utility functions for rule management
CATEGORY_TO_GROUP_MAPPING = {
    "SUBQUERY_OPTIMIZATION": "子查询优化",
    "JOIN_OPTIMIZATION": "连接优化",
    "PREDICATE_OPTIMIZATION": "谓词简化",
    "CONSTANT_OPTIMIZATION": "常量折叠",
    "AGGREGATE_OPTIMIZATION": "聚合优化",
    "PROJECTION_OPTIMIZATION": "投影优化",
    "SORT_OPTIMIZATION": "排序优化",
    "SET_OPERATION_OPTIMIZATION": "集合优化",
}


def load_rule_knowledge_base() -> dict:
    """Load rule knowledge base from Rule_Examples.json"""
    try:
        rule_file = "/root/AgentRewrite/src/Rewrite_Middleware/Structured_Knowledge_Base/preparation/data/Rule_Examples.json"
        with open(rule_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading rule knowledge base: {e}")
        return {}

def get_rules_by_groups(groups: list) -> dict:
    """Get rules from knowledge base based on optimization groups"""
    rule_kb = load_rule_knowledge_base()
    result = {}

    # Group mapping to categories in Rule_Examples.json
    group_mapping = {v: k for k, v in CATEGORY_TO_GROUP_MAPPING.items()}

    for group in groups:
        category = group_mapping.get(group)
        if category and category in rule_kb:
            result[group] = rule_kb[category].get("rules", {})

    return result


def get_all_rules() -> dict:
    """Get the full rule library without filtering by optimization group."""
    rule_kb = load_rule_knowledge_base()
    result = {}

    for category, category_data in rule_kb.items():
        group_name = CATEGORY_TO_GROUP_MAPPING.get(category, category)
        rules = category_data.get("rules", {}) if isinstance(category_data, dict) else {}
        if isinstance(rules, dict) and rules:
            result[group_name] = rules

    return result

RULE_EXAMPLES_MAX_PER_RULE = 1
_RULE_EXAMPLE_FINGERPRINT_GENERATOR = SQLFingerprintGenerator()


def _sql_fingerprint(sql_text: str) -> str:
    """Return a normalized SQL fingerprint for rule-example similarity."""
    if not isinstance(sql_text, str) or not sql_text.strip():
        return ""
    try:
        return _RULE_EXAMPLE_FINGERPRINT_GENERATOR.get_template(sql_text) or sql_text
    except Exception:
        return sql_text


def _fingerprint_tokens(fingerprint: str) -> Counter:
    """Tokenize a SQL fingerprint into a lightweight bag-of-words vector."""
    if not fingerprint:
        return Counter()
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[(),=*<>!+\-/:]", fingerprint.lower())
    return Counter(tokens)


def _cosine_similarity(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _rule_example_similarity(current_sql: str, example_query: str) -> Dict[str, Any]:
    """Compare current SQL with an example original SQL using fingerprint cosine similarity."""
    current_fp = _sql_fingerprint(current_sql)
    example_fp = _sql_fingerprint(example_query)
    score = _cosine_similarity(_fingerprint_tokens(current_fp), _fingerprint_tokens(example_fp))
    return {
        "score": float(score),
        "current_fingerprint": current_fp,
        "example_fingerprint": example_fp,
    }


def get_rule_examples(
    rule_ids: list,
    current_sql: Optional[str] = None,
    max_per_rule: int = RULE_EXAMPLES_MAX_PER_RULE,
) -> dict:
    """
    按规则 ID 拉取最相关示例。

    - 若某个 rule_id 只有 1 条示例：直接选择。
    - 若某个 rule_id 有多条示例且提供 current_sql：比较 current_sql 与 example.original_query
      的 SQL 指纹余弦相似度，选择 Top-1（或 max_per_rule 指定的 Top-K）。
    - 若未提供 current_sql：退化为文件顺序 Top-K，保持旧调用兼容。
    """
    rule_kb = load_rule_knowledge_base()
    candidates: Dict[str, list] = {}
    want = set(rule_ids)

    for category_data in rule_kb.values():
        for example in category_data.get("examples", []):
            rule_id = example.get("id")
            if rule_id not in want:
                continue
            candidates.setdefault(rule_id, []).append(example)

    result: Dict[str, list] = {}
    limit = max(1, int(max_per_rule or 1))
    for rule_id in rule_ids:
        rule_id_s = str(rule_id)
        items = candidates.get(rule_id_s, [])
        if not items:
            continue
        if len(items) == 1 or not current_sql:
            result[rule_id_s] = [dict(item) for item in items[:limit]]
            continue

        scored_items = []
        for item in items:
            meta = _rule_example_similarity(current_sql, item.get("original_query", ""))
            enriched = dict(item)
            enriched["_similarity_score"] = meta["score"]
            enriched["_current_fingerprint"] = meta["current_fingerprint"]
            enriched["_example_fingerprint"] = meta["example_fingerprint"]
            scored_items.append(enriched)
        scored_items.sort(key=lambda item: item.get("_similarity_score", 0.0), reverse=True)
        result[rule_id_s] = scored_items[:limit]

    return result


def format_rule_examples_for_semantic_check(ex_map: Dict[str, list]) -> str:
    """
    将 get_rule_examples 结果格式化为 SemanticCheck prompt 中「规则 ID + 描述 + 示例」的示例块。
    """
    if not ex_map:
        return (
            "（知识库 Rule_Examples.json 中未找到与「应用规则」ID 完全匹配的 examples 条目时："
            "仅依据上文的规则文字描述与两 SQL 判断；不臆造示例。）"
        )
    parts: List[str] = []
    for rid in sorted(ex_map.keys(), key=str):
        items = ex_map[rid]
        parts.append(f"## 规则 `{rid}` 的知识库示例（标准「原始 → 重写」）")
        for i, ex in enumerate(items, 1):
            if not isinstance(ex, dict):
                continue
            parts.append(f"  - 示例 {i}:")
            parts.append(f"    - 原始查询: {ex.get('original_query', '')}")
            parts.append(f"    - 重写查询: {ex.get('rewritten_query', '')}")
            parts.append(f"    - 示例内说明: {ex.get('rule_description', '')}")
        parts.append("")
    return "\n".join(parts).strip()


class RewriteAgent(Agent):
    """SQL Rewrite Execution Agent"""
    def __init__(self, mq: MessageQueue):
        super().__init__("RewriteAgent", mq, gpt = GPT(
            api_key=os.getenv("REWRITE_MODEL_API_KEY"),
            model=os.getenv("REWRITE_MODEL"),
            base_url=os.getenv("REWRITE_MODEL_URL"),
            thinking_type=os.getenv("REWRITE_MODEL_THINKING"),
            reasoning_effort=os.getenv("REWRITE_MODEL_REASONING_EFFORT"),
        ))
        self.watch(["DecisionAgent", "ReasoningAgent"])

    async def rewrite_with_rule_sequence(
        self,
        sql: str,
        rule_sequence: dict,
        optimization_direction: str,
        data_statistics: str,
        schema_content: Optional[str] = None,
        index_info: str = "",
        previous_feedback: Optional[dict] = None,
    ) -> dict:
        """Execute SQL rewriting based on selected rule sequence with semantic equivalence check"""
        applied_rules = rule_sequence.get("applied_rules", [])
        groups = rule_sequence.get("groups", "")

        sch = (schema_content or "").strip()
        schema_section = (
            f"""
        <SQL Schema（与当前查询相关的表结构）>
        {sch}
        </SQL Schema>
"""
            if sch
            else ""
        )
        idx = (index_info or "").strip()
        idx_section = (
            f"""
        <索引信息>
        {idx}
        </索引信息>
"""
            if idx
            else ""
        )
        previous_feedback_section = ""
        if previous_feedback:
            previous_feedback_section = f"""
        <上一轮评估反馈（重点修复）>
        {json.dumps(previous_feedback, ensure_ascii=False, indent=2)}
        </上一轮评估反馈（重点修复）>
"""
        rule_examples = get_rule_examples(applied_rules, current_sql=sql)
        rule_examples_text = format_rule_examples_for_semantic_check(rule_examples)
        rule_examples_section = f"""
        <规则示例库_examples（按已选规则匹配，供改写参考）>
        {rule_examples_text}
        </规则示例库_examples（按已选规则匹配，供改写参考）>
"""

        prompt = textwrap.dedent(f"""
        <Mission>
        你是一名经验丰富的 DBA，你的任务是按照指定的规则序列对SQL进行重写。

        1. 输入信息：
           - <base_sql>: 当前轮次的基底 SQL。第一轮它等于原始SQL；若进入下一轮，它就是上一轮 rewritten_sql
           - <rule_sequence>: 要应用的规则序列
           - <optimization_direction>: 优化方向
           - <统计信息>: 数据库表统计信息
           - （若下方提供）<SQL Schema>: 与当前查询相关的表 DDL
           - （若下方提供）<索引信息>: 表索引信息
           - <规则示例库_examples>: 与已选规则对应的标准「原始查询 -> 重写查询」示例，只能作为模式参考，不能照抄示例中的表名、列名或谓词
           - （若下方提供）<上一轮评估反馈（重点修复）>: 未改进/恶化原因与下一步改进建议，必须优先处理

        2. 重写要求：
           - **最高优先级约束**：必须优先保证 rewritten_sql 与原始SQL在语义上完全等价；任何优化都不得以改变结果集语义为代价
           - 参考 <规则示例库_examples> 中每条规则的典型改写模式和等价性注意点，但必须结合当前 SQL、Schema 和统计信息重新判断适用性
           - 一次性完成全部规则的应用，不分步骤应用，最终输出重写 SQL**：直接写最终版本
           - 确保SQL的语法正确性
           - **必须保持查询语义的等价性**：重写后的SQL必须与原始SQL在语义上完全等价
           - 若提供了上一轮评估反馈，说明这是增量改写场景；必须在当前基底 SQL 的基础上继续优化，而不是回退并从原始 SQL 重新开始
           - 生成可执行的SQL

        3. 语义等价性检查：
           - 重写后的SQL必须返回与原始SQL相同的结果集
           - 列名、数据类型、排序顺序等必须保持一致
           - 聚合函数、窗口函数、子查询等必须保持相同的语义
        4. **semantic_correctness_guarantee（语义正确性保证说明，必填）**：
           用**中文**分条写清（建议 3–8 条，须可被审计员**逐条**对照原 SQL 与重写 SQL 独立核验），至少包含：
           - 相对原 SQL 做了哪些**结构性**改写（下推/合并/去相关/CTE 展开等），对应到哪些谓词、JOIN 或子查询作用域。
           - **为何**在 Schema 与 TPC-DS/常规 SQL 语义下，这些改写仍与原始查询**结果集**等价；涉及 GROUP BY/主键/相关子查询时，写出键与相关列。
           - 若有易误判点（如过滤提前、HAVING/ORDER BY 与 LIMIT、NULL、重复行、相关子查询相关列），**明确**你判定仍等价的理由。
           不得写与代码不一致的空话；不得省略「保证说明」字段。

        <base_sql>
        {sql}

        <rule_sequence>
        {json.dumps(rule_sequence, ensure_ascii=False, indent=2)}

        <optimization_direction>
        {optimization_direction}

        <统计信息>
        {data_statistics}
{schema_section}{idx_section}{rule_examples_section}{previous_feedback_section}
        5. **只输出一个 JSON 对象**（不要 <rewrite> 标签、不要 markdown）。字段：
           - groups: 字符串 "{groups}"
           - applied_rules: 数组，与当前序列一致：{json.dumps(applied_rules, ensure_ascii=False)}
           - rewritten_sql: 重写后的完整可执行 SQL（字符串内换行用 \\n）
           - semantic_correctness_guarantee: 上述第 4 点「语义正确性保证说明」全文（多行可放在同一字符串，用 \\n 转义换行）
        """)

        thought_chain = await self.llm.get_LLM_response_async(
            prompt=prompt,
            json_format=True,
            json_schema=STRICT_JSON_SCHEMAS["rewrite"],
        )

        try:
            parsed, _ = parse_llm_json(thought_chain, {})
            if parsed and parsed.get("rewritten_sql"):
                parsed.setdefault("groups", groups)
                parsed.setdefault("applied_rules", applied_rules)
                _merge_rewrite_semantic_fields(parsed)
                return parsed

            rewrite_match = re.search(r"<rewrite>(.*?)</rewrite>", thought_chain, re.DOTALL)
            inner = rewrite_match.group(1).strip() if rewrite_match else thought_chain
            parsed, _ = parse_llm_json(inner, {})
            if parsed and parsed.get("rewritten_sql"):
                parsed.setdefault("groups", groups)
                parsed.setdefault("applied_rules", applied_rules)
                _merge_rewrite_semantic_fields(parsed)
                return parsed

            # If JSON parsing failed, try to extract SQL from response
            print("⚠️ 无法解析JSON格式，尝试从响应中提取SQL...")
            extracted_sql = self._extract_sql_from_response_robust(inner)
            if not extracted_sql:
                extracted_sql = self._extract_sql_from_response_robust(thought_chain)
            if extracted_sql:
                print(f"✅ 成功从响应中提取SQL")
                return {
                    "groups": groups,
                    "applied_rules": applied_rules,
                    "rewritten_sql": extracted_sql,
                    "semantic_correctness_guarantee": "",
                    "semantic_check": "",
                    "parse_error": True  # Flag to indicate parsing error
                }

            return {
                "groups": groups,
                "applied_rules": applied_rules,
                "rewritten_sql": sql,  # fallback to original
                "semantic_correctness_guarantee": "",
                "semantic_check": "",
                "parse_error": True
            }
        except json.JSONDecodeError as e:
            print(f"重写结果解析错误: {e}")
            # Try to extract SQL from response even when JSON parsing fails
            print("⚠️ 尝试从响应中提取SQL...")
            extracted_sql = self._extract_sql_from_response_robust(thought_chain)
            if extracted_sql:
                print(f"✅ 成功从响应中提取SQL")
                return {
                    "groups": groups,
                    "applied_rules": applied_rules,
                    "rewritten_sql": extracted_sql,
                    "semantic_correctness_guarantee": "",
                    "semantic_check": "",
                    "parse_error": True,
                    "error_info": f"JSON解析错误: {str(e)}"
                }
            
            return {
                "groups": groups,
                "applied_rules": applied_rules,
                "rewritten_sql": sql,
                "semantic_correctness_guarantee": "",
                "semantic_check": "",
                "parse_error": True,
                "error_info": f"JSON解析错误: {str(e)}"
            }

    async def semantic_fix(
        self,
        original_sql: str,
        rewritten_sql: str,
        applied_rules: list,
        differences: list,
    ) -> str:
        """在保留优化结构前提下最小修复语义，使与原始 SQL 结果集等价。"""
        diff_text = json.dumps(differences, ensure_ascii=False, indent=2) if differences else "[]"
        rules_text = json.dumps(applied_rules, ensure_ascii=False) if applied_rules else "[]"
        prompt = textwrap.dedent(f"""
        <Mission>
        你是 DBA。当前重写 SQL 经语义检查判定与原始 SQL 不等价。
        你的任务是：**仅修复 differences 指出的差异相关片段**，而不是对整条 SQL 重新改写。
        请在**尽量保留现有优化结构**（JOIN/CTE/子查询形态）的前提下，对「当前重写 SQL」做**最小修改**，使其与原始 SQL **结果集语义等价**。

        <original_sql>
        {original_sql}

        <当前重写 SQL>
        {rewritten_sql}

        <应用规则 ID>
        {rules_text}

        <语义差异点 differences>
        {diff_text}

        修复约束（必须遵守）：
        0. **最高优先级约束**：必须优先保证修复后 SQL 与 original_sql 语义等价；禁止为了“看起来更优”而改变结果集语义。
        1. 只允许改动 differences 直接涉及的片段（相关谓词、JOIN 条件、聚合列、子查询条件等）。
        2. 不要重排与 differences 无关的 JOIN/CTE 结构，不要整体重写 SQL。
        3. 保留当前重写 SQL 中已生效的优化（除非该优化正是导致不等价的根因）。
        4. 若 differences 为空或信息不足，返回原 rewritten_sql，不要臆造新重写。
        5. 输出必须是**可执行完整 SQL**，且与原始 SQL 结果集等价。

        输出要求：**只输出一个 JSON 对象**（不要 markdown），格式：
        {{"fixed_sql": "修复后的完整 SQL 单行或合理换行", "note": "一句话说明如何修复"}}
        """)
        resp = await self.llm.get_LLM_response_async(
            prompt=prompt,
            json_format=True,
            json_schema=STRICT_JSON_SCHEMAS["semantic_fix"],
        )
        data = parse_llm_json_with_default(resp, {})
        fixed = data.get("fixed_sql") or data.get("rewritten_sql") or ""
        return fixed.strip() if isinstance(fixed, str) else ""

    async def iterative_rewrite(
        self,
        sql: str,
        error_info: str,
        previous_rewrite: dict,
        data_statistics: Optional[str] = None,
        index_info: Optional[str] = None,
    ) -> str:
        """Iteratively rewrite SQL based on error feedback"""
        # Extract previous SQL and error history if available
        previous_sql = previous_rewrite.get("rewritten_sql", "")
        previous_error = previous_rewrite.get("error_info", "")
        stats_block = f"\n        <统计信息>\n        {data_statistics}\n" if data_statistics else ""
        idx_block = f"\n        <索引信息>\n        {index_info}\n" if index_info else ""

        prompt = textwrap.dedent(f"""
        <Mission>
        你是一名经验丰富的 DBA，你的任务是基于语法错误信息修正重写后的SQL。

        **重要**：请仔细阅读并分析 <error_info> 中的错误信息，这是数据库返回的具体语法错误，必须针对这个错误进行修正。

        1. 输入信息：
           - <original_sql>: 原始SQL（用于保持语义等价性）
           - <error_info>: **当前语法检查错误信息**（这是你需要修复的具体错误）
           - <previous_rewrite>: 上次重写结果（包含之前尝试的SQL和可能的错误信息）
        {stats_block}{idx_block}

        2. 修正要求：
           - **最高优先级约束**：必须优先保证 rewritten_sql 与 original_sql 的语义等价；语法修复不得引入语义漂移
           - **仔细分析 <error_info> 中的错误信息**，这是数据库返回的具体语法错误位置和原因
           - 如果 <previous_rewrite> 中包含之前的错误信息，参考它们避免重复同样的错误
           - 仅做“最小修复”：只修改触发错误的必要片段（如列引用、别名、GROUP BY项、函数参数）
           - 尽量保留 current rewritten SQL 的优化结构（CTE 组织、JOIN 形态、聚合层次、谓词布局），不要整体改写
           - 禁止无关改动：不要新增/删除非必要子查询、不要替换为原始 SQL，除非当前 SQL 无法通过最小修复挽救
           - 修正SQL语法错误，确保修正后的SQL能够通过语法检查
           - **必须保持查询语义正确性**：修正后的SQL必须与原始SQL在语义上完全等价
           - 生成可执行的、语法正确的SQL

        3. 输出要求：
           - **只输出一个 JSON 对象**（不要 markdown / 不要额外文本）
           - 必须包含字段 rewritten_sql，值为修正后的完整可执行 SQL
           - 可选字段 note 用一句话说明修复点
           - 示例：{{"rewritten_sql":"SELECT ...","note":"修复了别名不存在错误"}}

        <original_sql>
        {sql}

        <error_info>
        {error_info}

        <previous_rewrite>
        {json.dumps(previous_rewrite, ensure_ascii=False, indent=2)}
        """)

        response = await self.llm.get_LLM_response_async(
            prompt=prompt,
            json_format=True,
            json_schema=STRICT_JSON_SCHEMAS["iterative_rewrite"],
        )
        data = parse_llm_json_with_default(response, {})
        rewritten_sql = data.get("rewritten_sql") if isinstance(data, dict) else ""
        if isinstance(rewritten_sql, str) and rewritten_sql.strip():
            return rewritten_sql.strip()
        # Fallback for non-compliant model outputs.
        return self._extract_sql_from_response_robust(response)
    
    async def correct_sql(self, original_sql: str, rewritten_sql: str, error: str) -> str:
        """Correct SQL syntax errors while maintaining semantic equivalence"""
        prompt = textwrap.dedent(f"""
        你是一名经验丰富的 DBA，核心任务是修正 SQL 语法错误，同时保持语义等价性。
        **最高优先级约束**：必须优先保证修复后 SQL 与 original_sql 语义一致；不得为了通过语法而改变结果集语义。
        1. 修正以下 SQL 语句，使用提供的错误消息。

            * 注意：如果 SQL 包含双引号，请保留它们 exactly as they appear.
            <rewritten_sql>
            {rewritten_sql}

           错误消息:
            {error}

            下面是重写 SQL 的原始形式，用于参考模式。
            只修正 <rewritten_sql> 中的语法；不要与原始 SQL 对齐，
            且不要包含任何 "EXPLAIN (FORMAT JSON)" 子句！
            **重要**：修正后的SQL必须与原始SQL在语义上等价。

            <original_sql>
            {original_sql}

            请严格遵循以下格式：
            [format]
            </analysis>
            分析错误原因和修正方案
            </analysis>

            </corrected_sql>
            ```sql
            插入修正后的 SQL 语句。
            ```
            </corrected_sql>
            """)

        response = await self.llm.get_LLM_response_async(prompt=prompt)
        corrected_sql = self.extract_corrected_sql_content(response)
        return corrected_sql
    
    def extract_corrected_sql_content(self, text: str) -> str:
        """
        Extract content between </corrected_sql> and </corrected_sql> tags.
        """
        # First, try to match the format with ```sql
        pattern1 = r'</corrected_sql>\s*```sql\s*(.*?)\s*```\s*</corrected_sql>'
        match1 = re.search(pattern1, text, re.DOTALL)
        if match1:
            return match1.group(1).strip()

        # If no match is found, try to match the plain text format without ```
        pattern2 = r'</corrected_sql>\s*(.*?)\s*</corrected_sql>'
        match2 = re.search(pattern2, text, re.DOTALL)
        if match2:
            sql_content = match2.group(1).strip()
            # Remove any ```sql markers
            sql_content = re.sub(r'^```sql\s*', '', sql_content)
            sql_content = re.sub(r'\s*```$', '', sql_content)
            return sql_content.strip()
        
        print(f"Warning: Could not extract corrected SQL from response: {text[:200]}...")
        return ""

    def _extract_sql_from_response_robust(self, text: str) -> str:
        """Robust SQL extraction from LLM response"""
        # First, try to extract from JSON "rewritten_sql" field
        rewritten_sql_pattern = r'"rewritten_sql"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
        rewritten_match = re.search(rewritten_sql_pattern, text, re.DOTALL)
        if rewritten_match:
            sql_content = rewritten_match.group(1)
            try:
                # Use json.loads to properly unescape the JSON string
                sql_content = json.loads(f'"{sql_content}"')
                if sql_content.strip():
                    return sql_content.strip()
            except json.JSONDecodeError:
                # Fallback: basic unescaping
                sql_content = sql_content.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                if sql_content.strip():
                    return sql_content.strip()

        # Try to find SQL between ```sql and ```
        pattern1 = r'```sql\s*(.*?)\s*```'
        match1 = re.search(pattern1, text, re.DOTALL | re.IGNORECASE)
        if match1:
            return match1.group(1).strip()

        # Try to match the format without ```
        pattern2 = r'</rewritten_sql>\s*(.*?)\s*</rewritten_sql>'
        match2 = re.search(pattern2, text, re.DOTALL)
        if match2:
            sql_content = match2.group(1).strip()
            sql_content = re.sub(r'^```sql\s*', '', sql_content)
            sql_content = re.sub(r'\s*```$', '', sql_content)
            return sql_content.strip()

        # Last resort: look for WITH (CTE) or SELECT keyword
        # First try to match complete WITH statement
        with_match = re.search(r'(WITH.*?);', text, re.DOTALL | re.IGNORECASE)
        if with_match:
            return with_match.group(1).strip() + ";"

        # Fallback to SELECT keyword
        select_match = re.search(r'(SELECT.*?);', text, re.DOTALL | re.IGNORECASE)
        if select_match:
            return select_match.group(1).strip() + ";"

        print(f"Warning: Could not extract rewritten SQL from response")
        return ""


class SemanticCheckAgent(Agent):
    """语义等价检查：结构化 JSON 输出，供 LangGraph 管线使用。"""

    def __init__(self, mq: MessageQueue):
        api_key = os.getenv("SEMANTIC_CHECK_MODEL_API_KEY") 
        model = os.getenv("SEMANTIC_CHECK_MODEL")
        base_url = os.getenv("SEMANTIC_CHECK_MODEL_URL")
        super().__init__(
            "SemanticCheckAgent",
            mq,
            gpt=GPT(
                api_key=api_key,
                model=model,
                base_url=base_url,
                thinking_type=os.getenv("SEMANTIC_CHECK_MODEL_THINKING"),
                reasoning_effort=os.getenv("SEMANTIC_CHECK_MODEL_REASONING_EFFORT"),
            ),
        )

    def _parse_equivalence_response(self, response: str) -> dict:
        default = {
            "equivalent": False,
            "message": "解析失败",
            "differences": [{"type": "解析", "description": "无法解析 LLM 输出", "location": "N/A"}],
        }
        if not response or not str(response).strip():
            return default
        data = parse_llm_json_with_default(str(response), default)
        eq = data.get("equivalent", False)
        diffs = data.get("differences") or []
        if not isinstance(diffs, list):
            diffs = []
        if not diffs:
            eq = True
        return {
            "equivalent": bool(eq),
            "message": data.get("message", ""),
            "differences": diffs,
        }

    async def check_equivalence(
        self,
        original_sql: str,
        rewritten_sql: str,
        rewrite_rules: list,
        semantic_check: Optional[str] = None,
        semantic_correctness_guarantee: Optional[str] = None,
        schema_content: Optional[str] = None,
        index_info: Optional[str] = None,
        *,
        semantic_check_attempt: int = 0,
        last_rejected_rewritten_sql: Optional[str] = None,
        last_rejection_message: Optional[str] = None,
        last_rejection_differences: Optional[List[Dict[str, Any]]] = None,
    ) -> dict:
        # Build "规则ID + 规则描述" text for semantic audit context.
        if rewrite_rules:
            rule_kb = load_rule_knowledge_base()
            rule_desc_map: Dict[str, str] = {}
            if isinstance(rule_kb, dict):
                for category_data in rule_kb.values():
                    if not isinstance(category_data, dict):
                        continue
                    rules_block = category_data.get("rules", {})
                    if isinstance(rules_block, dict):
                        for rid, desc in rules_block.items():
                            rule_desc_map[str(rid)] = str(desc)
            rule_lines = []
            for rid in rewrite_rules:
                rid_s = str(rid)
                desc = rule_desc_map.get(rid_s, "（规则描述缺失）")
                rule_lines.append(f"- {rid_s}: {desc}")
            rules_text = "\n".join(rule_lines)
        else:
            rules_text = "无"
        rule_examples = get_rule_examples(rewrite_rules or [], current_sql=original_sql)
        rule_examples_text = format_rule_examples_for_semantic_check(rule_examples)
        rule_examples_ctx = textwrap.dedent(f"""

            <应用规则_examples（知识库标准示例，仅作等价模式参考）>
            {rule_examples_text}
            </应用规则_examples（知识库标准示例，仅作等价模式参考）>
            """)
        g = (str(semantic_correctness_guarantee).strip() if semantic_correctness_guarantee else "")
        s0 = (str(semantic_check).strip() if semantic_check is not None else "")
        # 新字段「语义正确性保证说明」优先；否则退化为旧名 semantic_check（同一段说明）。
        sem_text = g or s0
        attempt = int(semantic_check_attempt or 0)
        if sem_text:
            if attempt > 0:
                sem_block_title = "首轮「语义正确性保证说明」（自辩可能在后续修正中已部分失效；**等价性以当前<当前重写 SQL>为准，勿将首轮自辩与当前 SQL 矛盾时仍采信自辩**）"
            else:
                sem_block_title = "重写代理的语义正确性保证说明"
            sem_ctx = textwrap.dedent(f"""

            <{sem_block_title}>
            {sem_text}
            </{sem_block_title}>
            """)
        else:
            sem_ctx = "\n（未提供重写代理的「语义正确性保证说明」：须仅据两条 SQL 与 Schema 直接推理。）\n"

        if (
            attempt > 0
            and last_rejected_rewritten_sql
            and str(last_rejected_rewritten_sql).strip()
        ):
            diffs_s = (
                json.dumps(last_rejection_differences or [], ensure_ascii=False, indent=2)
                if last_rejection_differences
                else "[]"
            )
            retry_ctx = textwrap.dedent(f"""

            <语义修正_第{attempt + 1} 轮_必读>
            当前不是首轮检查：上一轮对 **另一份** 重写 SQL 判为不等价。下列内容用于你理解「**相对上一轮文字** 改了什么」，**不是**要求你继续用同一条首轮自辩来否定当前已经变化过的 SQL。

            --- 上轮被否定的「重写 SQL」（与下方 <当前重写 SQL> 可能不同，请逐字对比）---
            {last_rejected_rewritten_sql}
            --- 上轮结论 ---
            {last_rejection_message or "（无）"}
            --- 上轮指出的差异点（JSON）---
            {diffs_s}

            裁决要求：
            1) **以 <原始 SQL> 与 <当前重写 SQL> 的语义为唯一真值来源**；Schema / 关键约束为辅助。
            2) 首轮「语义正确性保证说明」中若与**当前**重写 SQL 不一致，**以当前 SQL 与事实为准**；不得用「自辩与旧版配合」的论证来否掉已按你方上次意见修正过的新版。
            3) 若可看出当前版相对「上轮被否 SQL」的语法/逻辑改动，可评估这些改动**是否**消除了上轮差异点；**不得**因首轮自辩中某条仍成立、而**当前**两 SQL 实际已一致，仍判不等价。
            </语义修正_第{attempt + 1} 轮_必读>
            """)
        else:
            retry_ctx = ""
        schema_ctx = (
            textwrap.dedent(f"""

            <SQL Schema（表结构及约束，判断等价时必用）>
            {schema_content.strip()}
            </SQL Schema>
            """)
            if schema_content and schema_content.strip()
            else "\n（未提供 Schema：仅基于 SQL 文本推理，对不确定处从宽。）\n"
        )
        index_ctx = (
            textwrap.dedent(f"""

            <索引信息（库中索引定义，可选参考）>
            {str(index_info).strip()}
            </索引信息>
            """)
            if index_info and str(index_info).strip()
            else ""
        )
        prior_note = (
            "若存在<语义修正_第N轮_必读>：当前「当前重写 SQL」可能已多轮更新；**不得以首轮自辩的过时论点重复否定**，须针对**当前**两 SQL 与（若有）上轮被否 SQL 的对比下结论。"
            if retry_ctx
            else "可将首轮自辩与两 SQL 交叉核验；自辩与**当前**重写 SQL 或事实矛盾时，以**当前** SQL 与事实为准，不得采信过时的自辩句段。"
        )
        prompt = textwrap.dedent(f"""
        你是 SQL 语义审计专家，你要分析判断原始SQL和重写SQL是否等价。等价含义：**同一库状态、同一参数下，两查询结果集相同（行集合与列语义一致；允许列名/顺序在逻辑上等价时的合理差异，但若会改变行数或聚合语义则不等价）**。

        输入优先级（**{prior_note}**）：
        1. **原始 SQL** 与 **当前重写 SQL** 的实际语义（谓词、JOIN、GROUP BY、DISTINCT、子查询相关性、NULL 处理等）——**永远最高**。
        2. **SQL Schema**：主键、唯一约束、函数依赖、可据此认可的等价变形（例如已知 PK 下 GROUP BY 的化简）。
        3. **索引信息**（若提供）：通常不改变关系层面的结果集语义；PRIMARY KEY/UNIQUE 类索引可辅助推断唯一性，与 Schema 一并用于等价推理。**不得以「有无非唯一索引」代替 SQL 逻辑判断是否等价**。
        4. **首轮「语义正确性保证说明」**（若提供）：**仅**作辅助线索，须与 1 逐条核对。若存在「语义修正_第N轮_必读」块，说明**当前重写 SQL 可能已按你方上一轮意见改过**——**禁止**用首轮自辩中仅适用**历史版本**的论述来否掉**已经变化后**的当前 SQL；自辩与**当前**重写 SQL 明显不符时，**忽略**不适用的自辩段。
        5. **应用规则（ID + 文字描述）**：本查询**声称**依这些规则做等价改写。当「当前重写 SQL」可判定为**完全按**规则描述所体现的典型变换意图忠实套用（同构的改写模式、未在规则意图之外缩小/扩大过滤范围、未擅自改变聚合/分组/去重语义），且与 **Schema/约束** 无矛盾时，应将其视为**强等价先验**：默认倾向 `equivalent=true`，除非你能给出**明确且可核验的结果集变化证据**。
        6. **应用规则 examples**：若提供了知识库标准示例，可参考其中「原始 -> 重写」的等价模式和注意点；但示例中的表名、列名和谓词只是模板，不得替代对当前两条 SQL 的实际语义判断。

        通用原则：
        - **规则驱动等价先验**：在 Schema 不否定结论的前提下，若重写在结构上等价于「对原 SQL 应用规则 R」且与 R 的文字描述所示变换意图一致、无多余语义偏移，应默认采纳 `equivalent=true`；只有当你能给出**明确、具体、可复核**的反例时，才允许推翻这一先验。
        - **举证责任原则**：仅当**能明确指出**会导致结果集不一致的差异时才允许判 `equivalent=false`。所谓“明确指出”是指：必须说明差异如何改变过滤范围、连接基数、重复行、NULL 语义、聚合粒度、DISTINCT/去重语义、排序/限制对结果集的影响等。若只是“结构不同”“看起来复杂”“可能有风险”，不足以判 false。
        - **结构变化不等于语义变化**：CTE 引入、相关子查询改写为 JOIN、过滤条件从 WHERE 移入 JOIN ON、常量折叠、冗余排序消除、预聚合/公共结果复用，本身都**不能**作为判 false 的依据；必须证明这些变换改变了最终结果集。
        - **证据不足时从宽**：如果你不能构造出具体结果集差异，就应判 `equivalent=true`，而不是因为怀疑或保守而判 false。
        - Schema 中有明确定义时，采纳与约束一致的等价推理（如主键列 A 下，GROUP BY A 与 GROUP BY A,B 且 B 函数依赖于 A）。
        - 对同一张表、没有额外过滤/去重/DISTINCT 的情况下，先按任意维度 GROUP BY 再对每组 COUNT(col) 并在外层 SUM(count_col)，
  与直接对整表 COUNT(col) 结果相同。
        - 两层聚合折叠规则（强约束）：若原SQL形如「内层按 (K,U) 分组，外层按 K 分组」，且重写为「按 K 单层聚合」，并且每个外层聚合列都满足以下逐列等价之一：
          MIN(MIN(v))->MIN(v)，MAX(MAX(v))->MAX(v)，SUM(SUM(v))->SUM(v)，SUM(COUNT(v))->COUNT(v)，则**必须判 equivalent=true**。
        - 禁止误判：在满足上述强约束时，不能使用“计数对象不同”“中间分组行数不同”“内层多了 ename”作为不等价理由；这些只影响中间态，不影响最终结果。
        - 若判 `equivalent=false`，`differences` 中每一项都必须对应一个**会改变最终结果集**的具体点；不得输出空泛差异。若做不到，请改判 `equivalent=true`。

        <原始 SQL>
        {original_sql}

        <当前重写 SQL>
        {rewritten_sql}
        {retry_ctx}
        {schema_ctx}{index_ctx}
        {sem_ctx}
        <应用规则_规则ID与文字描述>
        {rules_text}
        </应用规则_规则ID与文字描述>
        {rule_examples_ctx}

        **只输出一个 JSON**（不要其它文字）：
        {{
            "equivalent": true 或 false,
            "message": "一句话。首轮可简述自辩与事实；若存在多轮修正，请对**当前两 SQL**下结论，勿复述过时的首轮自辩。",
            "differences": [{{"type": "类型", "description": "一句话", "location": "位置"}}]
        }}
        equivalent 为 true 时 differences 必须为 []。
        """)
        response = await self.llm.get_LLM_response_async(
            prompt=prompt,
            json_format=True,
            json_schema=STRICT_JSON_SCHEMAS["semantic_equivalence"],
        )
        return self._parse_equivalence_response(response)


class SingleAgent(Agent):
    """
    Architecture ablation: one agent + one LLM replaces Decision / Reasoning /
    Rewrite / SemanticCheck agents. Method bodies are reused via class binding.
    """

    def __init__(self, mq: MessageQueue):
        super().__init__("SingleAgent", mq, gpt=_build_single_agent_gpt())

    # Decision
    initial_optimization_check = DecisionAgent.initial_optimization_check
    evaluate_with_costs = DecisionAgent.evaluate_with_costs

    # Reasoning
    select_rule_sequence = ReasoningAgent.select_rule_sequence

    # Rewrite
    rewrite_with_rule_sequence = RewriteAgent.rewrite_with_rule_sequence
    semantic_fix = RewriteAgent.semantic_fix
    iterative_rewrite = RewriteAgent.iterative_rewrite
    correct_sql = RewriteAgent.correct_sql
    extract_corrected_sql_content = RewriteAgent.extract_corrected_sql_content
    _extract_sql_from_response_robust = RewriteAgent._extract_sql_from_response_robust

    # Semantic check
    check_equivalence = SemanticCheckAgent.check_equivalence
    _parse_equivalence_response = SemanticCheckAgent._parse_equivalence_response

