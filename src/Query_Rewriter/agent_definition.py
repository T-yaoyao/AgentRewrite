
import json
import os
import re
import textwrap
import sys
from typing import Any, Dict, List, Optional

# Setup project paths
from src.utils.path_config import setup_python_path, load_project_env
setup_python_path()
load_project_env()

from src.utils.agent_template import MessageQueue, Agent
from src.utils.llm_client import GPT
from src.utils.llm_json_utils import parse_llm_json, parse_llm_json_with_default

STRICT_JSON_SCHEMAS = {
    "rule_selection": {
        "name": "rule_selection_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "applied_rules": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["applied_rules"],
            "additionalProperties": False,
        },
    },
    "evaluation": {
        "name": "rewrite_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "keep": {"type": "boolean"},
                "terminate": {"type": "boolean"},
                "reason": {"type": "string"},
                "next_step_advice": {"type": "string"},
            },
            "required": ["keep", "terminate", "reason", "next_step_advice"],
            "additionalProperties": False,
        },
    },
    "rewrite": {
        "name": "rewrite_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "applied_rules": {"type": "array", "items": {"type": "string"}},
                "step_sqls": {"type": "array", "items": {"type": "string"}},
                "semantic_correctness_guarantee": {"type": "string"},
            },
            "required": [
                "applied_rules",
                "step_sqls",
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

def _rewritten_sql_from_steps(parsed: dict, base_sql: str) -> bool:
    """最终 SQL 取 step_sqls 最后一项。空序列表示未改写，沿用 base_sql。"""
    parsed.pop("rewritten_sql", None)
    steps = parsed.get("step_sqls")
    if not isinstance(steps, list):
        return False
    if not steps:
        parsed["rewritten_sql"] = base_sql
        return True
    last = steps[-1]
    if not isinstance(last, str) or not last.strip():
        return False
    parsed["rewritten_sql"] = last.strip()
    return True


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


def _rule_catalog_entry(rule_data):
    """推理 agent 只接收规则描述。示例留在重写和等价判断。"""
    score = None
    desc = rule_data
    if isinstance(rule_data, tuple) and len(rule_data) >= 2:
        desc = rule_data[0]
        if isinstance(rule_data[1], (int, float)) and not isinstance(rule_data[1], bool):
            score = float(rule_data[1])
    if isinstance(desc, dict):
        desc = desc.get("description") or ""
    return str(desc or ""), score


def _stable_rule_catalog(rule_library: dict) -> tuple:
    """固定 rule_id 顺序的规则目录，以及放到 user 后缀里的 UCT 分数。"""
    rows = []
    scores = []
    for rule_id, rule_data in (rule_library or {}).items():
        desc, score = _rule_catalog_entry(rule_data)
        rows.append((str(rule_id), desc))
        if score is not None:
            scores.append((str(rule_id), score))
    rows.sort(key=lambda item: item[0])
    scores.sort(key=lambda item: item[0])
    body = "\n".join(f"- {rid}: {desc}" for rid, desc in rows)
    return body, scores


class ReasoningAgent(Agent):
    """MDP-based Reasoning Agent"""
    def __init__(self, mq: MessageQueue):
        super().__init__("ReasoningAgent", mq, gpt = GPT(
        api_key=os.getenv("REASONING_MODEL_API_KEY"),
        model=os.getenv("REASONING_MODEL"),
        base_url=os.getenv("REASONING_MODEL_URL"),
        thinking_type=os.getenv("REASONING_MODEL_THINKING"),
        reasoning_effort=os.getenv("REASONING_MODEL_REASONING_EFFORT"),
        ))

    async def select_rule_sequence(self, sql: str, rule_library: dict, data_statistics: str, explain_info: str, iteration_round: int = 1, previous_feedback: dict = None, few_shot_examples: list = None, index_info: str = "") -> dict:
        """根据 PlanAnalyzer 的执行计划分析，从全量规则库选择规则序列。"""
        rule_text, uct_scores = _stable_rule_catalog(rule_library)

        previous_feedback_text = ""
        if previous_feedback:
            previous_feedback_text = (
                f"<上一轮反馈>\n"
                f"此 SQL 已经经过第{iteration_round - 1}轮优化，但评估结果为需要继续优化。\n"
                f"{json.dumps(previous_feedback, ensure_ascii=False, indent=2)}\n"
                f"请基于上一轮的经验和失败原因重新选择规则序列，避免重复同样的错误。\n"
                f"</上一轮反馈>"
            )

        few_shot_context = ""
        if few_shot_examples:
            lines = [
                "<相似历史案例>",
                "以下是历史上相似 SQL 的优化案例，供参考（可以参考思路，但需要根据当前 SQL 选择规则）：",
            ]
            for i, example in enumerate(few_shot_examples[:3], 1):
                lines.append(f"案例{i}:")
                lines.append(f"- 原始SQL: {example.get('original_sql', '')}")
                lines.append(f"- 规则ID: {example.get('rule_sequence', [])}")
            lines.append("</相似历史案例>")
            few_shot_context = "\n".join(lines)

        if uct_scores:
            score_lines = "\n".join(f"- {rid}: {score:.3f}" for rid, score in uct_scores)
            uct_block = (
                "<uct_scores>\n"
                "下列分数是各规则在当前上下文上的 LinUCB 分数，不是规则说明的一部分。\n"
                "可以优先考虑高分规则，但必须结合当前 SQL 结构，不要只按分数选择。\n"
                f"{score_lines}\n"
                "</uct_scores>"
            )
        else:
            uct_block = "<uct_scores>\n本轮未提供 UCT 分数。\n</uct_scores>"

        system_message = textwrap.dedent(f"""
        <Mission>
        你是一名经验丰富的 DBA。你是规则选择的第一个模型：根据 PlanAnalyzer 给出的执行计划分析、当前 SQL 和统计信息，从全量规则库中挑选优化规则序列。你只负责选择规则，不进行任何重写操作，也不要等待别的 agent 预先指定优化组。

        1. 规则选择原则：
           - 只能选择与当前SQL结构相符、并且有助于改善执行计划、降低执行时间的规则，或者能够简化原始查询逻辑表达式的规则。
           - 当查询语句包含复杂的WHERE/JOIN条件，或存在重复子查询计算时，可考虑使用公共表表达式（CTE）
           - 若查询语句本身结构简单，或使用CTE无法减少冗余计算，则应避免过度使用CTE，多余的CTE可能会增加系统开销
        2. 规则选择逻辑：
           - 先读 <执行计划分析结果>，定位高代价算子和结构瓶颈
           - 再对照当前 SQL、统计信息和索引，从全量规则库中找出能一起解决这些瓶颈的规则
           - 考虑规则之间的依赖和配合：谓词下推、连接改写、子查询去相关、聚合或投影清理如果要先后作用，就按应用顺序放进同一条序列
           - 目标是一条重写规则序列，使后续重写能够优化执行计划并降低执行时间。存在多个互补规则时不要只选一条
           - UCT 分数相同或都接近冷启动时，不要默认选目录里靠前的那一条
           - 计划分析或者原始SQL已经指出重复子查询、过滤偏晚或可折叠常量时，必须选出针对该瓶颈的规则，applied_rules 不能返回 []
           - 按执行计划里的算子结构选规则，不要按表名选。相关子查询被逐行重复计算时，选能按相关键计算一次再连接的规则，不要选会把一次计算改回逐行执行的规则
           - 只有对聚合结果的每一次使用都成立的过滤才能下推到聚合前。不成立时，不要选靠复制这段聚合来下推的规则
           - 半连接用来先缩小输入。不要选会让同一张大表先被半连接扫描、再被相同键的 JOIN 再扫描一次的规则
           - 同一过滤或同一标量子查询出现多次时，选能让它只计算一次再被多处使用的规则。选择性高的条件在小表上、大表还要继续连接时，选能先得到小结果再连接大表的规则
           - 没有这类瓶颈、也确实没有任何规则能改善当前计划或者能够简化原始查询逻辑表达式时，applied_rules 才返回 []
        3. 输出要求：**只输出一个 JSON 对象**（不要 XML 标签、不要 markdown）。字段：
        - applied_rules: 字符串数组，按应用顺序列出规则 ID；必须来自 <rule_library>，禁止编造；若无适用规则则为 []

        示例：{{"applied_rules": ["RULE_ID_1", "RULE_ID_2"]}}

        <rule_library>
        {rule_text}
        </rule_library>
        """).strip()

        prompt = textwrap.dedent(f"""
        <统计信息>
        {data_statistics}

        <索引信息>
        {index_info}

        <sql语句（当前轮次基底 SQL；若为第2轮及以后，则是上一轮 rewritten_sql）>
        {sql}

        <执行计划分析结果>
        {explain_info}
        """).strip()
        if few_shot_context:
            prompt = f"{prompt}\n\n{few_shot_context}"
        if previous_feedback_text:
            prompt = f"{prompt}\n\n{previous_feedback_text}"
        prompt = f"{prompt}\n\n{uct_block}"

        thought_chain = await self.llm.get_LLM_response_async(
            prompt=prompt,
            system_message=system_message,
            json_format=True,
            json_schema=STRICT_JSON_SCHEMAS["rule_selection"],
        )

        default = {
            "applied_rules": [],
            "chosen_rule_prior": [],
        }
        parsed, _ = parse_llm_json(thought_chain, default)
        if parsed and isinstance(parsed.get("applied_rules"), list):
            parsed["chosen_rule_prior"] = []
            parsed.pop("groups", None)
            parsed.pop("rule_effect_scores", None)
            parsed.pop("rule_effect_confidence", None)
            return parsed
        sequence_match = re.search(r"<rule_sequence>(.*?)</rule_sequence>", thought_chain, re.DOTALL)
        inner = sequence_match.group(1).strip() if sequence_match else thought_chain
        parsed2, _ = parse_llm_json(inner, default)
        if parsed2 and isinstance(parsed2.get("applied_rules"), list):
            parsed2["chosen_rule_prior"] = []
            parsed2.pop("groups", None)
            parsed2.pop("rule_effect_scores", None)
            parsed2.pop("rule_effect_confidence", None)
            return parsed2
        return default


class EvaluationAgent(Agent):
    """评估本轮重写是否保留。"""
    def __init__(self, mq: MessageQueue):
        super().__init__("EvaluationAgent", mq, gpt = GPT(
    api_key=os.getenv("EVALUATION_MODEL_API_KEY"),
    model=os.getenv("EVALUATION_MODEL"),
    base_url=os.getenv("EVALUATION_MODEL_URL"),
    thinking_type=os.getenv("EVALUATION_MODEL_THINKING"),
    reasoning_effort=os.getenv("EVALUATION_MODEL_REASONING_EFFORT"),
))
        self.watch(["ReasoningAgent", "ExplainAgent"])

    async def evaluate_with_costs(self, optimization_info: dict, iteration_round: int = 1) -> dict:
        """Evaluate rewrite quality and whether to terminate optimization loop."""
        original_costs = optimization_info.get("original_costs", 0)
        rewritten_costs = optimization_info.get("rewritten_costs", 0)
        applied_rules = optimization_info.get("applied_rules", [])
        original_sql = optimization_info.get("original_sql", "")
        rewritten_sql = optimization_info.get("rewritten_sql", "")
        original_plan_analysis = optimization_info.get("original_plan_analysis", "")
        rewritten_plan_analysis = optimization_info.get("rewritten_plan_analysis", "")
        original_scan_rows = optimization_info.get("original_scan_rows", "")
        rewritten_scan_rows = optimization_info.get("rewritten_scan_rows", "")
        original_plan_depth = optimization_info.get("original_plan_depth", "")
        rewritten_plan_depth = optimization_info.get("rewritten_plan_depth", "")

        system_message = textwrap.dedent("""
        你负责判断本轮 SQL 重写是否保留，以及是否终止优化。依据估计代价、<scan_rows>、<plan_depth> 和两份 PlanAnalyzer 对比判断。优化器估计代价可能不准，小幅上涨本身不构成性能恶化。<scan_rows> 是各节点 Plan Rows 的累加，表示计划要处理的估计扫描行数，不是 Total Cost。<plan_depth> 是执行计划树的最大深度。数量级代价上涨指 rewritten_cost 达到 original_cost 的 10 倍或以上。

        只要满足下面任一条件，就保留当前 SQL 并终止：keep=true，terminate=true，next_step_advice 写“无需下一步优化”。
        1. 估计代价降低。
        2. <scan_rows> 中重写侧的估计扫描行数低于原始侧，且估计代价没有数量级上涨。在这个前提下，行数下降优先于小幅代价上升，也优先于连接方式变化。Hash Join 变成 Nested Loop，但内侧是索引扫描且扫描行数下降，仍然保留。扫描行数下降不能压过数量级的代价上涨：rewritten_cost >= 10 * original_cost 时，不得因 scan_rows 下降而保留。
        3. <plan_depth> 中重写侧的树深低于原始侧，或重写后的执行计划结构相对更优，执行时间会更短。包括但不限于减少了重复扫描、把过滤提前，或减少了聚合与连接的输入行数。本条不能压过数量级的代价上涨。
        4. 估计代价没有降低，但逻辑简化。包括但不限于常量折叠、去掉冗余谓词、过滤下推等。本条不适用于数量级代价上涨。
        5. 估计代价上升，但重写去掉了原始计划中的相关 SubPlan：相关标量子查询不再对每一行外层输入重复执行，而是按相关键分组计算一次，再连接回去。这种去相关即使估计代价上升、scan_rows 上升，也不要丢弃。若上涨已达数量级，则本条不成立，按下面的明显极大恶化处理。把已经按相关键分组的计算改回逐行相关 SubPlan，不属于本条。

        只有当前重写 SQL 相对原始 SQL 性能明显极大恶化时，才不保留并继续下一轮重新优化：keep=false，terminate=false。next_step_advice 只写下一轮可执行的 SQL 重写，下一轮在当前 rewritten_sql 上继续，不退回 original_sql。明显极大恶化满足下面任一情况：
        - 估计代价数量级上涨（rewritten_cost >= 10 * original_cost）。此时即使 <scan_rows> 下降、树深下降，或连接从 Hash Join 变成索引 Nested Loop，也不保留。
        - 估计代价大幅上升，并且 <scan_rows> 上升或计划明显做了更多工作，同时没有去掉相关 SubPlan。更多工作指同一张大表被多扫，或大表本身成为 Nested Loop 的内表并被反复扫描，或把已经按相关键分组的计算改回逐行相关 SubPlan。
        去掉相关 SubPlan 且代价上涨未达数量级时，不得判为极大恶化。扫描行数下降时，不得仅因 Hash Join 变成 Nested Loop 或树深变化判为极大恶化；数量级代价上涨本身就是极大恶化。

        忽略空白和末尾分号后，<original_sql> 与 <enhanced_sql> 相同，说明本轮没有改写。此时仍有查询重写空间则 terminate=false，并写下一步可重写建议；没有可重写空间，或只剩下索引、统计信息等物理手段，则 terminate=true，next_step_advice 写“无需下一步优化”。

        严格仅输出以下 JSON（不得有任何额外文本）：
        {
          "keep": true/false,
          "terminate": true/false,
          "reason": "写明 original_cost、rewritten_cost、涨幅，以及 <scan_rows> 和 <plan_depth> 的前后变化，并说明命中哪一条：代价降低；扫描行数下降且未达数量级代价上涨；树深下降或计划结构更优；逻辑简化；去掉相关 SubPlan；或性能明显极大恶化（含数量级代价上涨）。两条 SQL 一致时，写明是否还有查询重写空间。",
          "next_step_advice": "terminate 为 true 时写“无需下一步优化”；terminate 为 false 时写下一轮的 SQL 重写建议。"
        }
        """).strip()

        prompt = textwrap.dedent(f"""
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

        <scan_rows>
        original: {original_scan_rows}
        rewritten: {rewritten_scan_rows}
        </scan_rows>

        <plan_depth>
        original: {original_plan_depth}
        rewritten: {rewritten_plan_depth}
        </plan_depth>

        <original_plan_analysis>
        {original_plan_analysis}
        </original_plan_analysis>

        <rewritten_plan_analysis>
        {rewritten_plan_analysis}
        </rewritten_plan_analysis>

        <optimization_context>
        round: {iteration_round}
        applied_rules:
        {applied_rules}
        </optimization_context>
        """).strip()

        response = await self.llm.get_LLM_response_async(
            prompt=prompt,
            system_message=system_message,
            json_format=True,
            json_schema=STRICT_JSON_SCHEMAS["evaluation"],
        )

        fallback = {
            "keep": False,
            "terminate": True,
            "reason": "无法解析响应，默认终止优化",
            "next_step_advice": "无需下一步优化。",
        }
        result = parse_llm_json_with_default(response, fallback)
        if not isinstance(result, dict):
            return fallback
        result.setdefault("keep", False)
        result.setdefault("terminate", True)
        result.setdefault("reason", "")
        result.setdefault("next_step_advice", "")
        result["keep"] = bool(result.get("keep"))
        result["terminate"] = bool(result.get("terminate"))
        if result["terminate"] and not str(result.get("next_step_advice") or "").strip():
            result["next_step_advice"] = "无需下一步优化。"
        return result


# Utility functions for rule management
def load_rule_knowledge_base() -> list:
    """Load the flat rule list from Rule_Examples.json."""
    try:
        rule_file = "/root/AgentRewrite/src/Rewrite_Middleware/Structured_Knowledge_Base/preparation/data/Rule_Examples.json"
        with open(rule_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Error loading rule knowledge base: {e}")
        return []


def get_all_rules() -> dict:
    """Return {rule_id: description} for every rule, with no category grouping."""
    result = {}
    for rule in load_rule_knowledge_base():
        if not isinstance(rule, dict) or not rule.get("id"):
            continue
        result[str(rule["id"])] = str(rule.get("description") or "")
    return result


def get_rule_examples(rule_ids: list, current_sql: Optional[str] = None) -> dict:
    """Return each selected rule's description and all of its examples."""
    del current_sql
    by_id = {
        str(rule.get("id")): rule
        for rule in load_rule_knowledge_base()
        if isinstance(rule, dict) and rule.get("id")
    }
    result: Dict[str, dict] = {}
    for rule_id in rule_ids:
        rule = by_id.get(str(rule_id))
        if not rule:
            continue
        examples = [
            dict(example)
            for example in (rule.get("examples") or [])
            if isinstance(example, dict)
        ]
        result[str(rule_id)] = {
            "description": str(rule.get("description") or ""),
            "examples": examples,
        }
    return result


def format_rule_examples_for_semantic_check(ex_map: Dict[str, dict], rule_ids: Optional[list] = None) -> str:
    """按给定顺序输出每条规则的 ID、规则描述和全部示例。"""
    ordered = [str(rule_id) for rule_id in rule_ids] if rule_ids else list((ex_map or {}).keys())
    if not ordered:
        return "（未选择规则。）"
    parts: List[str] = []
    for rid in ordered:
        block = (ex_map or {}).get(rid)
        if not isinstance(block, dict):
            parts.append(f"规则ID: {rid}")
            parts.append("规则描述: （规则描述缺失）")
            parts.append("示例: 无")
            parts.append("")
            continue
        parts.append(f"规则ID: {rid}")
        parts.append(f"规则描述: {block.get('description', '')}")
        examples = block.get("examples") or []
        if not examples:
            parts.append("示例: 无")
        for i, ex in enumerate(examples, 1):
            if not isinstance(ex, dict):
                continue
            parts.append(f"示例 {i}:")
            example_desc = ex.get("example_description", "")
            if example_desc:
                parts.append(f"  示例描述: {example_desc}")
            parts.append(f"  原始查询: {ex.get('original_query', '')}")
            parts.append(f"  重写查询: {ex.get('rewritten_query', '')}")
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
        self.watch(["EvaluationAgent", "ReasoningAgent"])

    async def rewrite_with_rule_sequence(
        self,
        sql: str,
        rule_sequence: dict,
        schema_content: Optional[str] = None,
        previous_feedback: Optional[dict] = None,
    ) -> dict:
        """Execute SQL rewriting based on selected rule sequence with semantic equivalence check"""
        applied_rules = rule_sequence.get("applied_rules", [])

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
        previous_feedback_section = ""
        if previous_feedback:
            previous_feedback_section = f"""
        <上一轮评估反馈（重点修复）>
        {json.dumps(previous_feedback, ensure_ascii=False, indent=2)}
        </上一轮评估反馈（重点修复）>
"""
        rule_examples = get_rule_examples(applied_rules, current_sql=sql)
        rule_examples_text = format_rule_examples_for_semantic_check(rule_examples, applied_rules)
        rule_examples_section = f"""
        <已选规则（规则ID、规则描述、全部示例）>
        {rule_examples_text}
        </已选规则（规则ID、规则描述、全部示例）>
"""

        system_message = textwrap.dedent("""
        <Mission>
        你是一名经验丰富的 DBA，你的任务是按照指定的规则序列对SQL进行重写。

        1. 输入信息：
           - <base_sql>: 当前轮次的基底 SQL。第一轮它等于原始SQL；若进入下一轮，它就是上一轮 rewritten_sql
           - <rule_sequence>: 要应用的规则序列
           - （若 user 中提供）<SQL Schema>: 与当前查询相关的表 DDL
           - <已选规则>: 按应用顺序列出每条规则的规则ID、规则描述和全部示例。规则描述说明规则本身；示例描述只说明这一条示例。示例只能作为模式参考，不能照抄其中的表名、列名或谓词
           - （若 user 中提供）<上一轮评估反馈（重点修复）>: 未改进/恶化原因与下一步改进建议，必须优先处理

        2. 重写要求：
           - **最高优先级约束**：必须优先保证 step_sqls 每一项尤其是最后一项与原始SQL在语义上完全等价；任何优化都不得以改变结果集语义为代价
           - 按 <已选规则> 中的规则ID、规则描述和示例改写。一条规则有多个示例时，按与当前 SQL 结构对应的那条来改写，不要把同一改法的示例混用
           - 仍只输出一次 JSON。按 applied_rules 的顺序逐条应用：step_sqls[i] 是在 base_sql 上只应用了前 i+1 条规则后的完整 SQL
           - step_sqls 与 applied_rules 等长、同序。每一步只新增该位置的那一条规则，不要在同一步提前应用后面的规则
           - step_sqls 的最后一项就是最终 SQL。无规则时 step_sqls 为 []
           - 确保SQL的语法正确性
           - **必须保持查询语义的等价性**：重写后的SQL必须与原始SQL在语义上完全等价
           - 若提供了上一轮评估反馈，说明这是增量改写场景；必须在当前基底 SQL 的基础上继续优化，而不是回退并从原始 SQL 重新开始
           - 规则描述写明不适用的情形就不要套用。下推到子查询时只投影后续需要的列。同一段聚合或过滤被多处使用时不要复制一份来绕过共享
           - 生成可执行的SQL

        3. 语义等价性检查：
           - 重写后的SQL必须返回与原始SQL相同的结果集
           - 列名、数据类型、排序顺序等必须保持一致
           - 聚合函数、窗口函数、子查询等必须保持相同的语义
        4. **semantic_correctness_guarantee（语义正确性保证说明，必填）**：
           用**中文**分条写清（建议 3–5 条，须可被审计员**逐条**对照原 SQL 与重写 SQL 独立核验），至少包含：
           - 相对原 SQL 做了哪些**结构性**改写（下推/合并/去相关/CTE 展开等），对应到哪些谓词、JOIN 或子查询作用域。
           - **为何**在 Schema 与 TPC-DS/常规 SQL 语义下，这些改写仍与原始查询**结果集**等价；涉及 GROUP BY/主键/相关子查询时，写出键与相关列。
           - 若有易误判点（如过滤提前、HAVING/ORDER BY 与 LIMIT、NULL、重复行、相关子查询相关列），**明确**你判定仍等价的理由。
           不得写与代码不一致的空话；不得省略「保证说明」字段。

        5. **只输出一个 JSON 对象**（不要 <rewrite> 标签、不要 markdown）。字段：
           - applied_rules: 数组，与 user 中给出的本次序列一致
           - step_sqls: 字符串数组，与 applied_rules 等长。step_sqls[i] 为只应用到第 i+1 条规则时的完整 SQL；最后一项就是最终 SQL。不要输出 rewritten_sql
           - semantic_correctness_guarantee: 上述第 4 点「语义正确性保证说明」全文（多行可放在同一字符串，用 \\n 转义换行）
        """).strip()

        prompt = textwrap.dedent(f"""
        本次 applied_rules 必须原样返回: {json.dumps(applied_rules, ensure_ascii=False)}

        <base_sql>
        {sql}

        <rule_sequence>
        {json.dumps(rule_sequence, ensure_ascii=False, indent=2)}
        """).strip()
        prompt = f"{prompt}\n{schema_section}{rule_examples_section}{previous_feedback_section}"

        thought_chain = await self.llm.get_LLM_response_async(
            prompt=prompt,
            system_message=system_message,
            json_format=True,
            json_schema=STRICT_JSON_SCHEMAS["rewrite"],
        )

        try:
            parsed, _ = parse_llm_json(thought_chain, {})
            if parsed and _rewritten_sql_from_steps(parsed, sql):
                parsed.pop("groups", None)
                parsed.setdefault("applied_rules", applied_rules)
                _merge_rewrite_semantic_fields(parsed)
                return parsed

            rewrite_match = re.search(r"<rewrite>(.*?)</rewrite>", thought_chain, re.DOTALL)
            inner = rewrite_match.group(1).strip() if rewrite_match else thought_chain
            parsed, _ = parse_llm_json(inner, {})
            if parsed and _rewritten_sql_from_steps(parsed, sql):
                parsed.pop("groups", None)
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
                    "applied_rules": applied_rules,
                    "rewritten_sql": extracted_sql,
                    "semantic_correctness_guarantee": "",
                    "semantic_check": "",
                    "parse_error": True  # Flag to indicate parsing error
                }

            return {
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
                    "applied_rules": applied_rules,
                    "rewritten_sql": extracted_sql,
                    "semantic_correctness_guarantee": "",
                    "semantic_check": "",
                    "parse_error": True,
                    "error_info": f"JSON解析错误: {str(e)}"
                }
            
            return {
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
        """对照两条 SQL 找出可能不等价处，在不改变已选规则应用的前提下做最小修复。"""
        diff_text = json.dumps(differences, ensure_ascii=False, indent=2) if differences else "[]"
        rules_text = json.dumps(applied_rules, ensure_ascii=False) if applied_rules else "[]"
        system_message = textwrap.dedent("""
        <Mission>
        你是 DBA。确定性等价检查判定当前重写 SQL 与原始 SQL 不等价。检查结果通常只有 NEQ，不会给出具体差异位置。
        你的任务是：自己对照 original_sql 与当前重写 SQL，判断哪里可能导致结果集不同，并在**不改变已选规则已经完成的改写**的前提下做**最小修改**，使两条 SQL 的结果集等价。

        修复约束（必须遵守）：
        0. **最高优先级约束**：必须优先保证修复后 SQL 与 original_sql 语义等价；禁止为了“看起来更优”而改变结果集语义。
        1. differences 里若只有 NEQ、为空，或没有 SQL 内部位置，自己检查谓词范围、JOIN 条件、聚合与 GROUP BY、DISTINCT、NULL、重复行、相关列和 ORDER BY/LIMIT，找出可能导致结果集不同的片段。
        2. 只改你判定会导致结果集变化的片段。不要重排无关的 JOIN/CTE，不要整体重写，做最小结构修改。
        3. 不应用任何新的优化规则，也不要新增、撤销、重排或替换已选规则已经完成的改写。
        4. 能修复时，输出必须是**可执行完整 SQL**，且与原始 SQL 结果集等价。

        输出要求：**只输出一个 JSON 对象**（不要 markdown），格式：
        {"fixed_sql": "修复后的完整 SQL 单行或合理换行", "note": "一句话说明你判定的不等价点和如何修复"}
        """).strip()
        prompt = textwrap.dedent(f"""
        <original_sql>
        {original_sql}

        <当前重写 SQL>
        {rewritten_sql}

        <应用规则 ID>
        {rules_text}

        <语义差异点 differences>
        {diff_text}
        """).strip()
        resp = await self.llm.get_LLM_response_async(
            prompt=prompt,
            system_message=system_message,
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

        system_message = textwrap.dedent("""
        <Mission>
        你是一名经验丰富的 DBA，你的任务是基于语法错误信息修正重写后的SQL。

        **重要**：请仔细阅读并分析 <error_info> 中的错误信息，这是数据库返回的具体语法错误，必须针对这个错误进行修正。

        1. 输入信息：
           - <original_sql>: 原始SQL（用于保持语义等价性）
           - <error_info>: **当前语法检查错误信息**（这是你需要修复的具体错误）
           - <previous_rewrite>: 上次重写结果（包含之前尝试的SQL和可能的错误信息）
           - 若 user 中提供统计信息或索引信息，一并参考

        2. 修正要求：
           - **最高优先级约束**：必须优先保证 rewritten_sql 与 original_sql 的语义等价；语法修复不得引入语义漂移
           - **仔细分析 <error_info> 中的错误信息**，这是数据库返回的具体语法错误位置和原因
           - 如果 <previous_rewrite> 中包含之前的错误信息，参考它们避免重复同样的错误
           - 仅做“最小修复”：只修改触发语法错误的必要片段（如列引用、别名、GROUP BY项、函数参数）
           - 不应用任何优化规则，也不要新增、撤销、重排或替换已选规则已经完成的改写
           - 禁止无关改动：不要新增/删除非必要子查询，不要整体改写
           - 如果必须改动规则应用才能通过语法检查，返回 {"rewritten_sql":"","note":"无法在保留规则应用的前提下修复"}
           - 能修复时，确保修正后的SQL能够通过语法检查
           - **必须保持查询语义正确性**：修正后的SQL必须与原始SQL在语义上完全等价
           - 生成可执行的、语法正确的SQL

        3. 输出要求：
           - **只输出一个 JSON 对象**（不要 markdown / 不要额外文本）
           - 必须包含字段 rewritten_sql，值为修正后的完整可执行 SQL
           - 可选字段 note 用一句话说明修复点
           - 示例：{"rewritten_sql":"SELECT ...","note":"修复了别名不存在错误"}
        """).strip()
        prompt = textwrap.dedent(f"""
        <original_sql>
        {sql}

        <error_info>
        {error_info}

        <previous_rewrite>
        {json.dumps(previous_rewrite, ensure_ascii=False, indent=2)}
        """).strip()
        if stats_block or idx_block:
            prompt = f"{prompt}\n{stats_block}{idx_block}"

        response = await self.llm.get_LLM_response_async(
            prompt=prompt,
            system_message=system_message,
            json_format=True,
            json_schema=STRICT_JSON_SCHEMAS["iterative_rewrite"],
        )
        data = parse_llm_json_with_default(response, {})
        if isinstance(data, dict) and "rewritten_sql" in data:
            rewritten_sql = data.get("rewritten_sql")
            if isinstance(rewritten_sql, str) and rewritten_sql.strip():
                return rewritten_sql.strip()
            return ""
        return self._extract_sql_from_response_robust(response)

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
        rule_examples = get_rule_examples(rewrite_rules or [], current_sql=original_sql)
        rule_examples_text = format_rule_examples_for_semantic_check(rule_examples, rewrite_rules or [])
        rule_examples_ctx = textwrap.dedent(f"""

            <已选规则（规则ID、规则描述、全部示例）>
            {rule_examples_text}
            </已选规则（规则ID、规则描述、全部示例）>
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
        system_message = textwrap.dedent("""
        你是 SQL 语义审计专家，你要分析判断原始SQL和重写SQL是否等价。等价含义：**同一库状态、同一参数下，两查询结果集相同（行集合与列语义一致；允许列名/顺序在逻辑上等价时的合理差异，但若会改变行数或聚合语义则不等价）**。

        输入优先级：
        1. **原始 SQL** 与 **当前重写 SQL** 的实际语义（谓词、JOIN、GROUP BY、DISTINCT、子查询相关性、NULL 处理等）——**永远最高**。
        2. **SQL Schema**：主键、唯一约束、函数依赖、可据此认可的等价变形（例如已知 PK 下 GROUP BY 的化简）。
        3. **索引信息**（若提供）：通常不改变关系层面的结果集语义；PRIMARY KEY/UNIQUE 类索引可辅助推断唯一性，与 Schema 一并用于等价推理。**不得以「有无非唯一索引」代替 SQL 逻辑判断是否等价**。
        4. **首轮「语义正确性保证说明」**（若提供）：**仅**作辅助线索，须与 1 逐条核对。若存在「语义修正_第N轮_必读」块，说明**当前重写 SQL 可能已按你方上一轮意见改过**——**禁止**用首轮自辩中仅适用**历史版本**的论述来否掉**已经变化后**的当前 SQL；自辩与**当前**重写 SQL 明显不符时，**忽略**不适用的自辩段。
        5. **已选规则（规则ID、规则描述、全部示例）**：本查询**声称**依这些规则做等价改写。规则描述说明规则本身，示例描述只说明对应那一条示例。当「当前重写 SQL」可判定为**完全按**规则描述和对应示例所体现的典型变换意图忠实套用（同构的改写模式、未在规则意图之外缩小/扩大过滤范围、未擅自改变聚合/分组/去重语义），且与 **Schema/约束** 无矛盾时，应将其视为**强等价先验**：默认倾向 `equivalent=true`，除非你能给出**明确且可核验的结果集变化证据**。示例中的表名、列名和谓词只是模板，不得替代对当前两条 SQL 的实际语义判断。

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

        **只输出一个 JSON**（不要其它文字）：
        {
            "equivalent": true 或 false,
            "message": "一句话。首轮可简述自辩与事实；若存在多轮修正，请对**当前两 SQL**下结论，勿复述过时的首轮自辩。",
            "differences": [{"type": "类型", "description": "一句话", "location": "位置"}]
        }
        equivalent 为 true 时 differences 必须为 []。
        """).strip()
        prompt = textwrap.dedent(f"""
        {prior_note}

        <原始 SQL>
        {original_sql}

        <当前重写 SQL>
        {rewritten_sql}
        """).strip()
        prompt = (
            f"{prompt}\n{retry_ctx}{schema_ctx}{index_ctx}\n{sem_ctx}\n"
            f"{rule_examples_ctx}"
        )
        response = await self.llm.get_LLM_response_async(
            prompt=prompt,
            system_message=system_message,
            json_format=True,
            json_schema=STRICT_JSON_SCHEMAS["semantic_equivalence"],
        )
        return self._parse_equivalence_response(response)

