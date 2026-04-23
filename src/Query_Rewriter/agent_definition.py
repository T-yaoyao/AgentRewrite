
import json
import os
import re
import textwrap
import sys
from typing import Any, Dict, Optional

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
            },
            "required": ["terminate", "reason"],
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
            },
            "required": ["groups", "applied_rules", "rewritten_sql"],
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
        """
        Keep only positive numeric scores for selected rules and normalize to sum=1.
        """
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
        """
        Keep confidence values in [0, 1] for selected rules only.
        """
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
        """Select appropriate rule sequence based on DecisionAgent's advice with UCT scores"""
        advice_text = json.dumps(decision_advice, ensure_ascii=False, indent=2)

        # Build rule library text with UCT scores
        rule_text = ""
        all_groups = []
        all_rules = []
        
        # Check if rule_library contains UCT scores (tuple format) or plain descriptions
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
            
            # Sort rules by UCT score if available (descending)
            sorted_rules = []
            for rule_id, rule_data in rules.items():
                if has_uct_scores and isinstance(rule_data, tuple):
                    desc, score = rule_data
                    sorted_rules.append((rule_id, desc, score))
                else:
                    sorted_rules.append((rule_id, rule_data, 1.0))  # Default score 1.0
            
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
        你是一名经验丰富的 DBA，你的任务是基于决策Agent提供的优化方向，从规则库中挑选合适的优化规则序列。你只负责选择规则，不进行任何重写操作。

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

        <sql语句>
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
        - groups: 字符串，必须与「当前优化方向」一致："{groups_text}"
        - applied_rules: 字符串数组，按应用顺序列出规则 ID；必须来自 <rule_library>，禁止编造；若无适用规则则为 []
        - rule_effect_scores: 对象，键为 applied_rules 中的 rule_id，值为该规则预估贡献分（正数）。建议总和约为 1；若无规则则 {{}}
        - rule_effect_confidence: 对象，键为 applied_rules 中的 rule_id，值为该规则效果估计的置信度（0到1之间）

        示例：{{"groups": "{groups_text}", "applied_rules": ["RULE_ID_1"], "rule_effect_scores": {{"RULE_ID_1": 1.0}}, "rule_effect_confidence": {{"RULE_ID_1": 0.8}}}}
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
            parsed["rule_effect_scores"] = self._sanitize_rule_effect_scores(
                parsed.get("applied_rules", []),
                parsed.get("rule_effect_scores", {}),
            )
            parsed["rule_effect_confidence"] = self._sanitize_rule_effect_confidence(
                parsed.get("applied_rules", []),
                parsed.get("rule_effect_confidence", {}),
            )
            return parsed
        sequence_match = re.search(r"<rule_sequence>(.*?)</rule_sequence>", thought_chain, re.DOTALL)
        inner = sequence_match.group(1).strip() if sequence_match else thought_chain
        parsed2, _ = parse_llm_json(inner, default)
        if parsed2 and isinstance(parsed2.get("applied_rules"), list):
            parsed2["chosen_rule_prior"] = []
            parsed2["rule_effect_scores"] = self._sanitize_rule_effect_scores(
                parsed2.get("applied_rules", []),
                parsed2.get("rule_effect_scores", {}),
            )
            parsed2["rule_effect_confidence"] = self._sanitize_rule_effect_confidence(
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
    base_url=os.getenv("DECISION_MODEL_URL")
))
        self.watch(["ReasoningAgent", "ExplainAgent"])

    async def initial_optimization_check(self, sql: str, data_statistics: str, explain_info: str, few_shot_examples: list = None, index_info: str = "") -> dict:
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
        
        prompt = textwrap.dedent(f"""
        <Mission>
        你是一名经验丰富的 SQL查询重写优化专家，你的任务是判断给定的 SQL 查询是否可以通过查询重写优化来提升执行性能。

        1. 分析输入信息：
           - <sql语句>: 原始SQL查询
           - <统计信息>: 数据库表统计信息
           - <索引信息>: 表索引（索引名与定义）
           - <执行计划分析结果>: 智能执行计划分析器生成的分析结果
           {few_shot_context}

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

        prompt = textwrap.dedent(f"""
        你负责评估 SQL 优化是否达标，并判断是否终止优化流程。

        基于 `original_sql` 与 `rewritten_sql/enhanced_sql` 的 EXPLAIN、cost、执行时间等信息进行综合评估，重点对比：
        1) 执行计划变化（扫描方式、连接顺序/算法、过滤下推、排序聚合、临时表/回表等）；
        2) 成本差异（注意 cost 可能不精确，仅作参考）；
        3) 重写 SQL 改进点（执行计划层面 + SQL 逻辑层面）；
        4) 执行时间表现及是否存在基数估计误差。

        注意：`original_sql` 和 `enhanced_sql` 的 costs 来自优化器，可能不够精确；必须基于详细分析客观判断 rewritten_sql 是否满足成功重写关键指标。

        终止条件：
        - `terminate = true`：
          a. `rewritten_sql` 的 costs < `ori_sql` 的 costs；或
          b. `rewritten_sql` 执行时间 >= `ori_sql`，但属于基数估计误差导致，且重写本质仍有改进。
        - `terminate = false`：
          `rewritten_sql` 执行时间 >= `ori_sql`，且属于明显性能恶化。

        严格仅输出以下 JSON（不得有任何额外文本）：
        {{
          "terminate": true/false,
          "reason": "简要说明依据：执行计划变化、成本差异、改进点、执行时间对比、是否基数估计误差、最终结论。"
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
        }
        result = parse_llm_json_with_default(response, fallback)
        if not isinstance(result, dict):
            return fallback
        result.setdefault("terminate", True)
        result.setdefault("reason", "")
        return result


# Utility functions for rule management
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
    group_mapping = {
        "子查询优化": "SUBQUERY_OPTIMIZATION",
        "连接优化": "JOIN_OPTIMIZATION",
        "谓词简化": "PREDICATE_OPTIMIZATION",
        "常量折叠": "CONSTANT_OPTIMIZATION",
        "聚合优化": "AGGREGATE_OPTIMIZATION",
        "投影优化": "PROJECTION_OPTIMIZATION",
        "排序优化": "SORT_OPTIMIZATION",
        "集合优化": "SET_OPERATION_OPTIMIZATION"
    }

    for group in groups:
        category = group_mapping.get(group)
        if category and category in rule_kb:
            result[group] = rule_kb[category].get("rules", {})

    return result

RULE_EXAMPLES_MAX_PER_RULE = 2


def get_rule_examples(rule_ids: list, max_per_rule: int = RULE_EXAMPLES_MAX_PER_RULE) -> dict:
    """按规则 ID 拉取示例；每个规则最多 max_per_rule 条（默认 TOP2），减少重写提示长度。"""
    rule_kb = load_rule_knowledge_base()
    result: Dict[str, list] = {}
    want = set(rule_ids)

    for category_data in rule_kb.values():
        for example in category_data.get("examples", []):
            rule_id = example.get("id")
            if rule_id not in want:
                continue
            bucket = result.setdefault(rule_id, [])
            if len(bucket) >= max_per_rule:
                continue
            bucket.append(example)

    return result


class RewriteAgent(Agent):
    """SQL Rewrite Execution Agent"""
    def __init__(self, mq: MessageQueue):
        super().__init__("RewriteAgent", mq, gpt = GPT(
            api_key=os.getenv("REWRITE_MODEL_API_KEY"),
            model=os.getenv("REWRITE_MODEL"),
            base_url=os.getenv("REWRITE_MODEL_URL")
        ))
        self.watch(["DecisionAgent", "ReasoningAgent"])

    async def rewrite_with_rule_sequence(
        self,
        sql: str,
        rule_sequence: dict,
        rule_examples: dict,
        optimization_direction: str,
        data_statistics: str,
        schema_content: Optional[str] = None,
        index_info: str = "",
    ) -> dict:
        """Execute SQL rewriting based on selected rule sequence with semantic equivalence check"""
        applied_rules = rule_sequence.get("applied_rules", [])
        groups = rule_sequence.get("groups", "")

        # Build rule examples text
        examples_text = ""
        for rule_id in applied_rules:
            if rule_id in rule_examples:
                examples_text += f"### {rule_id}\n"
                for example in rule_examples[rule_id]:
                    examples_text += f"原始查询: {example.get('original_query', '')}\n"
                    examples_text += f"重写查询: {example.get('rewritten_query', '')}\n"
                    examples_text += f"规则描述: {example.get('rule_description', '')}\n\n"

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

        prompt = textwrap.dedent(f"""
        <Mission>
        你是一名经验丰富的 DBA，你的任务是按照指定的规则序列对SQL进行重写。

        1. 输入信息：
           - <original_sql>: 原始SQL查询
           - <rule_sequence>: 要应用的规则序列
           - <rule_examples>: 相关规则的重写示例
           - <optimization_direction>: 优化方向
           - <统计信息>: 数据库表统计信息
           - （若下方提供）<SQL Schema>: 与当前查询相关的表 DDL
           - （若下方提供）<索引信息>: 表索引信息

        2. 重写要求：
           - 严格按照applied_rules中的规则顺序依次应用
           - 每个规则应用后都要确保SQL的正确性
           - 参考rule_examples中的示例进行重写
           - **必须保持查询语义的等价性**：重写后的SQL必须与原始SQL在语义上完全等价
           - 生成可执行的SQL

        3. 语义等价性检查：
           - 重写后的SQL必须返回与原始SQL相同的结果集
           - 列名、数据类型、排序顺序等必须保持一致
           - 聚合函数、窗口函数、子查询等必须保持相同的语义

        <original_sql>
        {sql}

        <rule_sequence>
        {json.dumps(rule_sequence, ensure_ascii=False, indent=2)}

        <rule_examples>
        {examples_text}

        <optimization_direction>
        {optimization_direction}

        <统计信息>
        {data_statistics}
{schema_section}{idx_section}
        4. **只输出一个 JSON 对象**（不要 <rewrite> 标签、不要 markdown）。字段：
           - groups: 字符串 "{groups}"
           - applied_rules: 数组，与当前序列一致：{json.dumps(applied_rules, ensure_ascii=False)}
           - rewritten_sql: 重写后的完整可执行 SQL（字符串内换行用 \\n）
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
                parsed.setdefault("semantic_check", "")
                return parsed

            rewrite_match = re.search(r"<rewrite>(.*?)</rewrite>", thought_chain, re.DOTALL)
            inner = rewrite_match.group(1).strip() if rewrite_match else thought_chain
            parsed, _ = parse_llm_json(inner, {})
            if parsed and parsed.get("rewritten_sql"):
                parsed.setdefault("groups", groups)
                parsed.setdefault("applied_rules", applied_rules)
                parsed.setdefault("semantic_check", "")
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
                    "parse_error": True  # Flag to indicate parsing error
                }

            return {
                "groups": groups,
                "applied_rules": applied_rules,
                "rewritten_sql": sql,  # fallback to original
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
                    "parse_error": True,
                    "error_info": f"JSON解析错误: {str(e)}"
                }
            
            return {
                "groups": groups,
                "applied_rules": applied_rules,
                "rewritten_sql": sql,
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
            gpt=GPT(api_key=api_key, model=model, base_url=base_url),
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
        schema_content: Optional[str] = None,
        index_info: Optional[str] = None,
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
        sem_text = ""
        if semantic_check is not None:
            s = str(semantic_check).strip()
            if s:
                sem_text = s
        # 仅承载重写代理原文，Instructions 统一写在下方唯一的主 prompt 里，避免「两套说明」重复。
        sem_ctx = (
            textwrap.dedent(f"""

            <重写代理的 semantic_check说明>
            {sem_text}
            </重写代理的 semantic_check说明>
            """)
            if sem_text
            else "\n（未提供重写代理的 semantic_check。）\n"
        )
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
        prompt = textwrap.dedent(f"""
        你是 SQL 语义审计专家，你要分析判断原始SQL和重写SQL是否等价。等价含义：**同一库状态、同一参数下，两查询结果集相同（行集合与列语义一致；允许列名/顺序在逻辑上等价时的合理差异，但若会改变行数或聚合语义则不等价）**。

        输入优先级：
        1. **原始 SQL** 与 **当前重写 SQL** 的实际语义（谓词、JOIN、GROUP BY、DISTINCT、子查询相关性、NULL 处理等）。
        2. **SQL Schema**：主键、唯一约束、函数依赖、可据此认可的等价变形（例如已知 PK 下 GROUP BY 的化简）。
        3. **索引信息**（若提供）：通常不改变关系层面的结果集语义；PRIMARY KEY/UNIQUE 类索引可辅助推断唯一性，与 Schema 一并用于等价推理。**不得以「有无非唯一索引」代替 SQL 逻辑判断是否等价**。
        4. **重写代理的 semantic_check**（若上方 XML 块中有内容）：必须逐条对照 1–3 与两条 SQL 核验；对于semantic_check中的每条内容进行分析，只有你认为全部正确时并可推出结果集等价则倾向 equivalent=true；与事实矛盾、遗漏关键差异或过度推断则 equivalent=false，并在 differences 中写出具体差异。
        5. **应用规则**：作为改写意图参考。

        通用原则：
        - 疑罪从无：仅当**能明确指出**会导致结果集不一致的差异时判 equivalent=false。
        - Schema 中有明确定义时，采纳与约束一致的等价推理（如主键列 A 下，GROUP BY A 与 GROUP BY A,B 且 B 函数依赖于 A）。
        - 对同一张表、没有额外过滤/去重/DISTINCT 的情况下，先按任意维度 GROUP BY 再对每组 COUNT(col) 并在外层 SUM(count_col)，
  与直接对整表 COUNT(col) 结果相同。
        - 两层聚合折叠规则（强约束）：若原SQL形如「内层按 (K,U) 分组，外层按 K 分组」，且重写为「按 K 单层聚合」，并且每个外层聚合列都满足以下逐列等价之一：
          MIN(MIN(v))->MIN(v)，MAX(MAX(v))->MAX(v)，SUM(SUM(v))->SUM(v)，SUM(COUNT(v))->COUNT(v)，则**必须判 equivalent=true**。
        - 禁止误判：在满足上述强约束时，不能使用“计数对象不同”“中间分组行数不同”“内层多了 ename”作为不等价理由；这些只影响中间态，不影响最终结果。
        - 无 semantic_check 内容时，不得编造代理意图；有则以上述第 4 条为主轴审计。

        <原始 SQL>
        {original_sql}

        <当前重写 SQL>
        {rewritten_sql}
        {schema_ctx}{index_ctx}
        {sem_ctx}
        <应用规则（规则ID + 规则描述）>
        {rules_text}

        **只输出一个 JSON**（不要其它文字）：
        {{
            "equivalent": true 或 false,
            "message": "一句话（若核验了 semantic_check说法是正确的，则需体现是否采纳其主张）",
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

