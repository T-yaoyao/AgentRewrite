
from openai import OpenAI, AsyncOpenAI
import json
import os
import re
import textwrap
import sys
import json
import textwrap
from typing import List, Dict, Set, Optional, Union

# Setup project paths
from src.utils.path_config import setup_python_path, load_project_env
setup_python_path()
load_project_env()

from src.utils.agent_template import MessageContent, Message, MemoryWindow, MessageQueue,  Agent
from src.utils.llm_client import GPT

class ReasoningAgent(Agent):
    """MDP-based Reasoning Agent"""
    def __init__(self, mq: MessageQueue):
        super().__init__("ReasoningAgent", mq, gpt = GPT(
        api_key=os.getenv("REASONING_MODEL_API_KEY"),
        model=os.getenv("REASONING_MODEL"),
        base_url=os.getenv("REASONING_MODEL_URL")
        ))

        self.api_key = os.getenv("REASONING_MODEL_API_KEY")  # Get the API key from the environment variable
        self.model = os.getenv("REASONING_MODEL")
        self.base_url = os.getenv("REASONING_MODEL_URL")
        self.async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )


    async def get_answer(self, prompt):
        reasoning_content = ""  # 
        is_answering = False   # 
        messages= []

        # messages.append({"role": "system", "content": "Initiate your response with '<think>\\n' at the beginning of every output."})
        messages.append(({"role": "user", "content": prompt}))

        completion = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0.0
        )

        async for chunk in completion:
            if not chunk.choices:
                if hasattr(chunk, 'usage') and chunk.usage:
                    print("\nUsage:")
                    print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    print(delta.reasoning_content, end='', flush=True)
                    reasoning_content += delta.reasoning_content
                else:
                    if delta.content != "" and is_answering is False:
                        is_answering = True
                        break

        return reasoning_content
        
        
    async def analyze_sql(self, sql: str, data_statistics, explain_info : str) -> dict:

        prompt = textwrap.dedent(f"""
        <任务目标>
       你是一名经验丰富的 DBA，核心任务是为用户完成高质量的 SQL 查询重写工作。                    
        1.基于统计信息和执行计划分析结果，对 SQL 进行性能诊断，并给出有针对性的重写优化方向。
        针对每一项优化策略，需评定重写评分，同时确定该 SQL 的最优优化方案。
        注意：当查询语句包含复杂的WHERE/JOIN条件，或存在重复子查询计算时，可考虑使用公共表表达式（CTE） ——CTE 能够提升查询的可读性与执行性能。若查询语句本身结构简单，或使用 CTE 无法减少冗余计算，则应避免过度使用 CTE，多余的 CTE 可能会增加系统开销。
        2. 如果必要，重写 SQL 查询，确保满足以下标准：
            - 可执行性
            - 等价性
            - 效率（执行效率与计算效率）
            - 可读性

        重要提示若经评估，目标 SQL 查询语句已处于充分优化状态，或结构过于简单无需调整，则直接给出该 SQL 的最终版本，同时判定其 “无需进一步优化”，并在回复末尾标注关键词 TERMINATE。                      
        <sql语句>
        {sql}

        <统计信息>
        {data_statistics}

        <执行计划分析结果>
        {explain_info}
        """)
        
        thought_chain = await self.get_answer(
            prompt=prompt,
        )
        self.send_message(
            MessageContent(text=thought_chain),
            role="ReasoiningAgent",
            receiver="SummaryAgent"
        )

        return thought_chain

    

    async def analyze_sql_report(self, sql: str, data_statistics, report: dict, explain_info: str) -> dict: 

        prompt = textwrap.dedent(f"""
        <Mission>
        你是一名经验丰富的 DBA，核心任务是为用户完成高质量的 SQL 查询重写工作。                    
        1.基于统计信息和执行计划分析结果，对 SQL 进行性能诊断，并给出有针对性的重写优化方向。
        针对每一项优化策略，需评定重写评分，同时确定该 SQL 的最优优化方案。
        注意：当查询语句包含复杂的WHERE/JOIN条件，或存在重复子查询计算时，可考虑使用公共表表达式（CTE） ——CTE 能够提升查询的可读性与执行性能。若查询语句本身结构简单，或使用 CTE 无法减少冗余计算，则应避免过度使用 CTE，多余的 CTE 可能会增加系统开销。
        2. 如果必要，重写 SQL 查询，确保满足以下标准：
            - 可执行性
            - 等价性
            - 效率（执行效率与计算效率）
            - 可读性


        3. 特别说明：你会获取到此前的重写报告作为参考。
                                 
        若经评估，目标 SQL 查询语句已处于充分优化状态，或结构过于简单无需调整，则直接判定其 “无需进一步优化”，并在回复末尾标注关键词 TERMINATE。

        <sql语句>
        {sql}

        <统计信息>
        {data_statistics}

        <执行计划分析结果>
        {explain_info}

        <重写报告>
        {report}

        """)
        
        thought_chain = await self.get_answer(
            prompt=prompt,
        )
        self.send_message(
            MessageContent(text=thought_chain),
            role="ReasoiningAgent",
            receiver="SummaryAgent"
        )

        return thought_chain

    
class DecisionAgent(Agent):
    """Summarize Optimization Suggestions Agent"""
    def __init__(self, mq: MessageQueue):
        super().__init__("DecisionAgent", mq, gpt = GPT(
    api_key=os.getenv("DECISION_MODEL_API_KEY"),
    model=os.getenv("DECISION_MODEL"),
    base_url=os.getenv("DECISION_MODEL_URL")
))
        self.watch(["ReasoningAgent", "ExplainAgent"])
    

    async def summarize_chain(self, chain: str, original_sql: str, data_statistics) -> dict:
        prompt = textwrap.dedent(f"""
        <Mission>
        你是一名经验丰富的 DBA，核心任务是为用户完成高质量的 SQL 查询重写工作。
        1. 总结重写建议和新的重写 SQL 查询，基于详细的 SQL 重写报告 [chain]。
        每个建议应归入以下类别之一：
        - 谓词简化
        - 子查询优化
        - 查询优化
        - 连接优化
        - 常量折叠
       2. 随后检查 produced_sql，挖掘其潜在的优化空间，在保证结果等价性的前提下对其进行进一步优化（例如：将过滤条件提前至FROM子句的JOIN环节、在主查询中保留过滤条件以优化 CTE 结构、常量折叠、日期 / 数值计算优化、冗余谓词简化等）。
        此阶段还可结合数据统计信息（data_statistics） 提升 SQL 执行效率，尤其是在考虑创建 CTE 的场景下。
        注意：切勿盲目创建 CTE，尤其是当查询包含大量WHERE/JOIN条件，或冗余子查询计算较少时，更应避免滥用 CTE。                    

        3. 请严格遵循以下格式：
        [format]
        </produced_sql>
        ```sql
                                 
        ```
        </produced_sql>
        
        </advice>
        [
                {{
                    "group": "",
                    "produced_suggestion": ""
                }},
                ... // 如果还有更多建议，则继续添加
        ]
        </advice>
                                 
        </analysis>
        // 重写 SQL 语句的分析
        </analysis>
        
        </enhanced_sql>
        ```sql
                                 
        ```
        </enhanced_sql>
                                 

        [chain]
            {chain} 

        [oringal_sql]
            {original_sql}
        """)
        
        response = await self.llm.get_LLM_response_async(
            prompt=prompt,
            json_format= False
        )
        return response 
    

    async def check_equivalence(self, ori_sql: str, rewritten_sql: str, rewrite_advice) -> dict:
        prompt = textwrap.dedent(f"""
        你是一名经验丰富的 DBA，核心任务是检查原始 SQL 和改进 SQL 的等价性。
        1. 检查原始 SQL 和改进 SQL 的等价性。如果认为改进 SQL 不等价于原始 SQL，请提供修正后的 SQL。
        2. 你还有重写思路过程需要考虑，以使 SQL 更高效。注意：不要在 SQL 中包含注释，并尽量避免进行过多的交换。

        <original_sql>:
        {ori_sql}

        <rewritten_sql>:
        {rewritten_sql}

        <rewritten_idea_process>:
        {rewrite_advice}
        3. 请严格遵循以下格式：
        [format]
        </analysis>

        </analysis>
        
        </equivalence>
            // True/False
        </equivalence>
        
        </corrected_sql>
          // 如果是false，则插入修正后的 SQL；否则留空。
        </corrected_sql>
        """)
        
        response = await self.llm.get_LLM_response_async(
            prompt=prompt,
            json_format=False
        )
        return response
    
    async def select_sql(self, original_sql:str, query_pairs:list) -> dict:
        prompt = textwrap.dedent(f"""
        你是一名经验丰富的 DBA，核心任务是选择最有效的改进 SQL 语句。
        1. 你已获得多个等价于原始 SQL 查询的 SQL 语句，每个改进 SQL 语句都有其自己的重写过程。
        2. 你的任务是选择最有效的改进 SQL 语句。
        3. 请严格遵循以下格式：
        [format]
        </analysis>
            // 填充选定 SQL 语句的分析。
        </analysis>
        </selected_id>
            // 填充你认为最好的选定 ID。
        </selected_id>
        """)
        
        response = await self.llm.get_LLM_response_async(
            prompt=prompt,
            json_format=False
        )
        return response

    
    async def evaluate(self, ori_sql: str, enhanced_sql: str, report: str, worker_equivalence_flags: List[bool] = None) -> dict:
        # Check if all workers failed equivalence check
        equivalence_failure_info = ""
        if worker_equivalence_flags is not None and all(not flag for flag in worker_equivalence_flags):
            equivalence_failure_info = """
        
        **IMPORTANT CONTEXT**: 
        所有并行工作器都失败了等价性检查，不得不回退到原始 SQL。 
        这意味着重写后的查询不等价于原始查询，而不是原始查询已经是最优的。
        系统尝试了 SQL 优化，但重写版本失败了等价性验证。
        考虑这种情况作为失败优化尝试，而不是表明不需要优化。
        """

        prompt = textwrap.dedent(f"""
        你负责评估 SQL 优化是否符合标准。请根据以下信息决定是否终止优化过程：
        你想终止优化过程吗？

        * 注意：<original_sql> 和 <enhanced_sql> 的执行时间来自数据库优化器，可能不精确。基于详细分析做出决定。
        注意：<original_sql> 和 <enhanced_sql> 的执行时间来自数据库优化器，可能不精确。基于详细分析做出决定。
        * 客观评估改进 SQL 是否满足成功重写的指标。

        终止条件：
        [True]:
            1. enhanced_sql 执行时间 < ori_sql 执行时间 and enhanced_sql执行没有报错.
            2. enhanced_sql 执行时间 ≥ ori_sql 执行时间，由于基数估计不准确，但你仍可将此次重写视为一种优化。

        [False]:
            enhanced_sql 执行时间 ≥ ori_sql 执行时间，或 enhanced_sql 执行失败。
            {equivalence_failure_info}
            
        请严格遵循以下 JSON 格式返回你的答案：
        {{
            "terminate": True/False,
            "reason": ""  // 提供你的理由。
        }}

        <original_sql>:
        {ori_sql}

        <enhanced_sql>:
        {enhanced_sql}

        <report>:
        {report}
        """)
        
        response = await self.llm.get_LLM_response_async(
            prompt=prompt,
            json_format=False
        )
        return response

    
    def extract_advice_content(self, text: str) -> str:
        """
        Extract content between </advice> and </advice> tags.
        """
        pattern = r'</advice>\s*(.*?)\s*</advice>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())
        return ""
    
    def extract_selected_id_content(self, text: str) -> str:
        """
        Extract content between </selected_id> and </selected_id> tags.
        """
        pattern = r'</selected_id>\s*(.*?)\s*</selected_id>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()  # 直接返回字符串，不需要JSON解析
        return ""
    

    def extract_sql_candidate_content(self, text: str) -> str:
        """
        Extract content between </sql_candidate> and </sql_candidate> tags.
        """
        # First, try to match the format with ```sql
        pattern1 = r'</sql_candidate>\s*```sql\s*(.*?)\s*```\s*</sql_candidate>'
        match1 = re.search(pattern1, text, re.DOTALL)
        if match1:
            return match1.group(1).strip()

        # Try to match the format without ```
        pattern2 = r'</sql_candidate>\s*(.*?)\s*</sql_candidate>'
        match2 = re.search(pattern2, text, re.DOTALL)
        if match2:
            sql_content = match2.group(1).strip()
            sql_content = re.sub(r'^```sql\s*', '', sql_content)
            sql_content = re.sub(r'\s*```$', '', sql_content)
            return sql_content.strip()
        
        print(f"Warning: Could not extract SQL candidate from response")
        return ""

    def extract_enhanced_sql_content(self, text: str) -> str:
        """
        Extract content between </enhanced_sql> and </enhanced_sql> tags.
        """
        # First, try to match the format with ```sql
        pattern1 = r'</enhanced_sql>\s*```sql\s*(.*?)\s*```\s*</enhanced_sql>'
        match1 = re.search(pattern1, text, re.DOTALL)
        if match1:
            return match1.group(1).strip()

        # Try to match the format without ```
        pattern2 = r'</enhanced_sql>\s*(.*?)\s*</enhanced_sql>'
        match2 = re.search(pattern2, text, re.DOTALL)
        if match2:
            sql_content = match2.group(1).strip()
            sql_content = re.sub(r'^```sql\s*', '', sql_content)
            sql_content = re.sub(r'\s*```$', '', sql_content)
            return sql_content.strip()
        
        print(f"Warning: Could not extract enhanced SQL from response")
        return ""
    
    def extract_equivalence_content(self, text: str) -> str:
        """
        Extract content between </equivalence> and </equivalence> tags.
        """
        pattern = r'</equivalence>\s*(.*?)\s*</equivalence>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def extract_corrected_sql_content(self, text: str) -> str:
        """
        Extract content between </corrected_sql> and </corrected_sql> tags.
        Support multiple formats: code blocks with ```sql and plain text without.
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

    async def merge_advice(self, base_sql: str, optimizations: List[str], rag_optimizations: List[str]) -> str:
        """Merge optimization suggestions into SQL"""
        optimizations_str = [
            json.dumps(opt) if isinstance(opt, dict) else str(opt)
            for opt in optimizations
        ]
        prompt = textwrap.dedent(f"""
        你是一名经验丰富的 DBA，核心任务是合并专家知识中的 RAG 优化建议和原始优化建议，并返回最终的重写建议。
        仔细考虑原始优化建议和 RAG 优化建议的有效性，并返回以下格式的 JSON：
        审慎评估原始优化建议与检索增强生成（RAG）优化方案的有效性，并按以下格式返回结果。
 
        原始 SQL 语句：
        {base_sql}

        原始优化建议：
        {optimizations_str}

        RAG 优化建议：
        {rag_optimizations}

        请严格遵循以下 JSON 格式返回你的答案：
        {{
            {{
                "group": "",
                "produced_suggestion": ""  // 合并建议
            }}
        }}
        """)
        response = await self.llm.get_LLM_response_async(
            prompt=prompt,
            json_format=False
        )
        # return self._extract_sql(response)
        return response
    
    def retrieve_reasoning_chain(self) -> str:
        """Retrieve reasoning chain from message queue"""
        messages = self.retrieve_memories(k=1)
        return [msg.content.text for msg in messages if msg.content and msg.content.text]


class AssistantAgent(Agent):
    """Execution Plan Analysis Agent"""
    def __init__(self, mq: MessageQueue):
        super().__init__("AssistantAgent", mq, gpt = GPT(
    api_key=os.getenv("ASSISTANT_MODEL_API_KEY"),
    model=os.getenv("ASSISTANT_MODEL"),
    base_url=os.getenv("ASSISTANT_MODEL_URL")
))
    
    def extract_corrected_sql_content(self, text: str) -> str:
        """
        Extract content between </corrected_sql>```sql and ```</corrected_sql> tags.
        """
        pattern = r'</corrected_sql>\s*```sql\s*(.*?)\s*```\s*</corrected_sql>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    async def _correct_sql(self, original_sql: str, rewritten_sql: str, error: str) -> str:
        """Correct SQL syntax errors"""
        prompt = textwrap.dedent(f"""
        你是一名经验丰富的 DBA，核心任务是修正 SQL 语法错误。
        1. 修正以下 SQL 语句，使用提供的错误消息。

            * 注意：如果 SQL 包含双引号，请保留它们 exactly as they appear.
            <rewritten_sql>
            {rewritten_sql}

           错误消息:
            {error}

            下面是重写 SQL 的原始形式，用于参考模式。
            只修正 <rewritten_sql> 中的语法；不要与原始 SQL 对齐，
            且不要包含任何 "EXPLAIN (FORMAT JSON)" 子句！

            <original_sql>
            {original_sql}

            请严格遵循以下格式：
            [format]
            </analysis>

            </analysis>

            </corrected_sql>
            // 插入修正后的 SQL 语句。
            </corrected_sql>
            """)

        
        response = await self.llm.get_LLM_response_async(
            prompt=prompt
        )
        
        corrected_sql = self.extract_corrected_sql_content(response)
        return corrected_sql
    
    def extract_analysis_content(self, text: str) -> str:
        """
        Extract content between </analysis> and </analysis> tags.
        """
        print(text)
        if text is None:
            print("Error: text is None")
            return None
        pattern = r'</analysis>\s*(.*?)\s*</analysis>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()#  match.group(1).strip()
        return ""

    
    async def  generate_report(self, ori_explain_result: list, re_explain_result: list, imp_explain_result: list) -> str:
        prompt = textwrap.dedent(f"""
        <Mission>
        你是一名经验丰富的 DBA，核心任务是生成详细的报告，基于原始 EXPLAIN 分析、重写 EXPLAIN 分析和增强 EXPLAIN 分析。
        你应该考虑并比较它们与包含这些部分的报告：
        1. 成本效率：
            - 总体成本变化百分比  
            - 最昂贵计划节点成本变化  
        2. 计划特征：
            - 扫描类型转换 (例如：Seq Scan → Index Scan)  
            - 连接算法改进 (例如：Hash Join → Merge Join)  
            - 显式排序消除和中间结果集减少   
        3. 资源利用：
            - 内存使用 (Hash/Buffer 节点变化)
            - 工作线程数调整
        4. 其他改进：
                                 
        注意：在你的报告中，不需要复制 EXPLAIN 结果，只需进行 NLP 分析和比较。
                                 
        请严格遵循以下格式：
        [format]
        </analysis>
            ...
        </analysis>
                                 
        </report>
            ... // 不需要重复提及 EXPLAIN 结果
        </report>
                                                      
        <ori_explain_result>
        {ori_explain_result}

        <re_explain_result>
        {re_explain_result}

        <imp_explain_result>
        {imp_explain_result}

        """)
        return await self.llm.get_LLM_response_async(
            prompt=prompt
        )


