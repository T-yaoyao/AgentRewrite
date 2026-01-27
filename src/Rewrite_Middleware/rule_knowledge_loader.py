"""
规则知识库加载器
从 Rule_Examples.json 加载规则示例，包括 rule_description 和 produced_suggestion
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from src.utils.path_config import PROJECT_ROOT


class RuleKnowledgeLoader:
    """规则知识库加载器"""
    
    def __init__(self, rule_examples_path: Optional[str] = None):
        """
        初始化规则知识库加载器
        
        Args:
            rule_examples_path: Rule_Examples.json 文件路径，如果为None则使用默认路径
        """
        if rule_examples_path is None:
            rule_examples_path = PROJECT_ROOT / "documents" / "effective_rewrite_types" / "Rule_Examples.json"
        else:
            rule_examples_path = Path(rule_examples_path)
        
        self.rule_examples_path = rule_examples_path
        self.rule_knowledge: Dict[str, Dict] = {}
        self._load_rules()
    
    def _load_rules(self):
        """加载规则知识库"""
        try:
            with open(self.rule_examples_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 遍历所有类别和规则
            for category in data:
                for rule in category.get('dsb', []):
                    rule_id = rule.get('id')
                    if rule_id:
                        self.rule_knowledge[rule_id] = {
                            'id': rule_id,
                            'original_query': rule.get('original_query', ''),
                            'rewritten_query': rule.get('rewritten_query', ''),
                            'rule_description': rule.get('rule_description', ''),
                            'rewrite_suggestion': rule.get('rewrite_suggestion', []),
                            # 提取 produced_suggestion
                            'produced_suggestion': self._extract_produced_suggestion(rule.get('rewrite_suggestion', []))
                        }
            
            print(f"✓ 成功加载 {len(self.rule_knowledge)} 个规则示例")
        except FileNotFoundError:
            print(f"⚠️  警告: 规则知识库文件未找到: {self.rule_examples_path}")
        except Exception as e:
            print(f"❌ 加载规则知识库失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_produced_suggestion(self, rewrite_suggestion: List[Dict]) -> str:
        """
        从 rewrite_suggestion 中提取 produced_suggestion
        
        Args:
            rewrite_suggestion: rewrite_suggestion 列表
            
        Returns:
            合并后的 produced_suggestion 文本
        """
        suggestions = []
        for item in rewrite_suggestion:
            if isinstance(item, dict) and 'produced_suggestion' in item:
                suggestions.append(item['produced_suggestion'])
        return ' '.join(suggestions) if suggestions else ''
    
    def get_rule_info(self, rule_id: str) -> Optional[Dict]:
        """
        获取指定规则的完整信息
        
        Args:
            rule_id: 规则ID（如 "AGGREGATE_MERGE"）
            
        Returns:
            规则信息字典，如果不存在则返回None
        """
        return self.rule_knowledge.get(rule_id)
    
    def get_rules_info(self, rule_ids: List[str]) -> List[Dict]:
        """
        获取多个规则的完整信息
        
        Args:
            rule_ids: 规则ID列表
            
        Returns:
            规则信息列表（只包含存在的规则）
        """
        rules_info = []
        for rule_id in rule_ids:
            rule_info = self.get_rule_info(rule_id)
            if rule_info:
                rules_info.append(rule_info)
        return rules_info
    
    def get_rule_description(self, rule_id: str) -> str:
        """
        获取规则的描述
        
        Args:
            rule_id: 规则ID
            
        Returns:
            规则描述，如果不存在则返回空字符串
        """
        rule_info = self.get_rule_info(rule_id)
        return rule_info.get('rule_description', '') if rule_info else ''
    
    def get_produced_suggestion(self, rule_id: str) -> str:
        """
        获取规则的 produced_suggestion
        
        Args:
            rule_id: 规则ID
            
        Returns:
            produced_suggestion，如果不存在则返回空字符串
        """
        rule_info = self.get_rule_info(rule_id)
        return rule_info.get('produced_suggestion', '') if rule_info else ''
    
    def get_rule_example(self, rule_id: str) -> Optional[Dict]:
        """
        获取规则的示例（原始SQL和重写后SQL）
        
        Args:
            rule_id: 规则ID
            
        Returns:
            包含 original_query 和 rewritten_query 的字典，如果不存在则返回None
        """
        rule_info = self.get_rule_info(rule_id)
        if rule_info:
            return {
                'original_query': rule_info.get('original_query', ''),
                'rewritten_query': rule_info.get('rewritten_query', '')
            }
        return None
    
    def format_rules_for_prompt(self, rule_ids: List[str]) -> str:
        """
        格式化规则信息用于 prompt
        
        Args:
            rule_ids: 规则ID列表
            
        Returns:
            格式化后的规则信息字符串
        """
        rules_info = self.get_rules_info(rule_ids)
        
        if not rules_info:
            return "未找到相关规则信息"
        
        formatted_lines = []
        for rule in rules_info:
            rule_id = rule['id']
            description = rule.get('rule_description', '')
            suggestion = rule.get('produced_suggestion', '')
            example = rule.get('original_query', '')
            rewritten = rule.get('rewritten_query', '')
            
            formatted_lines.append(f"规则ID: {rule_id}")
            formatted_lines.append(f"  描述: {description}")
            if suggestion:
                formatted_lines.append(f"  建议: {suggestion}")
            if example:
                formatted_lines.append(f"  示例原始SQL: {example}")
            if rewritten:
                formatted_lines.append(f"  示例重写SQL: {rewritten}")
            formatted_lines.append("")
        
        return "\n".join(formatted_lines)


# 全局单例实例
_rule_loader: Optional[RuleKnowledgeLoader] = None


def get_rule_loader() -> RuleKnowledgeLoader:
    """获取全局规则加载器单例"""
    global _rule_loader
    if _rule_loader is None:
        _rule_loader = RuleKnowledgeLoader()
    return _rule_loader


