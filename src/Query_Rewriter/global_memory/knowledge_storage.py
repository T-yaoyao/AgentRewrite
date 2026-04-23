"""
Knowledge Storage Module
Stores successful SQL optimization cases to knowledge base
"""

from typing import List, Dict, Optional
from .vector_store import VectorStore
from .sql_fingerprint import SQLFingerprintGenerator


class KnowledgeStorage:
    """Store optimization knowledge to vector database"""
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize knowledge storage
        
        Args:
            vector_store: VectorStore instance
        """
        self.vector_store = vector_store
        self.fingerprint_generator = SQLFingerprintGenerator()
    
    def store_successful_optimization(
        self,
        original_sql: str,
        rewritten_sql: str,
        rule_sequence: List[str],
        groups: str,
        original_cost: float,
        rewritten_cost: float,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Store a kept rewrite case: rewritten SQL must differ from original and yield a SQL fingerprint.
        
        Args:
            original_sql: Original SQL query
            rewritten_sql: Rewritten SQL query
            rule_sequence: List of applied rule IDs
            groups: Optimization groups (comma-separated)
            original_cost: Original query cost
            rewritten_cost: Rewritten query cost
            metadata: Additional metadata
            
        Returns:
            Record ID if stored successfully, None otherwise
        """
        if not rewritten_sql or (rewritten_sql.strip() == original_sql.strip()):
            print("⚠️ 跳过存储：重写 SQL 与原始相同或为空")
            return None

        # Generate fingerprint from original SQL
        fingerprint = self.fingerprint_generator.get_template(original_sql)
        
        if not fingerprint:
            print(f"⚠️ 跳过存储：无法生成SQL指纹")
            return None
        
        # Prepare metadata
        storage_metadata = metadata or {}
        storage_metadata.update({
            "original_sql": original_sql,
            "rewritten_sql": rewritten_sql
        })
        
        # Store in vector database
        try:
            if original_cost > 0:
                cost_reduction_rate = (original_cost - rewritten_cost) / original_cost
                cost_note = f"{cost_reduction_rate * 100:.2f}%"
            else:
                cost_note = "N/A（原始估计代价为0或未解析）"
            print(f"💾 正在存储优化案例到向量数据库...")
            print(f"   - SQL指纹长度: {len(fingerprint)}")
            print(f"   - 规则序列: {rule_sequence}")
            print(f"   - 优化组别: {groups}")
            print(f"   - 估计代价: 原始={original_cost}, 重写={rewritten_cost}，相对变化: {cost_note}")
            
            record_id = self.vector_store.add(
                sql_fingerprint=fingerprint,
                rule_sequence=rule_sequence,
                groups=groups,
                metadata=storage_metadata,
            )
            
            if record_id:
                print(f"✅ 成功存储，记录ID: {record_id}")
            else:
                print(f"⚠️ 存储返回None")
            
            return record_id
        except Exception as e:
            print(f"⚠️ 存储优化案例失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def update_hit_frequency(self, record_id: str):
        """Update hit frequency for a record"""
        try:
            self.vector_store.update_frequency(record_id)
        except Exception as e:
            print(f"⚠️ Failed to update frequency: {e}")

