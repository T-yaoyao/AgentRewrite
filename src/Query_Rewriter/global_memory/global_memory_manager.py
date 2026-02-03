"""
Global Memory Manager
Main interface for global knowledge base operations
"""

from typing import List, Dict, Optional
from .vector_store import VectorStore
from .knowledge_retriever import KnowledgeRetriever
from .knowledge_storage import KnowledgeStorage


class GlobalMemoryManager:
    """Main manager for global memory/knowledge base"""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize global memory manager
        
        Args:
            storage_path: Path to store vector database
        """
        self.vector_store = VectorStore(storage_path)
        self.retriever = KnowledgeRetriever(self.vector_store)
        self.storage = KnowledgeStorage(self.vector_store)
    
    def retrieve(self, sql: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve similar optimization cases
        
        Args:
            sql: SQL query
            top_k: Number of results
            
        Returns:
            List of similar cases
        """
        return self.retriever.retrieve(sql, top_k=top_k)
    
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
        Store successful optimization case
        
        Args:
            original_sql: Original SQL
            rewritten_sql: Rewritten SQL
            rule_sequence: Applied rules
            groups: Optimization groups
            original_cost: Original cost
            rewritten_cost: Rewritten cost
            metadata: Additional metadata
            
        Returns:
            Record ID if stored
        """
        return self.storage.store_successful_optimization(
            original_sql=original_sql,
            rewritten_sql=rewritten_sql,
            rule_sequence=rule_sequence,
            groups=groups,
            original_cost=original_cost,
            rewritten_cost=rewritten_cost,
            metadata=metadata
        )
    
    def update_hit_frequency(self, record_id: str):
        """Update hit frequency for a record"""
        self.storage.update_hit_frequency(record_id)
    
    def delete_record(self, record_id: str) -> bool:
        """
        Delete a record by ID
        
        Args:
            record_id: Record ID to delete
            
        Returns:
            True if deleted successfully
        """
        return self.vector_store.delete(record_id)
    
    def clear_all(self) -> bool:
        """
        Clear all records from the knowledge base
        
        Returns:
            True if cleared successfully
        """
        return self.vector_store.clear_all()
    
    def list_all_records(self) -> List[Dict]:
        """
        List all records in the knowledge base
        
        Returns:
            List of record IDs and metadata
        """
        record_ids = self.vector_store.get_all_ids()
        records = []
        for record_id in record_ids:
            # Get metadata for each record
            results = self.vector_store.query("", top_k=1000)  # Get all
            for result in results:
                if result.get('id') == record_id:
                    records.append({
                        'id': record_id,
                        'sql_fingerprint': result.get('metadata', {}).get('sql_fingerprint', '')[:100],
                        'rule_sequence': result.get('metadata', {}).get('rule_sequence', ''),
                        'cost_reduction_rate': result.get('metadata', {}).get('cost_reduction_rate', '0'),
                        'frequency': result.get('metadata', {}).get('frequency', '0')
                    })
                    break
        return records

