"""
Knowledge Retriever Module
Retrieves similar SQL optimization cases from knowledge base
"""

import json
from typing import List, Dict, Optional
from .vector_store import VectorStore
from .sql_fingerprint import SQLFingerprintGenerator


class KnowledgeRetriever:
    """Retrieve optimization knowledge from vector database"""
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize knowledge retriever
        
        Args:
            vector_store: VectorStore instance
        """
        self.vector_store = vector_store
        self.fingerprint_generator = SQLFingerprintGenerator()
    
    def retrieve(self, sql: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve similar SQL optimization cases
        
        Args:
            sql: Original SQL query
            top_k: Number of results to return
            
        Returns:
            List of similar cases with scores and metadata
        """
        # Generate SQL fingerprint
        fingerprint = self.fingerprint_generator.get_template(sql)
        
        if not fingerprint:
            return []
        
        # Query vector store
        results = self.vector_store.query(fingerprint, top_k=top_k)
        
        # Format results
        formatted_results = []
        for result in results:
            metadata = result['metadata']
            try:
                rule_sequence = json.loads(metadata.get('rule_sequence', '[]'))
            except:
                rule_sequence = []
            
            formatted_result = {
                "id": result['id'],
                "score": result['score'],
                "sql_fingerprint": metadata.get('sql_fingerprint', ''),
                "rule_sequence": rule_sequence,
                "groups": metadata.get('groups', ''),
                "cost_reduction_rate": float(metadata.get('cost_reduction_rate', '0')),
                "original_cost": float(metadata.get('original_cost', '0')),
                "rewritten_cost": float(metadata.get('rewritten_cost', '0')),
                "frequency": int(metadata.get('frequency', '0'))
            }
            formatted_results.append(formatted_result)
        
        return formatted_results

