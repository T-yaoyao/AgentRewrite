"""
Vector Store Module
Manages vector database for storing and retrieving SQL optimization knowledge
"""

import os
import json
import uuid
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Fix SQLite version issue for ChromaDB
# ChromaDB requires sqlite3 >= 3.35.0, but system may have older version
# IMPORTANT: This must happen BEFORE importing chromadb
try:
    import pysqlite3
    import sys
    # Replace the default sqlite3 module with pysqlite3 BEFORE any other imports
    sys.modules['sqlite3'] = pysqlite3
    print("✅ Replaced system sqlite3 with pysqlite3-binary for ChromaDB compatibility")
except ImportError as e:
    print(f"⚠️ pysqlite3-binary not available: {e}")
    print("💡 Install with: pip install pysqlite3-binary")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
    print("✅ ChromaDB imported successfully")
except (ImportError, RuntimeError) as e:
    CHROMADB_AVAILABLE = False
    if isinstance(e, RuntimeError) and "sqlite3" in str(e):
        print("⚠️ SQLite version too old for ChromaDB even with pysqlite3 replacement")
        print("💡 This may be due to import order - pysqlite3 needs to be imported before chromadb")
    else:
        print(f"⚠️ ChromaDB import failed: {e}")
        print("💡 Falling back to in-memory storage")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available, using simple hash-based similarity")


class VectorStore:
    """Vector database manager for SQL optimization knowledge"""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize vector store
        
        Args:
            storage_path: Path to store vector database. If None, uses default location.
        """
        # Set default storage path
        if storage_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            # Get model name from environment variable to distinguish different model tests
            import os
            model_name = os.getenv("REASONING_MODEL", "deepseek-v3.2")
            
            # Extract model identifier for path naming
            # Only support deepseek-v3.2 and deepseek-r1
            if model_name:
                model_name_lower = model_name.lower()
                if "deepseek" in model_name_lower:
                    if "r1" in model_name_lower or "r-1" in model_name_lower:
                        model_id = "deepseek-r1"
                    elif "v3.2" in model_name_lower or "v3-2" in model_name_lower or "v32" in model_name_lower:
                        model_id = "deepseek-v3.2"
                    elif "v3" in model_name_lower:
                        # Default v3 to v3.2
                        model_id = "deepseek-v3.2"
                    else:
                        # Default to v3.2 for other deepseek models
                        model_id = "deepseek-v3.2"
                else:
                    # Default to v3.2 for non-deepseek models
                    model_id = "deepseek-v3.2"
            else:
                model_id = "deepseek-v3.2"
            
            storage_path = str(project_root / "data" / "global_memory" / model_id / "chroma_db")
        
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Set local cache directory for models
                # Create models cache directory in project root
                cache_dir = Path(__file__).parent.parent.parent.parent / "models_cache"
                cache_dir.mkdir(exist_ok=True)

                # Set Hugging Face cache directory and mirror
                os.environ['HF_HOME'] = str(cache_dir / "huggingface")
                os.environ['TRANSFORMERS_CACHE'] = str(cache_dir / "transformers")
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                os.environ['HF_HUB_CACHE'] = str(cache_dir / "huggingface" / "hub")

                # Use a lightweight model for SQL embeddings with local cache
                # Try different model name formats
                model_names = [
                    'sentence-transformers/all-MiniLM-L6-v2',
                    'all-MiniLM-L6-v2'
                ]

                for model_name in model_names:
                    try:
                        self.embedding_model = SentenceTransformer(
                            model_name,
                            cache_folder=str(cache_dir / "sentence_transformers")
                        )
                        print(f"✅ Loaded sentence-transformers model '{model_name}' for embeddings (cached in {cache_dir})")
                        print("🔗 Using Hugging Face mirror: https://hf-mirror.com")
                        break  # Success, exit the loop
                    except Exception as e:
                        print(f"⚠️ Failed to load model '{model_name}': {e}")
                        continue

                if self.embedding_model is None:
                    print("❌ Failed to load any embedding model")
                    print("💡 The system will continue without embeddings (reduced functionality)")
            except Exception as e:
                print(f"⚠️ Failed to initialize embedding model: {e}")
                print("💡 The system will continue without embeddings (reduced functionality)")
        
        # Initialize ChromaDB
        self.collection = None
        if CHROMADB_AVAILABLE:
            try:
                client = chromadb.PersistentClient(
                    path=storage_path,
                    settings=Settings(anonymized_telemetry=False)
                )
                self.collection = client.get_or_create_collection(
                    name="sql_optimization_knowledge",
                    metadata={"description": "SQL optimization knowledge base"}
                )
                print(f"✅ Initialized ChromaDB at {storage_path}")
            except Exception as e:
                print(f"⚠️ Failed to initialize ChromaDB: {e}")
                self.collection = None
        
        # Fallback: in-memory storage
        if self.collection is None:
            self._in_memory_store: Dict[str, Dict] = {}
            print("⚠️ Using in-memory storage (ChromaDB not available)")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text
        
        Args:
            text: Input text (SQL fingerprint)
            
        Returns:
            Embedding vector
        """
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            except Exception as e:
                print(f"⚠️ Embedding generation failed: {e}")
        
        # Fallback: simple hash-based "embedding"
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        # Convert to 128-dim vector (MD5 is 128 bits)
        hash_bytes = hash_obj.digest()
        return [float(b) / 255.0 for b in hash_bytes]
    
    def add(
        self,
        sql_fingerprint: str,
        rule_sequence: List[str],
        groups: str,
        cost_reduction_rate: float,
        original_cost: float,
        rewritten_cost: float,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add a new optimization case to the knowledge base
        
        Args:
            sql_fingerprint: Normalized SQL template
            rule_sequence: List of applied rule IDs
            groups: Optimization groups (comma-separated)
            cost_reduction_rate: Cost reduction percentage
            original_cost: Original query cost
            rewritten_cost: Rewritten query cost
            metadata: Additional metadata
            
        Returns:
            Record ID
        """
        record_id = str(uuid.uuid4())
        embedding = self.generate_embedding(sql_fingerprint)
        
        # Prepare metadata
        record_metadata = {
            "sql_fingerprint": sql_fingerprint,
            "rule_sequence": json.dumps(rule_sequence),
            "groups": groups,
            "cost_reduction_rate": str(cost_reduction_rate),
            "original_cost": str(original_cost),
            "rewritten_cost": str(rewritten_cost),
            "frequency": "1",
            "success": "true"
        }
        
        if metadata:
            record_metadata.update(metadata)
        
        if self.collection:
            try:
                self.collection.add(
                    ids=[record_id],
                    embeddings=[embedding],
                    metadatas=[record_metadata],
                    documents=[sql_fingerprint]
                )
            except Exception as e:
                print(f"⚠️ Failed to add to ChromaDB: {e}")
                # Fallback to in-memory
                self._in_memory_store[record_id] = {
                    "embedding": embedding,
                    "metadata": record_metadata,
                    "document": sql_fingerprint
                }
        else:
            # In-memory storage
            self._in_memory_store[record_id] = {
                "embedding": embedding,
                "metadata": record_metadata,
                "document": sql_fingerprint
            }
        
        return record_id
    
    def query(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """
        Query similar SQL fingerprints
        
        Args:
            query_text: Query SQL fingerprint
            top_k: Number of results to return
            
        Returns:
            List of similar records with scores
        """
        query_embedding = self.generate_embedding(query_text)
        results = []
        
        if self.collection:
            try:
                # Query ChromaDB
                db_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
                
                # Format results
                if db_results['ids'] and len(db_results['ids'][0]) > 0:
                    for i in range(len(db_results['ids'][0])):
                        result = {
                            "id": db_results['ids'][0][i],
                            "score": 1.0 - db_results['distances'][0][i] if 'distances' in db_results else 0.0,
                            "metadata": db_results['metadatas'][0][i],
                            "document": db_results['documents'][0][i] if 'documents' in db_results else ""
                        }
                        results.append(result)
            except Exception as e:
                print(f"⚠️ ChromaDB query failed: {e}")
                # Fallback to in-memory
                results = self._query_in_memory(query_embedding, top_k)
        else:
            # In-memory query
            results = self._query_in_memory(query_embedding, top_k)
        
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _query_in_memory(self, query_embedding: List[float], top_k: int) -> List[Dict]:
        """Query in-memory store using cosine similarity"""
        import math
        
        def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            """Calculate cosine similarity"""
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            return dot_product / (magnitude1 * magnitude2)
        
        similarities = []
        for record_id, record in self._in_memory_store.items():
            similarity = cosine_similarity(query_embedding, record['embedding'])
            similarities.append({
                "id": record_id,
                "score": similarity,
                "metadata": record['metadata'],
                "document": record['document']
            })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]
    
    def update_frequency(self, record_id: str):
        """Update hit frequency for a record"""
        if self.collection:
            try:
                # Get current metadata
                result = self.collection.get(ids=[record_id])
                if result['metadatas'] and len(result['metadatas']) > 0:
                    metadata = result['metadatas'][0]
                    current_freq = int(metadata.get('frequency', '0'))
                    metadata['frequency'] = str(current_freq + 1)
                    
                    # Update in ChromaDB
                    self.collection.update(
                        ids=[record_id],
                        metadatas=[metadata]
                    )
            except Exception as e:
                print(f"⚠️ Failed to update frequency: {e}")
        else:
            # In-memory update
            if record_id in self._in_memory_store:
                metadata = self._in_memory_store[record_id]['metadata']
                current_freq = int(metadata.get('frequency', '0'))
                metadata['frequency'] = str(current_freq + 1)
    
    def delete(self, record_id: str) -> bool:
        """
        Delete a record by ID
        
        Args:
            record_id: Record ID to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if self.collection:
            try:
                self.collection.delete(ids=[record_id])
                return True
            except Exception as e:
                print(f"⚠️ Failed to delete from ChromaDB: {e}")
                return False
        else:
            # In-memory delete
            if record_id in self._in_memory_store:
                del self._in_memory_store[record_id]
                return True
            return False
    
    def clear_all(self) -> bool:
        """
        Clear all records from the knowledge base
        
        Returns:
            True if cleared successfully, False otherwise
        """
        if self.collection:
            try:
                # Get all IDs
                all_ids = self.collection.get()['ids']
                if all_ids:
                    self.collection.delete(ids=all_ids)
                return True
            except Exception as e:
                print(f"⚠️ Failed to clear ChromaDB: {e}")
                return False
        else:
            # In-memory clear
            self._in_memory_store.clear()
            return True
    
    def get_all_ids(self) -> List[str]:
        """
        Get all record IDs in the knowledge base
        
        Returns:
            List of record IDs
        """
        if self.collection:
            try:
                return self.collection.get()['ids']
            except Exception as e:
                print(f"⚠️ Failed to get IDs from ChromaDB: {e}")
                return []
        else:
            return list(self._in_memory_store.keys())

