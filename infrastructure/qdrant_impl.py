# infrastructure/qdrant_impl.py
import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from infrastructure.base_store import BaseVectorStore

class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant implementation optimized for pre-filtering operations 
    on the HNSW graph layer.
    """
    def __init__(self, collection_name: str = "coupang_catalog"):
        self.collection_name = collection_name
        self.client = None
        self.vector_dim = 1536  # Default dimension for text-embedding-3-small

    def connect(self) -> None:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        # prefer_grpc=True uses HTTP/2 for ultra-low connection multiplexing overhead
        self.client = QdrantClient(host=host, port=port, prefer_grpc=True)
        
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )

    def stage_one_search(self, query_vector: List[float], top_k: int = 50, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        qdrant_filter = None
        
        # Staff Engineering Archetype: Pre-filter at DB level to protect LLM context windows
        if filters:
            conditions = []
            if filters.get("is_rocket_delivery") is not None:
                conditions.append(FieldCondition(key="is_rocket_delivery", match=MatchValue(value=filters["is_rocket_delivery"])))
            qdrant_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter
        )
        
        # Standardize return payload matching the interface contract
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]