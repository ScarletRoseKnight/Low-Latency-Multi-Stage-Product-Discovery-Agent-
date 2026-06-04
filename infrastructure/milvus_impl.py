# infrastructure/milvus_impl.py
import os
from typing import List, Dict, Any
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
from infrastructure.base_store import BaseVectorStore

class MilvusVectorStore(BaseVectorStore):
    """
    Milvus implementation handling isolated schema creations and Boolean expressions 
    for fast e-commerce metadata partitioning.
    """
    def __init__(self, collection_name: str = "coupang_catalog"):
        self.collection_name = collection_name
        self.vector_dim = 1536

    def connect(self) -> None:
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        connections.connect("default", host=host, port=port)
        
        # Automatically establish schema and collections if missing in local sandbox
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
                FieldSchema(name="is_rocket_delivery", dtype=DataType.BOOL),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=512)
            ]
            schema = CollectionSchema(fields, description="Coupang Catalog Schema")
            collection = Collection(self.collection_name, schema)
            
            # Configure default ANN indexing parameter values for fast calculations
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 64}
            }
            collection.create_index(field_name="vector", index_params=index_params)
            
        self.collection = Collection(self.collection_name)
        self.collection.load() # Load segments into RAM for hot execution paths

    def stage_one_search(self, query_vector: List[float], top_k: int = 50, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        # Map dynamic Python filter rules to Milvus declarative Boolean strings
        expr = None
        if filters and filters.get("is_rocket_delivery") is not None:
            bool_val = "true" if filters["is_rocket_delivery"] else "false"
            expr = f"is_rocket_delivery == {bool_val}"

        search_params = {"metric_type": "COSINE", "params": {"ef": 32}}
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["name", "is_rocket_delivery"]
        )
        
        # Flatten and align result signatures matching the interface template
        standardized_hits = []
        if results:
            for hit in results[0]:
                standardized_hits.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": {
                        "name": hit.entity.get("name"),
                        "is_rocket_delivery": hit.entity.get("is_rocket_delivery")
                    }
                })
        return standardized_hits