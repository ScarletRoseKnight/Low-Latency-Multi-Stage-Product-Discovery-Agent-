# infrastructure/base_store.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseVectorStore(ABC):
    """
    Abstract Base Class acting as the Interface for Coupang-scale vector stores.
    Ensures absolute decoupling between core agent loops and database-specific SDKs.
    """
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection or initialization configurations to the vector cluster."""
        pass

    @abstractmethod
    def stage_one_search(self, query_vector: List[float], top_k: int = 50, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a high-recall, low-latency similarity search over high-dimensional vectors.
        Must support database-level metadata filtering (Pre-filtering).
        """
        pass