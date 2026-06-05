# pipelines/distributed_embed.py
import ray
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import mlflow

if not ray.is_initialized():
    ray.init()

# 멀티 GPU 인프라 토폴로지에 대응할 수 있는 분산 비동기 Ray Actor 정의
@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
class DistributedEmbeddingWorker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("Xenova/text-embedding-3-small", use_fast=True)
        self.model = AutoModel.from_pretrained("Xenova/text-embedding-3-small").to(self.device)
        self.model.eval()

    def generate_embeddings(self, batch_products: list[dict]) -> list[dict]:
        product_texts = [p["product_name"] for p in batch_products]
        
        with torch.no_grad():
            # 진짜 문자열 텍스트 배치 토크나이징 및 GPU 디바이스 텐서 캐스팅
            encoded = self.tokenizer(product_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            model_outputs = self.model(**encoded)
            
            # 정보 손실 방지를 위한 마스크 기반 Mean Pooling 문장 벡터 추출 (시니어 표준)
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            raw_embeddings = (model_outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
            numpy_embeddings = raw_embeddings.cpu().numpy().tolist()

        # 스키마에 고차원 추론 벡터 주입 결합
        for idx, product in enumerate(batch_products):
            product["vector_embedding"] = numpy_embeddings[idx]
            
        return batch_products

def run_distributed_embedding_pipeline(raw_catalog_records: list[dict]):
    # MLOps 표준에 맞춰 인덱싱 이력 추적 중앙 제어 개시
    mlflow.start_run(run_name="Coupang-Production-Distributed-Embedding")
    
    # 워커 스케일에 최적화된 데이터 청크 병렬 분할
    allocated_workers_count = 4
    data_chunks = np.array_split(raw_catalog_records, allocated_workers_count)
    
    workers_pool = [DistributedEmbeddingWorker.remote() for _ in range(allocated_workers_count)]
    
    # 각 노드 GPU 슬롯으로 연산을 동시 분산 비동기 트리거
    async_futures = [workers_pool[i].generate_embeddings.remote(data_chunks[i].tolist()) for i in range(allocated_workers_count)]
    
    # 각 워커 분산 처리 완료값 일괄 취합 수집
    aggregated_final_results = ray.get(async_futures)
    
    mlflow.log_param("total_catalog_items_processed", len(raw_catalog_records))
    mlflow.log_metric("distributed_gpu_embedding_success", 1.0)
    
    print(f"Ray Cluster Distributed Computation Completed. Bulk Export Ready for Vector DB.")
    mlflow.end_run()
