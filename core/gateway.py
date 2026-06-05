# core/gateway.py
import asyncio
import time
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from core.triton_client import HighThroughputTritonClient
from infrastructure.qdrant_impl import QdrantVectorStore

# 실제 Production 컨테이너 내부에서 가동될 ASGI 웹 애플리케이션 정의
app = FastAPI(title="Coupang Production Discovery Gateway Engine")

class SearchResponse(BaseModel):
    status: str
    query: str
    server_latency_ms: float
    results: list[int]

class HighThroughputAgentGateway:
    def __init__(self):
        self.triton_client = HighThroughputTritonClient()
        self.vector_store = QdrantVectorStore(host="localhost", port=6333)

    async def pipeline_search_execution(self, query: str, user_id: str, top_k: int = 5) -> list[int]:
        # Stage-1: 초저지연 시맨틱 벡터 서치 (후보군 50개 제한 추출)
        candidate_ids = await self.vector_store.stage_one_search(query, limit=50)
        
        if not candidate_ids:
            return []

        # Stage-2: Triton Server 기반 고정밀 Cross-Encoder 모델 실시간 스코어링
        triton_scores = await self.triton_client.compute_reranking_scores(query, candidate_ids)

        # Stage-3: [비즈니스 로직 결합] 딥러닝 문맥 점수 + 쿠팡 광고(Ad Sponsored) 가중합
        final_ranked_items = []
        for idx, cid in enumerate(candidate_ids):
            base_ml_score = triton_scores[idx] if idx < len(triton_scores) else 0.0
            
            # Redis 같은 고속 분산 캐시 데이터 마트 연동 구조를 모사
            is_ad_sponsored = (cid % 7 == 0)  # 7의 배수 ID 상품은 광고주가 입찰한 광고 상품으로 정의
            ad_boost_multiplier = 0.35 if is_ad_sponsored else 0.0
            
            # 실시간 상품 매력도(CTR 보정값) 결합
            ctr_score_weight = 0.12 if (cid % 3 == 0) else 0.0
            
            # Multi-Objective Optimization 최종 랭킹 점수 산출
            final_score = base_ml_score + ad_boost_multiplier + ctr_score_weight
            final_ranked_items.append((cid, final_score))

        # 4. 결합된 하이브리드 비즈니스 스코어 기준으로 최종 내림차순 정렬
        final_ranked_items.sort(key=lambda x: x[1], reverse=True)
        
        return [item[0] for item in final_ranked_items[:top_k]]

gateway = HighThroughputAgentGateway()

# 외부 상용 트래픽을 안전하게 수신할 고성능 비동기 API 엔드포인트 노출
@app.post("/v1/predict/search", response_model=SearchResponse)
async def real_time_search_serving(
    query: str = Query(..., description="User search query string"),
    user_id: str = Query(..., description="Unique user identification hash token")
):
    start_time = time.perf_counter()
    try:
        # 블로킹이 없는 완전 비동기 파이프라인 수행
        top_products = await gateway.pipeline_search_execution(query, user_id)
        end_time = time.perf_counter()
        
        return SearchResponse(
            status="success",
            query=query,
            server_latency_ms=round((end_time - start_time) * 1000, 2),
            results=top_products
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discovery Pipeline Runtime Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # 하드웨어 멀티 코어를 완전히 소모하여 처리량을 극대화하는 4워커 기동 규칙 적용
    uvicorn.run("gateway:app", host="0.0.0.0", port=8000, workers=4, loop="uvloop")
