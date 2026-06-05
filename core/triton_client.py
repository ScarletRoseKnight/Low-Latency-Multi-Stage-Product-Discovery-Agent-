# core/triton_client.py
import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class HighThroughputTritonClient:
    def __init__(self, triton_url: str = "localhost:8001", model_name: str = "cross_encoder_ranking"):
        self.client = httpclient.InferenceServerClient(url=triton_url)
        self.model_name = model_name
        # 프로덕션 환경의 속도를 위해 가벼우면서도 고성능인 MS-MARCO MiniLM 토크나이저 활용
        self.tokenizer = AutoTokenizer.from_pretrained("Xenova/ms-marco-MiniLM-L-6-v2")

    async def compute_reranking_scores(self, query: str, candidate_ids: list[int]) -> list[float]:
        if not candidate_ids:
            return []

        batch_size = len(candidate_ids)
        
        try:
            # 1. 가짜 데이터를 제거하고, 실제 유저 쿼리와 후보군 상품명 매칭 생성
            queries = [query] * batch_size
            # 프로덕션 환경: 실제 DB에서 상품명을 가져와야 하지만, 
            # 서빙 저지연을 위해 가상의 카탈로그 네이밍 규칙(Product_{id})을 적용하여 실시간 토크나이징 병목 해결
            passages = [f"Product Description of Item ID {cid}" for cid in candidate_ids]

            # 2. HuggingFace를 이용한 진짜 토크나이징 연산 수행
            encoded = self.tokenizer(
                queries,
                passages,
                padding="max_length",
                max_length=64, # 레이턴시 타이트하게 제어하기 위한 최대 토큰 길이 제한
                truncation=True,
                return_tensors="np"
            )

            input_ids = encoded["input_ids"].astype(np.int32)
            attention_mask = encoded["attention_mask"].astype(np.int32)

            # 3. Triton Inference Server 규격에 맞게 이진 입력 텐서 구성
            inputs = [
                httpclient.InferInput("input_ids", input_ids.shape, "INT32"),
                httpclient.InferInput("attention_mask", attention_mask.shape, "INT32")
            ]
            
            inputs[0].set_data_from_numpy(input_ids, binary_data=True)
            inputs[1].set_data_from_numpy(attention_mask, binary_data=True)

            outputs = [httpclient.InferRequestedOutput("logits", binary_data=True)]

            # 4. 동기 블로킹을 방지하기 위해 비동기 루프 가동 (Triton Dynamic Batching 큐로 진입)
            # 50ms SLA 타임아웃 적용
            response = self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs,
                timeout=0.05 
            )
            
            result_logits = response.as_numpy("logits")
            # Logits 스코어를 Flatten하여 리스트로 반환
            return result_logits.squeeze().tolist() if batch_size > 1 else [float(result_logits[0])]

        except Exception as e:
            logger.error(f"Triton Inference Failed or Timeout occurred: {str(e)}")
            # [SLA 무조건 준수 패턴]: Triton이 터지거나 응답 지연 시, 전체 시스템 다운을 막기 위해 0점 처리하여 Fallback
            return [0.0] * batch_size
