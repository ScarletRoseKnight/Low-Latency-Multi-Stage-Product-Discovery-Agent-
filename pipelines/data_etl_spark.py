# pipelines/data_etl_spark.py
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def run_petabyte_scale_catalog_etl():
    # 대용량 처리를 위한 Spark 세션 초기화 및 최적화 메모리 할당 정책 수립
    spark = SparkSession.builder \
        .appName("CoupangScale-Clickstream-Aggregator-ETL") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.driver.memory", "16g") \
        .getOrCreate()

    try:
        # 1. 원천 데이터 파일 적재 (BigQuery 연동 레이어 혹은 가상의 데이터 레이크 파티션 경로)
        raw_clickstream_df = spark.read.parquet("hdfs:///analytics/raw_logs/clickstream/*")
        raw_product_catalog_df = spark.read.parquet("hdfs:///analytics/raw_logs/catalog/*")

        # 2. 실시간 광고 타겟팅 및 랭킹 피처 생성을 위한 로그 집계 (Feature Engineering)
        # 상품 ID별 총 노출수, 클릭수, 구매 전환율 계산
        aggregated_features_df = raw_clickstream_df.groupBy("product_id") \
            .agg(
                F.count(F.when(F.col("event_type") == "impression", 1)).alias("total_impressions"),
                F.count(F.when(F.col("event_type") == "click", 1)).alias("total_clicks"),
                F.count(F.when(F.col("event_type") == "purchase", 1)).alias("total_purchases")
            )

        # CTR(클릭률) 및 Conversion Rate(전환율) 피처 생성 후 결측치 0.0 방어 코드 적용
        processed_features_df = aggregated_features_df \
            .withColumn("ctr", F.when(F.col("total_impressions") > 0, F.col("total_clicks") / F.col("total_impressions")).otherwise(0.0)) \
            .withColumn("conversion_rate", F.when(F.col("total_clicks") > 0, F.col("total_purchases") / F.col("total_clicks")).otherwise(0.0))

        # 3. 마스터 상품 카탈로그 테이블과 가공 피처 Left Join 결합
        final_production_catalog = raw_product_catalog_df.join(
            processed_features_df, 
            on="product_id", 
            how="left"
        ).na.fill({"ctr": 0.0, "conversion_rate": 0.0, "total_purchases": 0})

        # 4. Ray 분산 임베딩 워커가 고속으로 읽어갈 수 있도록 정제 완료된 피처 데이터를 스토리지에 파티셔닝 저장
        final_production_catalog.write \
            .mode("overwrite") \
            .partitionBy("category_group") \
            .parquet("hdfs:///analytics/production_features/catalog_gold/")
            
    finally:
        spark.stop()

if __name__ == "__main__":
    run_petabyte_scale_catalog_etl()
