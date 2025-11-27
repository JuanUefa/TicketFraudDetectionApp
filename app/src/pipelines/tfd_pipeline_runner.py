### src/pipelines/tfd_pipeline_runner.py
 
import logging
import uuid
from datetime import datetime
import pandas as pd
 
from utils.input_output_utils.env_loader import *
from utils.logging_utils.logging_config import setup_logging
from utils.services_utils.clustering_utils import ClusteringUtils
from src.services.data_loader_service import DataLoaderService
from src.pipelines.data_preparation_pipeline import DataPreparationPipeline
from src.pipelines.data_transformation_pipeline import DataTransformationPipeline
from src.pipelines.feature_engineering_pipeline import FeatureEngineeringPipeline
from src.pipelines.clustering_pipeline import ClusteringPipeline
from src.services.output_service import OutputService
from src.services.fraud_scoring_service import FraudScoringService
 
def run_tfd_pipeline(table_name: str = None, sample_rows: int = 100, run_id: str = None, visualize_username_similarity: bool = True) -> pd.DataFrame:
    pd.set_option("display.float_format", "{:.3f}".format)
    run_id = run_id or str(uuid.uuid4())
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 
    logger = logging.getLogger("tfd-pipeline")
    logger.info("=" * 100)
    logger.info("Starting Ticket Fraud Detection Pipeline")
    logger.info(f"Run UUID: {run_id}")
    logger.info(f"Run Timestamp: {run_time}")
    logger.info("=" * 100)
 
    data_loader_service = DataLoaderService()
 
    logger.info(f"Loading data from Snowflake table: {table_name}")
    df_raw = data_loader_service.load_table_dynamic(table_name, sample_rows=sample_rows)
    print('----------------- Loaded Raw Data -----------------')
    logger.info(f"Data loaded successfully: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
 
    data_preparation_pipeline = DataPreparationPipeline(data_loader_service, df_raw)
    df = data_preparation_pipeline.run()
    print('----------------- Prepared Data -----------------')  
    logger.info(f"Data preparation complete. Shape: {df.shape}")
 
    data_transformation_pipeline = DataTransformationPipeline(df)
    df = data_transformation_pipeline.run(
        visualize_username_similarity=visualize_username_similarity,
        num_perm=128,
        thresholds={"raw": 0.65, "sep": 0.60, "shape": 0.85, "deleet": 0.65},
        min_refined_similarity=0.70,
        neighbor_threshold=2,
        cluster_edge_min_sim=0.70,
        cluster_lsh_threshold=0.55,
        cluster_link_min_sim=0.65,
        audit_top_k=50,
        output_dir="."
    )
    logger.info(f"Data transformation complete. Shape: {df.shape}")
 
    feature_engineering_pipeline = FeatureEngineeringPipeline()
    df = feature_engineering_pipeline.run(df)
    logger.info(f"Feature engineering complete. Shape: {df.shape}")
 
    clustering_pipeline = ClusteringPipeline()
    df = clustering_pipeline.run(df)
    logger.info(f"Clustering complete. Shape: {df.shape}")
 
    output_service = OutputService()
    clustering_utils = ClusteringUtils()
    numerical_features, _ = clustering_utils.group_features(df)
 
    output_service.summarize_clustered_numerical_features(df)
    plot_path = output_service.plot_clustered_numerical_features_grid(
        df,
        numerical_features=numerical_features,
        n_cols=3,
        save_plots=True,
        show_plots=False
    )
    logger.info(f"Cluster summary reports and plots generated -> {plot_path}")
 
    logger.info("Running fraud scoring process...")
    fraud_scorer = FraudScoringService()
    df = fraud_scorer.run(df)
    logger.info(f"Fraud scoring complete. Shape: {df.shape}")
 
    data_loader_service.save_df_to_snowflake(df, run_id=run_id)
    logger.info("Data successfully saved to Snowflake.")
    logger.info(f"Pipeline execution completed successfully. Run ID: {run_id}")
    logger.info("=" * 100)
 
    return df, run_id