import logging
import sys
import uuid
from datetime import datetime
import pandas as pd
 
from utils.input_output_utils.env_loader import *
from utils.logging_utils.logging_config import setup_logging
 
from src.services.data_loader_service import DataLoaderService
from src.pipelines.data_preparation_pipeline import DataPreparationPipeline
from src.pipelines.data_transformation_pipeline import DataTransformationPipeline
from src.pipelines.feature_engineering_pipeline import FeatureEngineeringPipeline
from src.pipelines.clustering_pipeline import ClusteringPipeline
 
 
# --------------------------------------------------------------------------
# GLOBAL INITIALIZATION
# --------------------------------------------------------------------------
data_loader_service = DataLoaderService()
setup_logging(log_file=LOGS_FILE, level=LOG_LEVEL)
logger = logging.getLogger("tfd-main")
 
 
# --------------------------------------------------------------------------
# CORE PIPELINE FUNCTION (SHARED BETWEEN main.py AND main_service.py)
# --------------------------------------------------------------------------
def run_tfd_pipeline(table_name: str = None, sample_rows: int = 100, run_id: str = None) -> pd.DataFrame:
    """
    Core Ticket Fraud Detection Pipeline.
 
    Parameters
    ----------
    table_name : str
        Snowflake table name to load (if None, uses sample SQL file).
    sample_rows : int
        Number of rows to sample from Snowflake (default=100).
    run_id : str
        Optional unique ID for tracking pipeline runs.
 
    Returns
    -------
    pd.DataFrame
        Final transformed and clustered DataFrame (saved to Snowflake).
    """
 
    # --- Run setup ---
    pd.set_option("display.float_format", "{:.3f}".format)
    run_id = run_id or str(uuid.uuid4())
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 
    logger.info("=" * 80)
    logger.info(f"Starting Ticket Fraud Detection Pipeline")
    logger.info(f"Run UUID: {run_id}")
    logger.info(f"Run Timestamp: {run_time}")
    logger.info("=" * 80)
 
    # --- Step 1: Load Data ---
    if table_name:
        logger.info(f"Loading data dynamically from Snowflake table: {table_name}")
        df_raw = data_loader_service.load_table_dynamic(table_name, sample_rows=sample_rows)
    else:
        logger.info("Loading data from default local SQL sample file...")
        df_raw = data_loader_service.data_loader(query_file="ds_lottery_ai_data_cleansing_ueclf_24_sample_100.sql")
 
    logger.info(f"Data loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
 
    # --- Step 2: Data Preparation ---
    data_preparation_pipeline = DataPreparationPipeline(data_loader_service, df_raw)
    df = data_preparation_pipeline.run()
    logger.info(f"Data Preparation complete. Shape: {df.shape}")
 
    # --- Step 3: Data Transformation ---
    data_transformation_pipeline = DataTransformationPipeline(df)
    df = data_transformation_pipeline.run(
        visualize_username_similarity=True,
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
    ## SLIGHTLY MORE PERMISIVE SETUP
    """df = data_transformation_pipeline.run(
        visualize_username_similarity=True,
        num_perm=128,
        thresholds={"raw": 0.60, "sep": 0.55, "shape": 0.80, "deleet": 0.60},
        min_refined_similarity=0.65,
        neighbor_threshold=2,
        cluster_edge_min_sim=0.65,
        cluster_lsh_threshold=0.55,
        cluster_link_min_sim=0.65,
        audit_top_k=50,
        output_dir="."
    )"""
 
    logger.info(f"Data Transformation complete. Shape: {df.shape}")
 
    # --- Step 4: Feature Engineering ---
    pipeline_engineering_pipeline = FeatureEngineeringPipeline()
    df = pipeline_engineering_pipeline.run(df)
    logger.info(f"Feature Engineering complete. Shape: {df.shape}")
 
    # --- Step 5: Clustering ---
    clustering_pipeline = ClusteringPipeline()
    df = clustering_pipeline.run(df)
    logger.info(f"Clustering complete. Shape: {df.shape}")
 
    # --- Step 6: Save Results ---
    data_loader_service.save_df_to_snowflake(df, run_id=run_id)
    logger.info(f"Data saved successfully to Snowflake.")
    logger.info(f"Pipeline finished. Run ID: {run_id}")
 
    return df
 
 
# --------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------
def main():
    """
    Local CLI Entry Point.
    Example usage:
        python main.py DS_LOTTERY_AI_DATA_CLEANSING_UECLF24 100
    """
 
    if len(sys.argv) < 2:
        logger.warning("No table name provided. Running with local sample SQL.")
        table_name = None
        sample_rows = 100
    else:
        table_name = sys.argv[1]
        sample_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 100
 
    logger.info(f"Running pipeline for table: {table_name or 'LOCAL_SAMPLE'} | Sample size: {sample_rows}")
    run_tfd_pipeline(table_name=table_name, sample_rows=sample_rows)
 
 
if __name__ == "__main__":
    main()