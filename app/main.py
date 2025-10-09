import logging
import uuid
from datetime import datetime
import pandas as pd

from utils.input_output_utils.env_loader import *
from utils.connection_utils.snowflake_conn import run_query
from utils.query_utils.query_loader import load_query
from utils.logging_utils.logging_config import setup_logging
from utils.input_output_utils.checkpoints import save_checkpoint, load_checkpoint
from utils.connection_utils.snowflake_conn import get_snowflake_connection

from src.services.data_loader_service import DataLoaderService

from src.pipelines.data_preparation_pipeline import DataPreparationPipeline
from src.pipelines.data_transformation_pipeline import DataTransformationPipeline
from src.pipelines.feature_engineering_pipeline import FeatureEngineeringPipeline
from src.pipelines.clustering_pipeline import ClusteringPipeline

data_loader_service = DataLoaderService()

def main():

    pd.set_option("display.float_format", "{:.3f}".format)

    setup_logging(log_file=LOGS_FILE, level=LOG_LEVEL)
    run_id = str(uuid.uuid4())
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Run UUID: {run_id}")
    logging.info(f"Run Timestamp: {run_time}")
    logging.info("Starting Ticket Fraud Detection Pipeline...")

    logging.info('Hello')


    ## DATA LOADER SERVICE ##

    """df_raw = data_loader_service.data_loader(query_file="ds_lottery_ai_data_cleansing_ueclf_24_sample_100.sql")
    logging.info(f"Loaded raw data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    print('DATA LOADED')"""


    ## RUN QUERY FOR SAMPLE ##

    """query = load_query("ds_lottery_ai_data_cleansing_ueclf_24_sample_100.sql")
    rows, columns = run_query(query)

    df_raw = pd.DataFrame(rows, columns=columns)
    print("✅ Retrieved rows:", len(df_raw))
    print(df_raw.head()) """

    ## SAVE CHECKPOINTS ##
    # Save checkpoint
    """checkpoint_path = save_checkpoint(df_raw, "ds_lottery_ai_data_cleansing_ueclf_24_sample_100points")
    print(checkpoint_path)
    print(f"✅ Saved checkpoint at {checkpoint_path}")"""

    ## WORK OFFLINE WITH SAMPLE DATA ##
    # Load it back
    """test_file_path = "C:\\Users\\juan.prada\\OneDrive - civica-soft.com\\Documentos\\UEFA\\TicketFraudDetectionApp\\app\\data\\checkpoints\\ds_lottery_ai_data_cleansing_ueclf_24_sample_100points.csv"
    df = load_checkpoint(test_file_path) #checkpoint_path.name)
    print("Loaded checkpoint:")
    print(df)"""

    ## DATA PREPARATION PIPELINE

    df = data_loader_service.data_loader(query_file="ds_lottery_ai_data_cleansing_ueclf_24_sample_1000.sql")

    data_preparation_pipeline = DataPreparationPipeline(data_loader_service, df)
    df = data_preparation_pipeline.run()
 
    print(df.head())

    print(df.info())

    ## DATA TRANSFORMATION PIPELINE (updated to pass username-similarity params)
    data_transformation_pipeline = DataTransformationPipeline(df)
 
    df = data_transformation_pipeline.run(
        visualize_username_similarity=True,                 # saves audit CSV (top pairs) if True
        num_perm=128,                                       # MinHash size
        thresholds={"raw": 0.65, "sep": 0.60, "shape": 0.85, "deleet": 0.65},  # per-view LSH thresholds
        min_refined_similarity=0.70,                        # keep user-user pairs >= this after refinement
        neighbor_threshold=2,                               # flag if user has > this many same-cluster neighbors
        cluster_edge_min_sim=0.70,                          # edges used to form clusters
        cluster_lsh_threshold=0.55,                         # LSH threshold for cluster signatures (links)
        cluster_link_min_sim=0.65,                          # keep inter-cluster links >= this
        audit_top_k=50,                                     # how many top pairs to write in audit CSV
        output_dir="."                                      # where audit CSV is written
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
 
    print(df.head())

    pipeline_engineering_pipeline = FeatureEngineeringPipeline()
    df = pipeline_engineering_pipeline.run(df)
 
    print("Feature Engineering")
    print(df)

 
    clustering_pipeline = ClusteringPipeline()
    df = clustering_pipeline.run(df)
    print(df.columns)
    print(df)


if __name__ == "__main__":
    main()
