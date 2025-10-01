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

data_loader_service = DataLoaderService()

def main():

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
    test_file_path = "C:\\Users\\juan.prada\\OneDrive - civica-soft.com\\Documentos\\UEFA\\TicketFraudDetectionApp\\app\\data\\checkpoints\\ds_lottery_ai_data_cleansing_ueclf_24_sample_100points.csv"
    df = load_checkpoint(test_file_path) #checkpoint_path.name)
    print("✅ Loaded checkpoint:")
    print(df)

    ## DATA PREPARATION PIPELINE

    data_preparation_pipeline = DataPreparationPipeline(data_loader_service, df)
 
    df = data_preparation_pipeline.run()
 
    print(df.head())

    print(df.info())

    data_transformation_pipeline = DataTransformationPipeline(df)
    df = data_transformation_pipeline.run()

    print(df.head())


if __name__ == "__main__":
    main()
