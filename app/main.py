import logging
import uuid
from datetime import datetime
import pandas as pd

from utils.env_loader import *
from utils.snowflake_conn import run_query
from utils.query_loader import load_query
from utils.logging_config import setup_logging
from utils.checkpoints import save_checkpoint, load_checkpoint
from utils.snowflake_conn import get_snowflake_connection


def main():

    setup_logging(log_file=LOGS_FILE, level=LOG_LEVEL)
    run_id = str(uuid.uuid4())
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Run UUID: {run_id}")
    logging.info(f"Run Timestamp: {run_time}")
    logging.info("Starting Ticket Fraud Detection Pipeline...")

    logging.info('Hello')

    ## GET SAMPLE DATA TO DEV MVP ##

    if GET_SAMPLE_DATA:

        logging.info('Getting Sample Data')

        query = load_query("ds_lottery_ai_data_cleansing_ueclf_24_sample_100.sql")
        rows, columns = run_query(query)

        df_raw = pd.DataFrame(rows, columns=columns)
        print("✅ Retrieved rows:", len(df_raw))
        print(df_raw.head())

        # Save checkpoint
        checkpoint_path = save_checkpoint(df_raw, "ds_lottery_ai_data_cleansing_ueclf_24_sample_100points")
        print(f"✅ Saved checkpoint at {checkpoint_path}")

        # Load it back
        df_loaded = load_checkpoint(checkpoint_path.name)
        print("✅ Loaded checkpoint:")
        print(df_loaded)


if __name__ == "__main__":
    main()
