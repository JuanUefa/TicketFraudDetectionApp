### main.py
 
import logging
import sys
from utils.input_output_utils.env_loader import * 
from utils.logging_utils.logging_config import setup_logging
from src.pipelines.tfd_pipeline_runner import run_tfd_pipeline
 
# --------------------------------------------------------------------------
# GLOBAL INITIALIZATION
# --------------------------------------------------------------------------
setup_logging(log_file=LOGS_FILE, level=LOG_LEVEL)
logger = logging.getLogger("tfd-main")
 
# --------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        logger.warning("No table name provided. Running with local sample SQL.")
        table_name = None
        sample_rows = 100
    else:
        table_name = sys.argv[1]
        sample_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 100
 
    logger.info(f"Running pipeline for table: {table_name or 'LOCAL_SAMPLE'} | Sample size: {sample_rows}")
    run_tfd_pipeline(table_name=table_name, sample_rows=sample_rows, visualize_username_similarity=True)
 
if __name__ == "__main__":
    main()