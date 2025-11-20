# Import core libraries
import logging
import sys
 
# Import project utilities
from utils.input_output_utils.env_loader import *  # Loads environment variables like table names, paths
from utils.logging_utils.logging_config import setup_logging  # Central logging configuration
 
# Import the main pipeline logic
from src.pipelines.tfd_pipeline_runner import run_tfd_pipeline  # Entrypoint for running the TFD pipeline
 
# --------------------------------------------------------------------------
# GLOBAL INITIALIZATION
# --------------------------------------------------------------------------
 
# Initialize logging system (to console or file, based on config)
setup_logging(log_file=LOGS_FILE, level=LOG_LEVEL)
logger = logging.getLogger("tfd-main")
 
# --------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------
 
def main():
    # If no arguments are provided, default to local sample execution
    if len(sys.argv) < 2:
        logger.warning("No table name provided. Running with local sample SQL.")
        table_name = None
        sample_rows = 100
    else:
        # First argument is the table name; second is sample size (default 100)
        table_name = sys.argv[1]
        sample_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 100
 
    logger.info(f"Running pipeline for table: {table_name or 'LOCAL_SAMPLE'} | Sample size: {sample_rows}")
 
    # Execute the core pipeline logic
    run_tfd_pipeline(table_name=table_name, sample_rows=sample_rows, visualize_username_similarity=True)
 
# Python entrypoint
if __name__ == "__main__":
    main()