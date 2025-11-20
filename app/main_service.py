from datetime import datetime
import logging
import os
import sys
from flask import Flask, request, make_response, jsonify
 
# Set up logging with custom formatting
from utils.logging_utils.logging_config import setup_logging
 
# Import the main pipeline logic
from src.pipelines.tfd_pipeline_runner import run_tfd_pipeline
 
# Initialize Flask app
app = Flask(__name__)
 
# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
 
def get_logger():
    """
    Creates a logger for the service with console output.
    Suitable for containerized environments like SPCS or Docker.
    """
    logger = logging.getLogger("tfd_batch_service")
    logger.setLevel(logging.INFO)
 
    # Stream logs to stdout (for container logs)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
 
    logger.addHandler(handler)
    return logger
 
logger = get_logger()
 
# -----------------------------------------------------------------------------
# Health Check Endpoint
# -----------------------------------------------------------------------------
 
@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    """
    Basic health check endpoint to confirm the service is up.
    """
    return "Service is healthy", 200
 
# -----------------------------------------------------------------------------
# Main Endpoint to Trigger the Pipeline
# -----------------------------------------------------------------------------
 
@app.route("/run", methods=["GET", "POST"])
def run():
    """
    Triggers the Ticket Fraud Detection pipeline.
    Accepts both GET and POST requests with optional parameters:
      - table_name: the name of the Snowflake table to process
      - sample_rows: number of rows to process (default: 100)
    """
    try:
        if request.method == "GET":
            # Handle GET params (URL query)
            table_name = request.args.get("table_name", "UEFA_DEV_DWH.ML_SANDBOX.DS_LOTTERY_AI_DATA_CLEANSING_UECLF24")
            sample_rows = int(request.args.get("sample_rows", 100))
            logger.info(f"[GET] /run → table_name={table_name}, sample_rows={sample_rows}")
 
        elif request.method == "POST":
            # Handle JSON body in POST
            payload = request.get_json(force=True)
            logger.info(f"[POST] /run received JSON payload: {payload}")
            table_name = payload.get("table_name")
            sample_rows = payload.get("sample_rows", 100)
 
        if not table_name:
            return make_response(jsonify({"error": "Missing 'table_name'"}), 400)
 
        # Run the actual pipeline
        df, run_id = run_tfd_pipeline(table_name, sample_rows, visualize_username_similarity=False)
 
        # Construct the response
        response = {
            "status": "success",
            "run_id": run_id,
            "table_name": table_name,
            "rows_processed": len(df),
            "columns_output": len(df.columns),
            "timestamp": datetime.now().isoformat(),
            "fraud_score_summary_path": "data/output/reports/fraud_score_summary.csv",
            "fraud_tier_summary_path": "data/output/reports/fraud_tier_summary.csv",
            "feature_importance_path": "data/output/reports/fraud_feature_importance.csv",
        }
 
        return jsonify(response)
 
    except Exception as e:
        logger.exception("Error while executing TFD pipeline")
        return make_response(jsonify({"status": "error", "message": str(e)}), 500)
 
# -----------------------------------------------------------------------------
# Entrypoint for Container Execution
# -----------------------------------------------------------------------------
 
if __name__ == "__main__":
    SERVICE_HOST = os.getenv("SERVICE_HOST", "0.0.0.0")
    SERVICE_PORT = int(os.getenv("SERVICE_PORT", 8000))
 
    logger.info(f"Starting Ticket Fraud Detection Service on {SERVICE_HOST}:{SERVICE_PORT}")
    app.run(host=SERVICE_HOST, port=SERVICE_PORT)