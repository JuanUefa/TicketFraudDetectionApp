import logging
import os
import sys
import uuid
from datetime import datetime
from flask import Flask, request, make_response, jsonify
 
from utils.input_output_utils.env_loader import *
from utils.logging_utils.logging_config import setup_logging
from src.services.data_loader_service import DataLoaderService
from src.pipelines.data_preparation_pipeline import DataPreparationPipeline
from src.pipelines.data_transformation_pipeline import DataTransformationPipeline
from src.pipelines.feature_engineering_pipeline import FeatureEngineeringPipeline
from src.pipelines.clustering_pipeline import ClusteringPipeline
 
 
app = Flask(__name__)
 
# --- Logging setup ---
def get_logger():
    logger = logging.getLogger("tfd_batch_service")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger
 
logger = get_logger()
 
# --- Core function ---
def run_tfd_pipeline(table_name: str, sample_rows: int = None):
    logger.info(f"Starting TFD pipeline for table: {table_name}, sample_rows={sample_rows}")
 
    data_loader_service = DataLoaderService()
    query = f"SELECT * FROM {table_name}"
    if sample_rows:
        query += f" LIMIT {sample_rows}"
 
    # Load data
    df = data_loader_service.data_loader(query_string=query)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from Snowflake.")
 
    # Run pipeline steps
    data_preparation_pipeline = DataPreparationPipeline(data_loader_service, df)
    df = data_preparation_pipeline.run()
 
    data_transformation_pipeline = DataTransformationPipeline(df)
    df = data_transformation_pipeline.run()
 
    pipeline_engineering_pipeline = FeatureEngineeringPipeline()
    df = pipeline_engineering_pipeline.run(df)
 
    clustering_pipeline = ClusteringPipeline()
    df = clustering_pipeline.run(df)
 
    # Save output
    run_id = str(uuid.uuid4())
    data_loader_service.save_df_to_snowflake(df, run_id=run_id)
 
    logger.info(f"Pipeline completed successfully for run_id={run_id}")
    return {
        "status": "success",
        "run_id": run_id,
        "table_name": table_name,
        "rows_processed": len(df),
        "columns_output": len(df.columns),
        "timestamp": datetime.now().isoformat()
    }
 
# --- Flask routes ---
@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return "Service is healthy", 200
 
@app.route("/run", methods=["GET", "POST"])
def run():
    try:
        if request.method == "GET":
            table_name = request.args.get("table_name", "UEFA_DEV_DWH.ML_SANDBOX.DS_LOTTERY_AI_DATA_CLEANSING_UECLF24")
            sample_rows = int(request.args.get("sample_rows", 100))
            logger.info(f"GET /run: {table_name}, sample_rows={sample_rows}")
        elif request.method == "POST":
            payload = request.get_json(force=True)
            logger.info(f"POST /run received payload: {payload}")
            table_name = payload.get("table_name")
            sample_rows = payload.get("sample_rows", 100)
        result = run_tfd_pipeline(table_name, sample_rows)
        return jsonify(result)
    except Exception as e:
        logger.exception("Error running TFD pipeline")
        return make_response(jsonify({"status": "error", "message": str(e)}), 500)
 
# --- Entrypoint ---
if __name__ == "__main__":
    logger.info(f"Starting Ticket Fraud Detection Service on {SERVICE_HOST}:{SERVICE_PORT}")
    app.run(host=SERVICE_HOST, port=SERVICE_PORT)