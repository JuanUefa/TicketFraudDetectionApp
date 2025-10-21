import logging
import os
import sys
import uuid
from datetime import datetime
from flask import Flask, request, make_response, jsonify
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Force non-GUI backend for headless execution
 
# --------------------------------------------------------------------------
# Internal Imports
# --------------------------------------------------------------------------
from utils.input_output_utils.env_loader import *
from utils.logging_utils.logging_config import setup_logging
 
from src.services.data_loader_service import DataLoaderService
from src.pipelines.data_preparation_pipeline import DataPreparationPipeline
from src.pipelines.data_transformation_pipeline import DataTransformationPipeline
from src.pipelines.feature_engineering_pipeline import FeatureEngineeringPipeline
from src.pipelines.clustering_pipeline import ClusteringPipeline
from src.services.output_service import OutputService
from utils.services_utils.clustering_utils import ClusteringUtils
from src.services.fraud_scoring_service import FraudScoringService
 
# --------------------------------------------------------------------------
# Flask App Setup
# --------------------------------------------------------------------------
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
 
 
# --------------------------------------------------------------------------
# CORE PIPELINE FUNCTION
# --------------------------------------------------------------------------
def run_tfd_pipeline(table_name: str, sample_rows: int = None):
    """
    Core Ticket Fraud Detection Pipeline (Service version).
    Mirrors main.py but designed for API/Snowflake execution.
    """
    pd.set_option("display.float_format", "{:.3f}".format)
    run_id = str(uuid.uuid4())
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 
    logger.info("=" * 100)
    logger.info(f"Starting Ticket Fraud Detection Pipeline (Service Mode)")
    logger.info(f"Run UUID: {run_id}")
    logger.info(f"Run Timestamp: {run_time}")
    logger.info("=" * 100)
 
    # ------------------------------------------------------------------
    # STEP 1 — DATA LOADING
    # ------------------------------------------------------------------
    data_loader_service = DataLoaderService()
    query = f"SELECT * FROM {table_name}"
    if sample_rows:
        query += f" LIMIT {sample_rows}"
 
    df = data_loader_service.data_loader(query_string=query)
    logger.info(f"Data loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
 
    # ------------------------------------------------------------------
    # STEP 2 — DATA PREPARATION
    # ------------------------------------------------------------------
    data_preparation_pipeline = DataPreparationPipeline(data_loader_service, df)
    df = data_preparation_pipeline.run()
    logger.info(f"Data preparation complete. Shape: {df.shape}")
 
    # ------------------------------------------------------------------
    # STEP 3 — DATA TRANSFORMATION
    # ------------------------------------------------------------------
    data_transformation_pipeline = DataTransformationPipeline(df)
    df = data_transformation_pipeline.run(
        visualize_username_similarity=False,  # Disabled for API mode
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
 
    # ------------------------------------------------------------------
    # STEP 4 — FEATURE ENGINEERING
    # ------------------------------------------------------------------
    feature_engineering_pipeline = FeatureEngineeringPipeline()
    df = feature_engineering_pipeline.run(df)
    logger.info(f"Feature engineering complete. Shape: {df.shape}")
 
    # ------------------------------------------------------------------
    # STEP 5 — CLUSTERING
    # ------------------------------------------------------------------
    clustering_pipeline = ClusteringPipeline()
    df = clustering_pipeline.run(df)
    logger.info(f"Clustering complete. Shape: {df.shape}")
 
    # ------------------------------------------------------------------
    # STEP 6 — OUTPUT SUMMARIES & VISUALIZATIONS
    # ------------------------------------------------------------------
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
 
    # ------------------------------------------------------------------
    # STEP 7 — FRAUD SCORING
    # ------------------------------------------------------------------
    logger.info("Running fraud scoring process...")
    fraud_scorer = FraudScoringService()
    df = fraud_scorer.run(df)
    logger.info(f"Fraud scoring complete. Shape: {df.shape}")
 
    # ------------------------------------------------------------------
    # STEP 8 — SAVE RESULTS TO SNOWFLAKE
    # ------------------------------------------------------------------
    data_loader_service.save_df_to_snowflake(df, run_id=run_id)
    logger.info("Data successfully saved to Snowflake.")
    logger.info(f"Pipeline execution completed successfully. Run ID: {run_id}")
    logger.info("=" * 100)
 
    # ------------------------------------------------------------------
    # STEP 9 — BUILD RESPONSE SUMMARY
    # ------------------------------------------------------------------
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
 
    for key, val in response.items():
        logger.info(f"  - {key}: {val}")
 
    return response
 
 
# --------------------------------------------------------------------------
# FLASK ROUTES
# --------------------------------------------------------------------------
@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return "Service is healthy", 200
 
 
@app.route("/run", methods=["GET", "POST"])
def run():
    """
    Executes the Ticket Fraud Detection pipeline.
    - GET  → for local debugging (browser or curl)
    - POST → for Snowflake or API service integration
    """
    try:
        # --- Handle GET (local debug) ---
        if request.method == "GET":
            table_name = request.args.get(
                "table_name",
                "UEFA_DEV_DWH.ML_SANDBOX.DS_LOTTERY_AI_DATA_CLEANSING_UECLF24"
            )
            sample_rows = int(request.args.get("sample_rows", 100))
            logger.info(f"[GET] /run → table_name={table_name}, sample_rows={sample_rows}")
 
        # --- Handle POST (Snowflake/Batch) ---
        elif request.method == "POST":
            payload = request.get_json(force=True)
            logger.info(f"[POST] /run received JSON payload: {payload}")
            table_name = payload.get("table_name")
            sample_rows = payload.get("sample_rows", 100)
 
        if not table_name:
            return make_response(jsonify({"error": "Missing 'table_name'"}), 400)
 
        # --- Run pipeline ---
        result_summary = run_tfd_pipeline(table_name, sample_rows)
        return jsonify(result_summary)
 
    except Exception as e:
        logger.exception("Error while executing TFD pipeline")
        return make_response(jsonify({"status": "error", "message": str(e)}), 500)
 
 
# --------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info(f"Starting Ticket Fraud Detection Service on {SERVICE_HOST}:{SERVICE_PORT}")
    app.run(host=SERVICE_HOST, port=SERVICE_PORT)