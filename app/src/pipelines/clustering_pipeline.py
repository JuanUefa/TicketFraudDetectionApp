import logging
import pandas as pd
import numpy as np
from utils.services_utils.clustering_utils import ClusteringUtils
from src.services.clustering_service import ClusteringService
 
 
class ClusteringPipeline:
    """
    Pipeline to orchestrate quantile-based and modality-based clustering,
    with cluster label refactoring and auditing.
    """
 
    def __init__(self):
        self.clustering_utils = ClusteringUtils()
        self.clustering_service = ClusteringService()
 
    # ---------------------------------------------------------------------
    # Helper 1: Log cluster summaries (mean per cluster)
    # ---------------------------------------------------------------------
    def log_cluster_summary(self, df: pd.DataFrame, stage: str):
        logging.info(f"\n========== {stage.upper()} CLUSTERING SUMMARY ==========")
        cluster_columns = [
            col for col in df.columns if "_cluster" in col and "binary_cluster" not in col
        ]
 
        for cluster_col in cluster_columns:
            feature = cluster_col.replace("_cluster", "")
            if feature not in df.columns:
                continue
 
            try:
                summary = df.groupby(cluster_col)[feature].mean().round(3)
                logging.info(f"\nFeature: {feature}\n{summary.to_string()}")
            except Exception as e:
                logging.warning(f"Could not summarize {feature}: {e}")
 
        logging.info("========================================================")
 
    # ---------------------------------------------------------------------
    # Helper 2: Refactor cluster labels
    # ---------------------------------------------------------------------
    def cluster_refactoring(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cluster_columns = [
            col for col in df.columns if "_cluster" in col and "binary_cluster" not in col
        ]
 
        for cluster_col in cluster_columns:
            feature = cluster_col.replace("_cluster", "")
            if feature not in df.columns:
                continue
 
            try:
                series = pd.to_numeric(df[feature], errors="coerce")
                labels = df[cluster_col].values
                reordered = self.clustering_service.reorder_cluster_labels(series, labels)
                df[cluster_col] = reordered
                logging.info(f"Refactored cluster labels for {feature}")
            except Exception as e:
                logging.warning(f"Failed to refactor {feature}: {e}")
 
        return df
 
    # ---------------------------------------------------------------------
    # Main pipeline
    # ---------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting Clustering Pipeline")
 
        # Step 1: Quantile-based clustering
        df = self.clustering_service.quantile_based_clustering(
            df, exclude_columns=["app_id"], num_clusters=5
        )
 
        # Step 2: Group and classify numerical features
        numerical_features, binary_features = self.clustering_utils.group_features(df)
        distribution_types = self.clustering_utils.classify_numerical_features(df, numerical_features)
        logging.info(f"Distribution types: {distribution_types}")
 
        # Step 3: Modality-based clustering
        df = self.clustering_service.cluster_features_by_modality(
            df, numerical_features, distribution_types, visualize=True
        )
 
        # Step 4: Log summary before refactoring
        self.log_cluster_summary(df, "Before")
 
        # Step 5: Refactor all clusters logically
        df = self.cluster_refactoring(df)
 
        # Step 6: Log summary after refactoring
        self.log_cluster_summary(df, "After")

        # Step 7: Graph-based clustering using binary flags
        if binary_features:
            logging.info("Running Graph-Based Binary Clustering")
            df = self.clustering_service.graph_based_binary_clustering(
                df, binary_features, merge_clusters=True, visualize=False
            )
        logging.info("Graph-based clustering step completed successfully")
 
        logging.info("Clustering Pipeline completed successfully")
        return df