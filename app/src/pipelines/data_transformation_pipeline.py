# src/pipelines/data_transformation_pipeline.py
from __future__ import annotations
 
import logging
import pandas as pd
from functools import partial
 
from src.services.data_transformation_service import DataTransformationService
 
logger = logging.getLogger(__name__)
data_transformation_service = DataTransformationService()
 
 
class DataTransformationPipeline:
    """
    Orchestrates transformations on the prepared dataset.
    This pipeline should be run after DataPreparationPipeline.
    """
 
    def __init__(self, df: pd.DataFrame):
        self.df = df
 
    def run(
        self,
        *,
        # username similarity tuning knobs (you can override when calling run)
        visualize_username_similarity: bool = True,
        num_perm: int = 128,
        thresholds: dict | None = None,               # e.g. {"raw":0.65,"sep":0.60,"shape":0.85,"deleet":0.65}
        min_refined_similarity: float = 0.70,
        neighbor_threshold: int = 2,
        cluster_edge_min_sim: float = 0.70,
        cluster_lsh_threshold: float = 0.55,
        cluster_link_min_sim: float = 0.65,
        audit_top_k: int = 50,
        output_dir: str = ".",
    ) -> pd.DataFrame:
        logging.info("Starting Data Transformation Pipeline")
 
        df = (
            data_transformation_service.email_numerical_representation(self.df)
            .pipe(data_transformation_service.add_domain_tld_frequencies)
            .pipe(partial(
                data_transformation_service.email_based_features,
                include_name_match=True
            ))
            # --- UPDATED STAGE: optimized username similarity with full param pass-through
            .pipe(partial(
                data_transformation_service.compute_username_similarity_features,
                visualize=visualize_username_similarity,
                num_perm=num_perm,
                thresholds=thresholds,
                min_refined_similarity=min_refined_similarity,
                neighbor_threshold=neighbor_threshold,
                cluster_edge_min_sim=cluster_edge_min_sim,
                cluster_lsh_threshold=cluster_lsh_threshold,
                cluster_link_min_sim=cluster_link_min_sim,
                audit_top_k=audit_top_k,
                output_dir=output_dir,
            ))
            .pipe(data_transformation_service.identity_based_features)
            .pipe(data_transformation_service.behavioral_fraud_features)
            .pipe(data_transformation_service.geolocation_based_features)
            .pipe(data_transformation_service.fingerprint_based_features)
            .pipe(data_transformation_service.country_language_mismatch)
            .pipe(data_transformation_service.uncommon_browser_language)
            .pipe(data_transformation_service.unusual_local_time)
            .pipe(data_transformation_service.city_country_mismatch)
            # .pipe(data_transformation_service.prune_columns)
        )
 
        logging.info("Data Transformation Pipeline finished")
        logging.info(df.columns)
        return df