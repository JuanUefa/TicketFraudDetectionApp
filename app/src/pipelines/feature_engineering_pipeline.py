import logging
import pandas as pd

from src.services.feature_engineering_service import FeatureEngineeringService
 
 
class FeatureEngineeringPipeline:
    """
    Pipeline to orchestrate generic feature engineering steps.
    Mirrors the Snowflake POC but optimized for pandas.
    """
 
    def __init__(self,
                 protected_columns=None,
                 exclude_columns=None,
                 corr_threshold=0.75,
                 imbalance_threshold=0.9,
                 max_classes=5):
        self.service = FeatureEngineeringService()
        self.protected_columns = protected_columns or [
            "app_id", "username_entropy", "inv_semantic_score"
        ]
        self.exclude_columns = exclude_columns or ["app_id"]
        self.corr_threshold = corr_threshold
        self.imbalance_threshold = imbalance_threshold
        self.max_classes = max_classes
 
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting Feature Engineering Pipeline")
 
        # --- Step 1: Drop unique vars ---
        df = df.pipe(self.service.drop_unique_vars,
                     protected_cols=self.protected_columns)
 
        # --- Step 2: Correlation pruning ---
        pairs = self.service.detect_highly_correlated_columns(
            df, threshold=self.corr_threshold
        )
        df = df.pipe(
            self.service.drop_least_informative_correlated,
            correlated_pairs=pairs,
            protected_columns=self.protected_columns,
        )
 
        # --- Step 3: Drop imbalanced vars ---
        imbalanced = self.service.get_imbalanced_variables(
            df,
            max_classes=self.max_classes,
            imbalance_threshold=self.imbalance_threshold,
            protected_cols=self.protected_columns,
        )
        df = df.pipe(self.service.drop_imbalanced_columns, cols_to_drop=imbalanced)
 
        # --- Step 4: Skewness reduction ---
        df = df.pipe(self.service.reduce_skew_in_dataframe)
 
        # --- Step 5: Min-Max scaling (exclude IDs) ---
        df = df.pipe(self.service.apply_min_max_scaling,
                     exclude_columns=self.exclude_columns)
 
        logging.info("Feature Engineering Pipeline finished")
        return df