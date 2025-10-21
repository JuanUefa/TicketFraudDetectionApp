import logging
import pandas as pd
 
from utils.services_utils.fraud_scoring_utils import FraudScoringUtils
from src.schemas.fraud_score_schemas import RISK_DIRECTION, FEATURE_WEIGHTS

fraud_scoring_utils = FraudScoringUtils()

 
class FraudScoringService:
    """
    Core service orchestrating fraud score computation:
      - computes weighted normalized cluster risk
      - applies logical boosts
      - assigns tiers
      - exports results & plots
      - generates feature importance report
    """
 
    def __init__(self):
        self.weights = FEATURE_WEIGHTS
        self.directions = RISK_DIRECTION
 
    # -------------------------------------------------------------
    # Compute base weighted score
    # -------------------------------------------------------------
    def compute_base_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes a weighted, normalized base score across all cluster variables.
        """
        score = pd.Series(0, index=df.index)
 
        for feature, weight in self.weights.items():
            col = f"{feature}_cluster"
            if col not in df.columns:
                logging.warning(f"[FraudScoringService] Feature {col} missing — skipping")
                continue
 
            normalized = fraud_scoring_utils.normalize_cluster(df[col])
 
            # Flip direction if lower values imply higher risk
            if self.directions[feature] == "low":
                normalized = 1 - normalized
 
            df[f"{feature}_risk_contrib"] = normalized.round(3)
            score += weight * normalized
 
        return score.clip(0, 1)
 
    # -------------------------------------------------------------
    # Full fraud scoring workflow
    # -------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the full fraud scoring workflow.
        """
        logging.info("Starting fraud score computation pipeline...")
        df = df.copy()
 
        # Step 1: Compute base score
        base_score = self.compute_base_score(df)
 
        # Step 2: Apply logical risk boosts
        boosted_score = fraud_scoring_utils.apply_risk_boosts(df, base_score)
 
        # Step 3: Assign risk tiers
        df["fraud_score"] = boosted_score.round(3)
        df["fraud_risk_tier"] = fraud_scoring_utils.assign_risk_tiers(boosted_score)
 
        # Step 4: Export outputs
        output_cols = (
            ["app_id", "fraud_score", "fraud_risk_tier"]
            + [c for c in df.columns if c.endswith("_cluster") or c.endswith("_risk_contrib")]
        )
 
        fraud_scoring_utils.export_results(df[output_cols])
        fraud_scoring_utils.export_tier_summary(df)
        fraud_scoring_utils.plot_fraud_distribution(df)
        fraud_scoring_utils.export_feature_importance(df, self.weights)
 
        logging.info("Fraud scoring completed successfully and reports exported.")
        logging.info(f"Fraud risk tier counts:\n{df['fraud_risk_tier'].value_counts()}")
 
        return df