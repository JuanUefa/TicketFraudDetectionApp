import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
 
 
class FraudScoringUtils:
    """
    Utility class for fraud scoring support functions:
      - normalization
      - boosting rules
      - risk tier assignment
      - exporting reports
      - plotting summaries
    """
 
    def __init__(self):
        pass

    # -------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------
    def normalize_cluster(self, series: pd.Series) -> pd.Series:
        """
        Normalizes cluster label values between 0 and 1.
        """
        if series.max() == 0 or series.nunique() <= 1:
            return pd.Series(0, index=series.index)
        return (series - series.min()) / (series.max() - series.min())
 
    # -------------------------------------------------------------
    # Logical risk boosts
    # -------------------------------------------------------------
    def apply_risk_boosts(self, df: pd.DataFrame, score: pd.Series) -> pd.Series:
        """
        Applies rule-based boosts to fraud scores for specific strong-risk signals.
        """
        boosted = score.copy()
 
        # Example rule boosts
        if "browser_suspicious_cluster" in df.columns:
            boosted[df["browser_suspicious_cluster"] == 1] += 0.3
        if "inv_semantic_score_cluster" in df.columns:
            boosted[df["inv_semantic_score_cluster"] >= 3] += 0.2
 
        return boosted.clip(0, 1)
 
    # -------------------------------------------------------------
    # Risk tiers
    # -------------------------------------------------------------
    @staticmethod
    def assign_risk_tiers(score: pd.Series) -> pd.Series:
        """
        Categorizes numeric fraud scores into qualitative risk tiers.
        """
        bins = [0, 0.3, 0.6, 0.8, 1.0]
        labels = ["Low", "Moderate", "High", "Critical"]
        return pd.cut(score, bins=bins, labels=labels, include_lowest=True)
 
    # -------------------------------------------------------------
    # Export reports
    # -------------------------------------------------------------
    def export_results(self, df: pd.DataFrame, output_dir="data/output/reports") -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "fraud_score_summary.csv")
        df.to_csv(output_path, index=False, encoding="utf-8")
        logging.info(f"[FraudScoringUtils] Exported fraud score report → {output_path}")
        return output_path
 
    def export_tier_summary(self, df: pd.DataFrame, output_dir="data/output/reports") -> str:
        """
        Creates a high-level summary count of applications per fraud risk tier.
        """
        os.makedirs(output_dir, exist_ok=True)
        summary = df["fraud_risk_tier"].value_counts().rename_axis("tier").reset_index(name="count")
        summary = summary.sort_values("tier", ascending=True)
        output_path = os.path.join(output_dir, "fraud_tier_summary.csv")
        summary.to_csv(output_path, index=False, encoding="utf-8")
        logging.info(f"[FraudScoringUtils] Exported fraud tier summary → {output_path}")
        return output_path
 
    # -------------------------------------------------------------
    # Feature importance export + plot
    # -------------------------------------------------------------
    def export_feature_importance(self, df: pd.DataFrame, feature_weights: dict, output_dir="data/output/reports") -> str:
        """
        Computes average weighted contribution for each feature and exports both CSV and bar plot.
        """
        importance_data = []
        for feature, weight in feature_weights.items():
            contrib_col = f"{feature}_risk_contrib"
            if contrib_col in df.columns:
                avg_contrib = df[contrib_col].mean()
                importance_data.append({
                    "feature": feature,
                    "avg_contribution": round(avg_contrib, 3),
                    "weight": round(weight, 3),
                    "weighted_importance": round(avg_contrib * weight, 3)
                })
 
        importance_df = pd.DataFrame(importance_data).sort_values(
            by="weighted_importance", ascending=False
        )
 
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "fraud_feature_importance.csv")
        importance_df.to_csv(csv_path, index=False, encoding="utf-8")
        logging.info(f"[FraudScoringUtils] Exported feature importance report → {csv_path}")
 
        # Plot feature importance
        plot_dir = "data/output/plots"
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.barh(
            importance_df["feature"],
            importance_df["weighted_importance"],
            color="indianred",
            alpha=0.75
        )
        plt.xlabel("Weighted Importance")
        plt.ylabel("Feature")
        plt.title("Fraud Feature Importance (Weighted Contribution)")
        plt.gca().invert_yaxis()
 
        plot_path = os.path.join(plot_dir, "fraud_feature_importance.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        logging.info(f"[FraudScoringUtils] Saved feature importance plot → {plot_path}")
 
        return csv_path
 
    # -------------------------------------------------------------
    # Plot fraud score distribution
    # -------------------------------------------------------------
    def plot_fraud_distribution(self, df: pd.DataFrame, output_dir="data/output/plots") -> str:
        """
        Plots a histogram of fraud scores across all applications.
        """
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.hist(df["fraud_score"], bins=20, color="steelblue", alpha=0.75)
        plt.xlabel("Fraud Score")
        plt.ylabel("Number of Applications")
        plt.title("Fraud Score Distribution")
 
        plot_path = os.path.join(output_dir, "fraud_score_distribution.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
 
        logging.info(f"[FraudScoringUtils] Fraud score distribution plot saved → {plot_path}")
        return plot_path