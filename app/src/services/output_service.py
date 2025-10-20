import os
import math
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
import logging
logging.getLogger("matplotlib.category").setLevel(logging.ERROR)

class OutputService:
    """
    Handles output summaries and plots for clustered numerical features.
    """
 
    def __init__(self, output_dir: str = "data/output"):
        self.output_dir = output_dir
        self.report_dir = os.path.join(output_dir, "reports")
        self.plot_dir = os.path.join(output_dir, "plots")
 
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
 
    # ---------------------------------------------------------------------
    # Cluster Summary
    # ---------------------------------------------------------------------
    def summarize_clustered_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarizes each numerical feature by its corresponding cluster column.
        Produces one clean combined CSV file in tall format:
        [feature, cluster_label, count, mean, std].
        """
        cluster_cols = [col for col in df.columns if col.endswith("_cluster")]
        summary_records = []
    
        for cluster_col in cluster_cols:
            feature = cluster_col.replace("_cluster", "")
            if feature not in df.columns:
                continue
    
            try:
                grouped = (
                    df.groupby(cluster_col)[feature]
                    .agg(["count", "mean", "std"])
                    .round(3)
                    .reset_index()
                )
                grouped["feature"] = feature
                grouped.rename(columns={cluster_col: "cluster_label"}, inplace=True)
                summary_records.append(grouped)
            except Exception as e:
                logging.warning(f"[OutputService] Skipped feature {feature}: {e}")
    
        if not summary_records:
            logging.warning("[OutputService] No cluster summaries generated.")
            return pd.DataFrame()
    
        # Combine and sort
        summary_df = pd.concat(summary_records, ignore_index=True)
        summary_df = summary_df[["feature", "cluster_label", "count", "mean", "std"]]
        summary_df.sort_values(["feature", "cluster_label"], inplace=True)
    
        # Save single CSV
        output_path = os.path.join(self.report_dir, "cluster_summary_all.csv")
        summary_df.to_csv(output_path, index=False)
    
        logging.info(f"[OutputService] Combined cluster summary saved -> {output_path}")
        return summary_df

    # ---------------------------------------------------------------------
    # Cluster Plotting
    # ---------------------------------------------------------------------
    def plot_clustered_numerical_features_grid(
        self,
        df: pd.DataFrame,
        numerical_features: list,
        cluster_suffix: str = "_cluster",
        n_cols: int = 3,
        palette: str = "coolwarm",
        save_plots: bool = True,
        show_plots: bool = False,
    ) -> str:
        """
        Creates a grid of violin plots comparing numerical features vs. clusters.
        Completely suppresses Seaborn/Matplotlib future warnings.
        """
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", module="matplotlib")
    
        num_plots = len(numerical_features)
        if num_plots == 0:
            logging.warning("[OutputService] No numerical features provided for plotting.")
            return ""
    
        n_rows = math.ceil(num_plots / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
        axes = axes.flatten()
    
        for i, feature in enumerate(numerical_features):
            cluster_col = f"{feature}{cluster_suffix}"
    
            if cluster_col in df.columns:
                # Cast both axes to numeric
                df[feature] = pd.to_numeric(df[feature], errors="coerce").astype(float)
                df[cluster_col] = pd.to_numeric(df[cluster_col], errors="coerce").astype(int)
    
                # Explicitly tell Seaborn that hue == x
                sns.violinplot(
                    data=df,
                    x=cluster_col,
                    y=feature,
                    hue=cluster_col,
                    palette=palette,
                    legend=False,
                    ax=axes[i],
                )
    
                axes[i].set_title(f"{feature}", fontsize=11)
                axes[i].set_xlabel("Cluster")
                axes[i].set_ylabel("Value")
            else:
                axes[i].set_visible(False)
    
        # Clean up extra axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
    
        plt.tight_layout()
    
        plot_path = os.path.join(self.plot_dir, "clustered_features_grid.png")
        if save_plots:
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            logging.info(f"[OutputService] Saved combined cluster plot grid -> {plot_path}")
    
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    
        return plot_path