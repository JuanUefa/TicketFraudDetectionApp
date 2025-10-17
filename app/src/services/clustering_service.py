import logging
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler

from utils.services_utils.clustering_utils import ClusteringUtils

clustering_utils = ClusteringUtils()
 
class ClusteringService:
    """
    Service for quantile-based clustering with fallback for low-variance columns.
    """
 
    def quantile_based_clustering(
        self,
        df: pd.DataFrame,
        exclude_columns=None,
        num_clusters: int = 5,
    ) -> pd.DataFrame:
        logging.info(f"Running Quantile-Based Clustering with {num_clusters} clusters")
 
        exclude = set((exclude_columns or []))
        scaled_cols = [
            c for c in df.columns
            if c.endswith("_scaled") and c not in exclude
        ]
 
        logging.info(f"Running quantile-based clustering on {len(scaled_cols)} scaled columns.")
 
        for col in scaled_cols:
            try:
                series = pd.to_numeric(df[col], errors="coerce")
 
                # Drop NaN and check unique values
                unique_vals = series.dropna().unique()
                n_unique = len(unique_vals)
 
                if n_unique <= 1:
                    # No variance → one cluster
                    df[f"{col.replace('_scaled', '_cluster')}"] = 1
                    logging.warning(f"{col}: constant values — assigned to 1 cluster")
                    continue
 
                elif n_unique == 2:
                    # Binary fallback → 2 clusters directly
                    min_val, max_val = np.sort(unique_vals)
                    df[f"{col.replace('_scaled', '_cluster')}"] = np.where(
                        series <= min_val, 1, 2
                    )
                    logging.info(f"{col}: binary fallback clustering (2 clusters)")
                    continue
 
                # Regular quantile clustering for rich columns
                quantiles = np.linspace(0, 1, num_clusters + 1)
                bins = np.unique(series.quantile(quantiles).values)
 
                if len(bins) <= 2:
                    # Low-variance fallback → equal-width bins
                    bins = np.linspace(series.min(), series.max(), num_clusters + 1)
                    logging.warning(f"{col}: low variance — switched to equal-width clustering")
 
                clusters = pd.cut(
                    series,
                    bins=bins,
                    labels=range(1, len(bins)),
                    include_lowest=True
                ).astype("Int64")
 
                # Add cluster column (keeping scaled column for now)
                df[f"{col.replace('_scaled', '_cluster')}"] = clusters
 
                logging.info(
                    f"Clustered {col}: actual {len(bins) - 1} bins (method={'quantile' if len(bins) > 2 else 'fallback'})"
                )
 
            except Exception as e:
                logging.error(f"Error clustering {col}: {str(e)}")
 
        logging.info(f"Quantile-based clustering completed for {len(scaled_cols)} variables.")
        return df
    

    def cluster_features_by_modality(
        self,
        df: pd.DataFrame,
        numerical_features: list[str],
        distribution_types: dict,
        visualize: bool = False,
    ) -> pd.DataFrame:
        df = df.copy()
        scaler = StandardScaler()
        clustering_methods = clustering_utils.get_clustering_methods()
 
        for feature in numerical_features:
            modality = distribution_types.get(feature, "unimodal")
            methods = clustering_methods.get(modality, {})
            if not methods:
                logging.warning(f"No clustering method found for {feature} ({modality})")
                continue
 
            method_name, model = next(iter(methods.items()))
            logging.info(f"Clustering '{feature}' using {method_name} ({modality})")
 
            try:
                X = scaler.fit_transform(df[[feature]])
                series_scaled = pd.Series(X.flatten(), name=feature)
 
                # --- Unimodal ---
                if method_name == "StatisticalZScore":
                    labels = model(series_scaled)
 
                # --- Multimodal (GMM) ---
                elif method_name == "GMM":
                    labels = model.fit_predict(X)
 
                else:
                    raise ValueError(f"Unsupported clustering method: {method_name}")
 
                df[f"{feature}_cluster"] = labels
 
                if visualize:
                    if method_name == "StatisticalZScore":
                        clustering_utils.plot_statistical_zscore(series_scaled, feature)
                    elif method_name == "GMM":
                        clustering_utils.plot_gmm_clusters(series_scaled, feature, model)
 
            except Exception as e:
                logging.error(f"Clustering failed for {feature}: {e}")
                df[f"{feature}_cluster"] = np.nan
 
        return df