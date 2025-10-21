import logging
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
 
from utils.services_utils.clustering_utils import ClusteringUtils
 
clustering_utils = ClusteringUtils()
 
 
class ClusteringService:
    """
    Service for quantile-based clustering and adaptive modality-based clustering,
    with logical label ordering (lowest → highest).
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
            c for c in df.columns if c.endswith("_scaled") and c not in exclude
        ]
 
        logging.info(f"Running quantile-based clustering on {len(scaled_cols)} scaled columns.")
 
        for col in scaled_cols:
            try:
                series = pd.to_numeric(df[col], errors="coerce")
 
                # Drop NaN and check unique values
                unique_vals = series.dropna().unique()
                n_unique = len(unique_vals)
 
                if n_unique <= 1:
                    df[f"{col.replace('_scaled', '_cluster')}"] = 1
                    logging.warning(f"{col}: constant values — assigned to 1 cluster")
                    continue
 
                elif n_unique == 2:
                    min_val, max_val = np.sort(unique_vals)
                    df[f"{col.replace('_scaled', '_cluster')}"] = np.where(
                        series <= min_val, 1, 2
                    )
                    logging.info(f"{col}: binary fallback clustering (2 clusters)")
                    continue
 
                # Regular quantile clustering
                quantiles = np.linspace(0, 1, num_clusters + 1)
                bins = np.unique(series.quantile(quantiles).values)
 
                if len(bins) <= 2:
                    bins = np.linspace(series.min(), series.max(), num_clusters + 1)
                    logging.warning(f"{col}: low variance — switched to equal-width clustering")
 
                clusters = pd.cut(
                    series,
                    bins=bins,
                    labels=range(1, len(bins)),
                    include_lowest=True
                ).astype("Int64")
 
                df[f"{col.replace('_scaled', '_cluster')}"] = clusters
 
                logging.info(
                    f"Clustered {col}: actual {len(bins) - 1} bins "
                    f"(method={'quantile' if len(bins) > 2 else 'fallback'})"
                )
 
            except Exception as e:
                logging.error(f"Error clustering {col}: {str(e)}")
 
        logging.info(f"Quantile-based clustering completed for {len(scaled_cols)} variables.")
        return df
 
    def reorder_cluster_labels(self, series: pd.Series, labels: np.ndarray) -> np.ndarray:
        """
        Reorder cluster labels logically (ascending by mean feature value).
        """
        df_temp = pd.DataFrame({"value": series, "label": labels})
        cluster_means = df_temp.groupby("label")["value"].mean().sort_values().index.tolist()
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_means)}
 
        logging.debug(f"Cluster ordering mapping for feature: {label_mapping}")
        return np.array([label_mapping.get(lbl, lbl) for lbl in labels])
 
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
 
                if method_name == "StatisticalZScore":
                    labels = model(series_scaled)
                elif method_name == "GMM":
                    labels = model.fit_predict(X)
                else:
                    raise ValueError(f"Unsupported clustering method: {method_name}")
 
                ordered_labels = self.reorder_cluster_labels(series_scaled, labels)
                df[f"{feature}_cluster"] = ordered_labels
 
                if visualize:
                    if method_name == "StatisticalZScore":
                        clustering_utils.plot_statistical_zscore(series_scaled, feature)
                    elif method_name == "GMM":
                        clustering_utils.plot_gmm_clusters(series_scaled, feature, model)
 
                logging.info(
                    f"{feature}: {len(np.unique(ordered_labels))} clusters, ordered by mean value"
                )
 
            except Exception as e:
                logging.error(f"Clustering failed for {feature}: {e}")
                df[f"{feature}_cluster"] = np.nan
 
        return df
    

    def graph_based_binary_clustering(self, df: pd.DataFrame, binary_features: list[str], merge_clusters: bool = True, visualize: bool = True) -> pd.DataFrame:
        """
        Performs graph-based clustering based on shared binary feature activations.
        Nodes = rows; edges connect samples sharing binary '1' features.
        """
        df = df.copy()
        G = nx.Graph()
 
        # Add nodes
        for idx in df.index:
            G.add_node(idx)
 
        # Add edges between nodes sharing the same binary '1' flag
        for col in binary_features:
            similar_apps = df[df[col] == 1].index
            for i in similar_apps:
                for j in similar_apps:
                    if i != j:
                        G.add_edge(i, j)
 
        # Detect communities
        communities = list(greedy_modularity_communities(G))
        df["graph_cluster"] = 0
        for cluster_id, nodes in enumerate(communities):
            df.loc[list(nodes), "graph_cluster"] = cluster_id
 
        logging.info(f"Detected {len(communities)} initial graph-based clusters")
 
        # Optionally merge clusters with similar binary centroids
        if merge_clusters and df["graph_cluster"].nunique() >= 5:
            df = clustering_utils.merge_binary_clusters(df, binary_features)
            logging.info("Merged similar graph-based clusters using centroid similarity")
 
        # Optional visualization
        logging.info(f"Visualization = {visualize}")
        if visualize:
            clustering_utils.plot_graph(df, G, "graph_cluster")
 
        # Summaries
        summary = clustering_utils.graph_binary_summary(df, "graph_cluster", binary_features)
        tags = clustering_utils.feature_presence_per_cluster(summary)
        logging.info(f"Binary feature summary:\n{summary}")
        logging.info(f"Feature tags per cluster:\n{tags}")
 
        return df