import logging
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import zscore
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from networkx.algorithms.community import greedy_modularity_communities

from utils.input_output_utils.env_loader import *
 
 
class ClusteringUtils:
    """
    Utility class for clustering-related feature grouping and preprocessing.
    """
 
    def __init__(self):
        pass 

    def group_features(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """
        Partition numeric columns into binary and continuous (numerical) features.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
 
        if not numeric_cols:
            logging.warning("No numeric columns found in dataframe.")
            return [], []
 
        binary_features = [
            col for col in numeric_cols
            if set(df[col].dropna().unique()) == {0, 1}
        ]
 
        numerical_features = [col for col in numeric_cols if col not in binary_features]
 
        logging.info(f"Identified {len(binary_features)} binary features: {binary_features}")
        logging.info(f"Identified {len(numerical_features)} numerical features: {numerical_features}")
 
        return numerical_features, binary_features
 
    def detect_modal_distribution(self, feature: pd.Series, threshold: int = 2) -> str:
        """
        Detects if a numerical feature is unimodal or multimodal.
        Uses histogram-based peak detection to classify the distribution.
 
        Args:
            feature (pd.Series): Feature column to analyze.
            threshold (int): Minimum number of peaks to consider multimodal.
 
        Returns:
            str: 'unimodal' or 'multimodal'
        """
        data = feature.dropna().values
 
        if len(data) < 10:
            return "insufficient_data"
 
        density, bins = np.histogram(data, bins=30, density=True)
        peaks, _ = find_peaks(density, height=np.mean(density) * 0.8)
 
        modality = "multimodal" if len(peaks) >= threshold else "unimodal"
        return modality
 
    def classify_numerical_features(self, df: pd.DataFrame, numerical_features: list[str]) -> dict:
        """
        Classifies numerical features as unimodal or multimodal.
 
        Args:
            df (pd.DataFrame): Input DataFrame
            numerical_features (list[str]): Numeric columns to analyze
 
        Returns:
            dict: Mapping {feature_name: modality_label}
        """
        distribution_types = {}
 
        for feature in numerical_features:
            modality = self.detect_modal_distribution(df[feature])
            distribution_types[feature] = modality
            logging.info(f"Feature '{feature}': {modality}")
 
        return distribution_types
    
    # -------------------------------
    # Unimodal Statistical Clustering
    # -------------------------------
    def unimodal_statistical_clustering(self, x: pd.Series | np.ndarray) -> np.ndarray:
        """
        Simple statistical Z-score based clustering for unimodal features.
 
        Args:
            x (pd.Series or np.ndarray): Numeric feature to cluster.
 
        Returns:
            np.ndarray: Cluster labels (0–3 based on Z thresholds)
        """
        if isinstance(x, np.ndarray):
            x = pd.Series(x.flatten()) if x.ndim > 1 else pd.Series(x)
        elif not isinstance(x, pd.Series):
            raise ValueError("Expected a pandas Series or 1D ndarray")
 
        z = zscore(x.fillna(0))
        labels = np.zeros(len(x), dtype=int)
        labels[z > 0] = 1
        labels[z > 1] = 2
        labels[z > 2] = 3
        return labels
 
    # -------------------------------
    # GMM Model Factory
    # -------------------------------
    def get_clustering_methods(self):
        """Returns clustering methods mapped by distribution type."""
        return {
            "unimodal": {
                "StatisticalZScore": self.unimodal_statistical_clustering
            },
            "multimodal": {
                "GMM": GaussianMixture(n_components=3, covariance_type="full", random_state=42)
            }
        }
 
    # -------------------------------
    # Visualization Helpers
    # -------------------------------
    def plot_statistical_zscore(self, series, feature_name):
        """Plot distribution with Z-score thresholds."""
        z = zscore(series)
        thresholds = {
            "Low (Z < 0)": series[z <= 0].max(),
            "Moderate (0 < Z ≤ 1)": series[(z > 0) & (z <= 1)].max(),
            "High (1 < Z ≤ 2)": series[(z > 1) & (z <= 2)].max(),
            "Very High (Z > 2)": series[z > 2].min(),
        }
 
        plt.figure(figsize=(6, 4))
        sns.histplot(series, kde=True, bins=30, alpha=0.6)
        for label, value in thresholds.items():
            if pd.notna(value):
                plt.axvline(value, linestyle="dashed", linewidth=2, label=label)
 
        plt.xlabel(feature_name)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {feature_name} with Z-Score Thresholds")
        plt.legend()
        plt.tight_layout()
        #
        plt.savefig(PLOTS_PATH + f"Distribution of {feature_name} with Z-Score Thresholds")
 
    def plot_gmm_clusters(self, series, feature_name, gmm_model):
        """Visualize Gaussian Mixture Model clustering."""
        data = series.dropna().values
        means = gmm_model.means_.flatten()
        stds = np.sqrt(gmm_model.covariances_).flatten()
 
        plt.figure(figsize=(6, 4))
        sns.kdeplot(data, bw_adjust=0.5, fill=True, label="KDE")
        plt.scatter(means, np.zeros_like(means), color="red", marker="o", s=100, label="GMM Peaks")
        for mean, std in zip(means, stds):
            plt.errorbar(mean, 0, xerr=std, fmt="o", color="red", capsize=5)
 
        plt.title(f"GMM Estimated Peaks — {feature_name}")
        plt.legend()
        plt.tight_layout()
        #
        plt.savefig(PLOTS_PATH + f"GMM Estimated Peaks — {feature_name}")
 
        for i, (mean, std) in enumerate(zip(means, stds)):
            logging.info(f"{feature_name} | GMM Peak {i+1}: mean={mean:.2f}, std={std:.2f}")
    

        # ---------------------------------------------------------------------
    # Graph-based Binary Clustering Utilities
    # ---------------------------------------------------------------------
    def merge_binary_clusters(self, df: pd.DataFrame, binary_features: list[str]) -> pd.DataFrame:
        """
        Merges similar binary-based clusters using hierarchical clustering on binary centroids.
        """
        if "graph_cluster" not in df.columns:
            raise ValueError("Missing 'graph_cluster' column for merge operation.")
 
        binary_centroids = df.groupby("graph_cluster")[binary_features].mean()
        if binary_centroids.shape[0] < 2:
            return df  # Nothing to merge
 
        distance_matrix = pdist(binary_centroids, metric="euclidean")
        linkage_matrix = sch.linkage(distance_matrix, method="ward")
 
        merge_threshold = 0.3 * max(distance_matrix)  # Tunable parameter
        new_cluster_labels = sch.fcluster(linkage_matrix, merge_threshold, criterion="distance")
 
        cluster_mapping = dict(zip(binary_centroids.index, new_cluster_labels))
        df["graph_cluster"] = df["graph_cluster"].map(cluster_mapping)
 
        return df
 
    def plot_graph(self, df: pd.DataFrame, G: nx.Graph, graph_cluster_col: str):
        """
        Visualizes graph clusters and adds dynamically adjusted cluster labels.
        """
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(6, 4))
 
        nx.draw(
            G,
            pos,
            node_color=df[graph_cluster_col].map(lambda x: x % 10),
            cmap=plt.cm.viridis,
            node_size=50,
            alpha=0.6,
        )
        nx.draw_networkx_edges(G, pos, alpha=0.3)
 
        label_offsets = {}
        for cluster in df[graph_cluster_col].unique():
            cluster_nodes = df[df[graph_cluster_col] == cluster].index
            cluster_center = np.mean([pos[n] for n in cluster_nodes], axis=0)
            x_offset = (cluster % 2) * 0.02 - 0.01
            y_offset = (cluster % 3) * 0.02 - 0.01
            label_offsets[cluster] = (cluster_center[0] + x_offset, cluster_center[1] + y_offset)
 
        for cluster, (x, y) in label_offsets.items():
            plt.text(
                x,
                y,
                f"Cluster {cluster}",
                fontsize=10,
                color="black",
                bbox=dict(facecolor="white", alpha=0.7),
            )
 
        plt.title("Graph-Based Clustering with Adjusted Cluster Labels")
        plt.savefig(PLOTS_PATH + "Graph-Based Clustering with Adjusted Cluster Labels")
 
    def graph_binary_summary(self, df: pd.DataFrame, cluster_col: str, binary_features: list[str]) -> pd.DataFrame:
        """
        Summarizes the percentage of binary features per cluster,
        and includes the number of datapoints per cluster.
        """
        cluster_sizes = df.groupby(cluster_col).size().rename("cluster_size")
        binary_summary = df.groupby(cluster_col)[binary_features].mean() * 100
        binary_summary = binary_summary.round(2)
        summary = pd.concat([cluster_sizes, binary_summary], axis=1)
        return summary
 
    def feature_presence_per_cluster(self, binary_summary: pd.DataFrame, threshold: float = 0.3):
        """
        Identifies binary features significantly present in each cluster.
        """
        binary_features_per_cluster = binary_summary.apply(
            lambda x: list(x[x > (threshold * 100)].index), axis=1
        )
        return binary_features_per_cluster