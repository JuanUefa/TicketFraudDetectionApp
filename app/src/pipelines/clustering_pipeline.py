import logging

from utils.services_utils.clustering_utils import ClusteringUtils
from src.services.clustering_service import ClusteringService
 
 
class ClusteringPipeline:
    """
    Pipeline to orchestrate quantile-based clustering for scaled numerical features.
    """
 
    def __init__(self):
        self.clustering_utils = ClusteringUtils()
        self.clustering_service = ClusteringService()
 
    def run(self, df):
        logging.info("Starting Clustering Pipeline")
        df = self.clustering_service.quantile_based_clustering(
            df, 
            exclude_columns=["app_id"], 
            num_clusters=5
        )
        logging.info("Clustering Pipeline finished successfully")

        numerical_features, binary_features = self.clustering_utils.group_features(df)

        distribution_types = self.clustering_utils.classify_numerical_features(df, numerical_features)
        logging.info(distribution_types)
        
        df_clustered = self.clustering_service.cluster_features_by_modality(
            df, numerical_features, distribution_types, visualize=True
        )

        logging.info(df_clustered)

        return df