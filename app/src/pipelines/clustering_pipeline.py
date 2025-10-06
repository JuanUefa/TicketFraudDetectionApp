import logging
from src.services.clustering_service import ClusteringService
 
 
class ClusteringPipeline:
    """
    Pipeline to orchestrate quantile-based clustering for scaled numerical features.
    """
 
    def __init__(self):
        self.clustering_service = ClusteringService()
 
    def run(self, df):
        logging.info("Starting Clustering Pipeline")
        df = self.clustering_service.quantile_based_clustering(
            df, 
            exclude_columns=["app_id"], 
            num_clusters=5
        )
        logging.info("Clustering Pipeline finished successfully")
        return df