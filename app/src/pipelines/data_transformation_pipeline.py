import logging
import pandas as pd
from functools import partial

from src.services.data_transformation_service import DataTransformationService
 
data_transformation_service = DataTransformationService()

 
class DataTransformationPipeline:
    """
    Orchestrates transformations on the prepared dataset.
    This pipeline should be run after DataPreparationPipeline.
    """
 
    def __init__(self, df: pd.DataFrame):
        self.df = df
 
    def run(self) -> pd.DataFrame:
        logging.info("Starting Data Transformation Pipeline")

        df = (
            data_transformation_service.email_numerical_representation(self.df)
            .pipe(data_transformation_service.add_domain_tld_frequencies)
            .pipe(partial(data_transformation_service.email_based_features, include_name_match=True)) # Lightweight run: include_name_match=False
            .pipe(data_transformation_service.compute_username_similarity_features, visualize=True)            
            .pipe(data_transformation_service.identity_based_features)
            .pipe(data_transformation_service.behavioral_fraud_features)
            .pipe(data_transformation_service.geolocation_based_features)
            .pipe(data_transformation_service.fingerprint_based_features)
            .pipe(data_transformation_service.country_language_mismatch)
            .pipe(data_transformation_service.uncommon_browser_language)
            .pipe(data_transformation_service.unusual_local_time)
            .pipe(data_transformation_service.city_country_mismatch)
            #.pipe(data_transformation_service.prune_columns)
        )

        logging.info("Data Transformation Pipeline finished")
        logging.info(df.columns)
        return df