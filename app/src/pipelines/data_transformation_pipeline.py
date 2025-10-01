import logging
import pandas as pd
 
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
            .pipe(data_transformation_service.identity_based_features)
            .pipe(data_transformation_service.behavioral_fraud_features)
            .pipe(data_transformation_service.geolocation_based_features)
            .pipe(data_transformation_service.fingerprint_based_features)
            .pipe(data_transformation_service.country_language_mismatch)
            .pipe(data_transformation_service.uncommon_browser_language)
            .pipe(data_transformation_service.unusual_local_time)
            .pipe(data_transformation_service.city_country_mismatch)
            .pipe(data_transformation_service.prune_columns, keep=['passport_id', 'cleansed'])
        )

        logging.info("Data Transformation Pipeline finished")
        return df