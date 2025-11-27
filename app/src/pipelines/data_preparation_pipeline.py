import logging
import pandas as pd

from src.services.data_loader_service import DataLoaderService
from src.services.data_preparation_service import DataPreparationService

from src.schemas.ueclf24_schema import COLUMNS, RENAMES


 
class DataPreparationPipeline:
    """
    Orchestrates data preparation steps for UECLF24 dataset.
    """
 
    def __init__(self, data_loader_service: DataLoaderService, df: pd.DataFrame):
        self.data_loader_service = data_loader_service

        self.df = df

    def run(self) -> pd.DataFrame:

        data_preparation_service = DataPreparationService(self.data_loader_service, columns_list=COLUMNS, renames_dict=RENAMES)

        logging.info("Starting Data Preparation Pipeline (UECLF24)")

        df_timezone_map = self.data_loader_service.load_country_timezone_map()
        df_city_map = self.data_loader_service.load_city_country_map()
        df_country_iso = self.data_loader_service.load_country_code_map()

        df = (
            data_preparation_service.normalize_columns(self.df)
            .pipe(data_preparation_service.subset_dataframe)
            .pipe(data_preparation_service.rename_vars)
            .pipe(data_preparation_service.drop_duplicates)
            .pipe(
                data_preparation_service.enrich_dataframe,
                df_timezone_map=df_timezone_map,
                df_city_map=df_city_map,
                df_country_iso=df_country_iso
            )
            .pipe(data_preparation_service.missing_values_imputation)
            .pipe(data_preparation_service.variables_cleaning)
            .pipe(data_preparation_service.optimize_dtypes)
        )

        logging.info("Data Preparation Pipeline finished")
        return df
    

