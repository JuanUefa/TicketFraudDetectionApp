import logging
import pandas as pd
from src.services.data_preparation_service import DataPreparationService

from src.schemas.ueclf24_schema import COLUMNS, RENAMES


 
class DataPreparationPipeline:
    """
    Orchestrates data preparation steps for UECLF24 dataset.
    """
 
    def __init__(self, df, df_country_iso):
        self.df = df
        self.df_country_iso = df_country_iso


    def run(self) -> pd.DataFrame:

        data_preparation_service = DataPreparationService(columns_list=COLUMNS, renames_dict=RENAMES,
                                                          df_country_iso=self.df_country_iso)

        logging.info("Starting Data Preparation Pipeline (UECLF24)")

        # Step 2: Subset relevant columns
        df_subset = data_preparation_service.subset_dataframe(self.df)
        logging.info(f"Subset data: {df_subset.shape[0]} rows, {df_subset.shape[1]} columns")

        data_preparation_service.rename_vars(self.df_country_iso)

        df = (
            data_preparation_service.subset_dataframe(self.df)
            .pipe(data_preparation_service.rename_vars)
            .pipe(data_preparation_service.drop_duplicates)
            .pipe(data_preparation_service.missing_values_imputation)
            .pipe(data_preparation_service.variables_cleaning)
        )
 
        logging.info("Data Preparation Pipeline finished")
        return df
    

