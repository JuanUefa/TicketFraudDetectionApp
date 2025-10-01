# services/data_loading.py

import logging
import pandas as pd

from snowflake.snowpark import Session, DataFrame
from snowflake.snowpark.functions import col, lower, upper, trim

from utils.connection_utils.snowflake_conn import run_query
from utils.query_utils.query_loader import load_query
from utils.input_output_utils.checkpoints import save_checkpoint, load_checkpoint


class DataLoaderService:

    def __init__(self):
        pass 
 
    def data_loader(self, query_file) -> DataFrame: #tuple[DataFrame, DataFrame]:
        logging.info(f"START - data_loading")
    
        query = load_query(query_file)
        rows, columns = run_query(query)

        df_raw = pd.DataFrame(rows, columns=columns)
        logging.info(df_raw.shape)
        logging.info(df_raw.head())

        return df_raw
    

    # --- Specific lookups ---
 
    def load_country_timezone_map(self) -> pd.DataFrame:
        df = self.data_loader("country_timezone_map.sql")
        return df.rename(columns={"ISO2": "country", "TIMEZONE": "local_timezone"})
 
    def load_city_country_map(self) -> pd.DataFrame:
        df = self.data_loader("city_country_map.sql")
        return df.rename(columns={"CLEANED_CITY": "city", "COUNTRY_CODE": "expected_country_from_city"})
    

    def load_country_code_map(self) -> pd.DataFrame:
        df = self.data_loader(query_file="country_code_mapping.sql")
    
        # Standardize column names to lowercase
        df.columns = df.columns.str.lower()
    
        return df