# services/data_loading.py

import logging
import pandas as pd
from datetime import datetime

from snowflake.snowpark import Session, DataFrame
from snowflake.snowpark.functions import col, lower, upper, trim

from utils.connection_utils.snowflake_conn import run_query, get_snowflake_session
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
    

    def save_df_to_snowflake(
        self,
        df: pd.DataFrame,
        run_id: str,
        schema: str = "ML_SANDBOX",
        database: str = "UEFA_DEV_DWH",
        project: str = "TFD"
    ):
        """
        Saves a pandas DataFrame to Snowflake.
        Ensures type-safe upload (handles object columns like passport_id).
        """
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        table_name = f"{project}_{timestamp}_{run_id}".upper()
    
        logging.info(f"Saving DataFrame to Snowflake: {database}.{schema}.{table_name}")
        logging.info(f"DataFrame shape: {df.shape}")
    
        # --- Type sanity check ---
        for col in df.columns:
            if df[col].dtype == "object":
                logging.info(f"Converting column '{col}' to string for Snowflake compatibility.")
                df[col] = df[col].astype(str)
        # Replace "nan"/"None" literal strings with actual None (Snowflake NULL)
        df = df.replace({"nan": None, "None": None})
    
        session = get_snowflake_session()
    
        try:
            session.write_pandas(
                df,
                table_name=table_name,
                schema=schema,
                database=database,
                auto_create_table=True,
                overwrite=True,
                use_logical_type=False  # proper datetime handling
            )
    
            logging.info(f"Saved successfully to {database}.{schema}.{table_name}")
            return table_name
    
        finally:
            logging.info(f"Closing Snowflake session: {session.session_id}")
            session.close()