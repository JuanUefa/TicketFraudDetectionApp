# services/data_loading.py

import logging
import pandas as pd

from snowflake.snowpark import Session, DataFrame
from snowflake.snowpark.functions import col, lower, upper, trim

from utils.snowflake_conn import run_query
from utils.query_loader import load_query
from utils.checkpoints import save_checkpoint, load_checkpoint


class DataLoaderService:

    def __init__(self, ):
        pass 
 
    def data_loader(self, query_file) -> DataFrame: #tuple[DataFrame, DataFrame]:
        logging.info(f"START - data_loading")
    
        query = load_query(query_file)
        rows, columns = run_query(query)

        df_raw = pd.DataFrame(rows, columns=columns)
        logging.info(df_raw.shape)
        logging.info(df_raw.head())

        return df_raw