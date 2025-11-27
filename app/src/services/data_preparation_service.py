import pandas as pd
import numpy as np
import logging

from utils.services_utils.data_cleaning_utils import DataCleaningUtils
from src.services.data_loader_service import DataLoaderService

from src.schemas.cleaning_rules import CATEGORICAL_MAPS, BINARY_FLAGS, SPECIAL_CASES   

data_cleaning_utils = DataCleaningUtils()
 
class DataPreparationService:

    def __init__(self, data_loader_service: DataLoaderService, columns_list, renames_dict):
        self.data_loader_service = data_loader_service

        self.columns_list = columns_list
        # normalize keys here once
        self.renames_dict = {k.lower(): v for k, v in renames_dict.items()}

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.strip().str.lower()

        return df

 
    def subset_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Subset df to only the relevant UECLF24 columns.
        Missing columns are ignored with a warning.
        """

        self.columns_list = [col.strip().lower() for col in self.columns_list]
        existing_cols = [c for c in self.columns_list if c in df.columns]
        logging.info(f"Existing columns: {existing_cols}")
        missing_cols = set(self.columns_list) - set(existing_cols)
 
        if missing_cols:
            logging.warning(f"[subset_dataframe] Missing columns: {missing_cols}")

        df = df[existing_cols].copy()

 
        return df
    
    


 
    def rename_vars(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Running rename_vars")
    
        valid_renames = {k: v for k, v in self.renames_dict.items() if k in df.columns}
        missing = set(self.renames_dict.keys()) - set(valid_renames.keys())
    
        if missing:
            logging.warning(f"Missing columns during rename: {missing}")

        df = df.rename(columns=valid_renames)
    
        return df
    

    def drop_duplicates(self, df: pd.DataFrame, subset=None, keep="first") -> pd.DataFrame:
        """
        Drop duplicate rows from the dataframe.
 
        :param df: Input pandas DataFrame
        :param subset: Columns to consider for identifying duplicates (default: all columns)
        :param keep: Which duplicate to keep - 'first', 'last', or False (drop all)
        :return: DataFrame with duplicates removed
        """
        before = df.shape[0]
        df_clean = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        after = df_clean.shape[0]
 
        logging.info(f"Dropped {before - after} duplicate rows (kept {after})")

        return df_clean
    
    
    def enrich_dataframe(self, df: pd.DataFrame, df_timezone_map: pd.DataFrame, df_city_map: pd.DataFrame,
                         df_country_iso: pd.DataFrame) -> pd.DataFrame:

        df = df.merge(df_timezone_map, on="country", how="left")
        df = df.merge(df_city_map, on="city", how="left")

        df = (
        df.merge(
            df_country_iso[["country_name", "isocode"]],
            how="left",
            left_on="country",
            right_on="country_name",
            )
        )

        return df
    

    def missing_values_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values and clean date format for app_date column.
        Vectorized implementation with robust parsing.
        """
        logging.info("Running missing_values_imputation")
    
        default_date = pd.Timestamp("1970-01-01 00:00:00")
    
        if "app_date" in df.columns:
            # Replace invalid placeholders ("0", None, NaN) with NaT
            df["app_date"] = df["app_date"].replace(["0", "", None], np.nan)
    
            # Convert all valid entries to datetime (DD/MM/YYYY or DD/MM/YYYY HH:MM:SS)
            df["app_date"] = pd.to_datetime(
                df["app_date"],
                format="%d/%m/%Y %H:%M:%S",
                errors="coerce"
            )
    
            # Fill any remaining NaT with default date
            df["app_date"] = df["app_date"].fillna(default_date)
    
        return df
    
     
    def variables_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize variables for UECLF24 dataset.
        Uses externalized mappings for maintainability.
        """
        logging.info("Running variables_cleaning")
    
        # --- Apply categorical maps ---
        for col, mapping in CATEGORICAL_MAPS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna("0")
    
        # --- Apply special exact match rules ---
        for col, match_value in SPECIAL_CASES.items():
            if col in df.columns:
                df[col] = (df[col] == match_value).astype(int).astype(str)
    
        # --- Apply yes/no binary flag mapping ---
        for flag in BINARY_FLAGS:
            if flag in df.columns:
                df[flag] = df[flag].str.lower().map({"yes": "1", "no": "0"}).fillna("0")
    
        # --- Normalize and clean names (row-wise, still custom utils) ---
        if "first_name" in df.columns:
            df["first_name"] = df["first_name"].map(data_cleaning_utils.clean_names)
        if "last_name" in df.columns:
            df["last_name"] = df["last_name"].map(data_cleaning_utils.clean_names)
    
        # --- Extract email components ---
        if "email" in df.columns:
            email_components = df["email"].map(data_cleaning_utils.extract_email_components)
            df[["email_username", "email_domain", "email_tld"]] = pd.DataFrame(
                email_components.tolist(), index=df.index
            )
    
        # --- Clean browser language ---
        if "browser_language" in df.columns:
            df["browser_language"] = df["browser_language"].map(data_cleaning_utils.clean_browser_language)
    
        # --- Clean city ---
        if "city" in df.columns:
            df["city"] = df["city"].map(data_cleaning_utils.clean_city)
    
        return df
        

    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert high-cardinality string columns into category dtype
        to save memory and speed up groupby/joins.
        """
        logging.info("Running optimize_dtypes")
    
        categorical_candidates = [
            "passport_id", "contact_number", "browser", 
            "country", "provider_country", "city", "postcode"
        ]
    
        for col in categorical_candidates:
            if col in df.columns and df[col].dtype == "object":
                df[col] = df[col].astype("category")
    
        return df