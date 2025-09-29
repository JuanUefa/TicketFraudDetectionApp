import pandas as pd
import logging

from utils.data_cleaning_utils import DataCleaningUtils

data_cleaning_utils = DataCleaningUtils()

 
class DataPreparationService:

    def __init__(self, columns_list, renames_dict, df_country_iso):
        self.columns_list = columns_list
        self.renames_dict = renames_dict

        self.df_country_iso = df_country_iso

 
    def subset_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Subset df to only the relevant UECLF24 columns.
        Missing columns are ignored with a warning.
        """
        existing_cols = [c for c in self.columns_list if c in df.columns]
        missing_cols = set(self.columns_list) - set(existing_cols)
 
        if missing_cols:
            logging.warning(f"[subset_dataframe] Missing columns: {missing_cols}")
 
        return df[existing_cols]

 
    def rename_vars(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename dataframe columns for UECLF24 dataset.
        Also normalize all input columns to lowercase before applying rename rules.
        """
        logging.info("Running rename_vars for DS_LOTTERY_AI_DATA_CLEANSING_UECLF24")
    
        # Normalize all column names to lowercase for consistency
        df.columns = df.columns.str.lower()
    
        # Normalize renames dict keys as well
        valid_renames = {orig.lower(): new for orig, new in self.renames_dict.items() if orig.lower() in df.columns}
        missing = set([orig.lower() for orig in self.renames_dict.keys()]) - set(valid_renames.keys())
    
        if missing:
            logging.warning(f"Missing columns during rename: {missing}")
    
        return df.rename(columns=valid_renames)
    

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
    

    def missing_values_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values and clean date format for app_date column.
        """
        logging.info("Running missing_values_imputation")
    
        # Replace nulls and "0" with default date
        df["app_date"] = df["app_date"].replace("0", "01/01/1970 00:00:00").fillna("01/01/1970 00:00:00")
    
        # Convert to datetime (DD/MM/YYYY or DD/MM/YYYY HH:MM:SS)
        df["app_date"] = pd.to_datetime(
            df["app_date"],
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce"
        ).fillna(pd.to_datetime("1970-01-01 00:00:00"))
    
        return df
    

    def variables_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize variables for UECLF24 dataset.
        """
        logging.info("Running variables_cleaning")
 
        # Application question mappings
        if "uelf_24_app_increase_chances" in df.columns:
            df["uelf_24_app_increase_chances"] = df["uelf_24_app_increase_chances"].apply(
                lambda x: "1" if x == "I accept tickets in another category" else "0"
            )
 
        if "cleansed" in df.columns:
            df["cleansed"] = df["cleansed"].apply(
                lambda x: "1" if x == "CLEANSED" else ("0" if x == "OK" else "0")
            )
 
        if "uelf_24_app" in df.columns:
            df["uelf_24_app"] = df["uelf_24_app"].apply(
                lambda x: "1" if x == "I want to apply for tickets in any case"
                else ("0" if x == "I want to apply for tickets only if my team qualifies" else "0")
            )
 
        # Boolean flags mapping (yes/no → 1/0)
        for flag in ["ueclf_24_app", "uclf_24_app", "uelf_25_app"]:
            if flag in df.columns:
                df[flag] = df[flag].str.lower().map({"yes": "1", "no": "0"}).fillna("0")
 
        # Normalize and clean names
        if "first_name" in df.columns:
            df["first_name"] = df["first_name"].apply(data_cleaning_utils.clean_names)
        if "last_name" in df.columns:
            df["last_name"] = df["last_name"].apply(data_cleaning_utils.clean_names)
 
        # Extract email components
        if "email" in df.columns:
            df[["email_username", "email_domain", "email_tld"]] = (
                df["email"].apply(data_cleaning_utils.extract_email_components).apply(pd.Series)
            )
 
        # Clean browser language
        if "browser_language" in df.columns:
            df["browser_language"] = df["browser_language"].apply(data_cleaning_utils.clean_browser_language)
 
        # Clean and convert country to ISO2
        if "country" in df.columns:
            df["country"] = df["country"].str.upper()
            df = df.merge(
                self.df_country_iso[["country_name", "isocode"]],
                how="left",
                left_on="country",
                right_on="country_name"
            ).drop(columns=["country", "country_name"])
            df = df.rename(columns={"isocode": "country"})
 
        # Clean city
        if "city" in df.columns:
            df["city"] = df["city"].apply(data_cleaning_utils.clean_city)
 
        return df