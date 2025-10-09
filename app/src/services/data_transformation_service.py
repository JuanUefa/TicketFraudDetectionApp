from __future__ import annotations
import logging
import pandas as pd
import numpy as np
 
import logging
 
import numpy as np
import pandas as pd 
logger = logging.getLogger(__name__)
    
from utils.services_utils.data_transformation_utils import DataTransformationUtils

data_transformation_utils = DataTransformationUtils()
 
 
class DataTransformationService:
    """
    Compute identity, behavioral, geolocation, fingerprint, and other fraud features.
    """
    def __init__(self):
        self.username_pairs_df = None
        self.username_user_clusters = None
        self.username_cluster_summary = None
        self.username_cluster_links = None
 


    def email_numerical_representation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Running email_numerical_representation")
    
        # --- Vectorized features ---
        df["username_length"] = df["email_username"].astype(str).str.len()
        df["username_digit_count"] = df["email_username"].astype(str).str.count(r"\d")
        df["username_special_count"] = df["email_username"].astype(str).str.count(r"[^a-zA-Z0-9]")
        df["username_unique_chars"] = df["email_username"].astype(str).apply(lambda x: len(set(x)))
    
        df["username_digit_ratio"] = (df["username_digit_count"] / df["username_length"].replace(0, np.nan)).fillna(0)
        df["username_special_char_ratio"] = (df["username_special_count"] / df["username_length"].replace(0, np.nan)).fillna(0)
    
        # --- Expensive custom functions: compute once per unique username ---
        unique_usernames = df["email_username"].dropna().unique()
        entropy_map = {u: data_transformation_utils.text_entropy(u) for u in unique_usernames}
        kolmogorov_map = {u: data_transformation_utils.kolmogorov_complexity(u) for u in unique_usernames}
        numeric_seq_map = {u: data_transformation_utils.count_numeric_sequences(u) for u in unique_usernames}
    
        df["username_entropy"] = df["email_username"].map(entropy_map)
        df["username_kolmogorov_complexity"] = df["email_username"].map(kolmogorov_map)
        df["username_numeric_seq_count"] = df["email_username"].map(numeric_seq_map)
    
        # --- Inverse semantic score (deduped by combo key) ---
        combo_keys = (
            df[["first_name", "last_name", "email_username"]]
            .fillna("")
            .agg("|".join, axis=1)
        )
        semantic_map = {k: data_transformation_utils.inverse_semantic_score(*k.split("|")) for k in combo_keys.unique()}
        df["inv_semantic_score"] = combo_keys.map(semantic_map)
    
        return df
 
    
    def add_domain_tld_frequencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add frequency counts for email domains and TLDs.
        Requires columns: ['email_domain', 'email_tld'].
        """
        logging.info("Running add_domain_tld_frequencies")
    
        required_cols = {"email_domain", "email_tld"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            logging.warning(f"Skipping frequency features, missing: {missing}")
            return df
    
        # Direct mapping from value_counts (avoids dict conversion)
        df["domain_freq"] = df["email_domain"].map(df["email_domain"].value_counts())
        df["tld_freq"] = df["email_tld"].map(df["email_tld"].value_counts())
    
        return df
    

    
    def email_based_features(self, df: pd.DataFrame, include_name_match: bool = False) -> pd.DataFrame:
        """
        Extracts numerical and semantic features from the email username.
        Depends on `email_username`, `first_name`, and `last_name` columns.
        Vectorized for efficiency.
        """
        
        logging.info("Running email_based_features")
        utils = DataTransformationUtils()
    
        # --- Basic validation ---
        if "email_username" not in df.columns:
            logging.warning("Column 'email_username' missing — skipping email_based_features.")
            return df
    
        # --- Prepare input ---
        usernames = df["email_username"].fillna("").astype(str)
    
        # --- Vectorized feature extraction ---
        df["username_digit_count"] = utils.count_digits(usernames)
        df["username_special_count"] = utils.count_special_chars(usernames)
        df["username_unique_chars"] = utils.count_unique_chars(usernames)
        df["username_digit_ratio"] = utils.digit_ratio(usernames)
        df["username_special_char_ratio"] = utils.special_char_ratio(usernames)
        df["username_entropy"] = utils.text_entropy(usernames)
        df["username_kolmogorov_complexity"] = utils.kolmogorov_complexity(usernames)
        df["username_numeric_seq_count"] = utils.count_numeric_sequences(usernames)
    
        # --- Optional: inverse semantic score ---
        if {"first_name", "last_name"}.issubset(df.columns):
            df["inv_semantic_score"] = df.apply(
                lambda r: utils.inverse_semantic_score(r["first_name"], r["last_name"], r["email_username"]),
                axis=1
            )
        else:
            logging.warning("Columns 'first_name' or 'last_name' missing — skipping inverse_semantic_score.")
    
        # --- Optional: include name-username match flags ---
        if include_name_match and {"first_name", "last_name"}.issubset(df.columns):
            logging.info("Including name match flags in email_based_features")
    
            match_results = df.apply(
                lambda r: utils.match_name_in_username(r["first_name"], r["last_name"], r["email_username"]),
                axis=1
            )
    
            # Expand match results into individual columns
            df = pd.concat([df, match_results.apply(pd.Series)], axis=1)
    
        return df
    
    
    def identity_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Running identity_based_features")
    
        # --- Group by passport_id (one pass instead of 3) ---
        passport_grp = df.groupby("passport_id", observed=True).agg(
            passport_id_counts=("passport_id", "size"),
            name_variation_per_passport=("full_name", "nunique"),
            unique_emails_per_passport=("email", "nunique"),
        ).reset_index()
    
        # Merge results back in one go
        df = df.merge(passport_grp, on="passport_id", how="left")
    
        # Add flag directly (vectorized)
        df["name_variation_flag"] = (df["name_variation_per_passport"] > 1).astype(int)
    
        # --- Group by contact_number separately (needed only once) ---
        phone_counts = df.groupby("contact_number", observed=True)["contact_number"].transform("count")
        df["phone_counts"] = phone_counts
    
        # Drop intermediate
        df = df.drop(columns=["name_variation_per_passport"])
    
        return df
 
    
    def behavioral_fraud_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Running behavioral_fraud_features")
    
        # --- Precompute app timestamp once ---
        df["app_ts"] = (pd.to_datetime(df["app_date"], errors="coerce").astype("int64") // 10**9).astype("int32")
    
        # --- Groupby per passport: count + lag ---
        passport_grp = df.groupby("passport_id", observed=True).agg(
            app_count_per_passport=("passport_id", "size"),
        )
    
        # Shifted timestamps (time since last app)
        df["last_app_ts"] = df.groupby("passport_id", observed=True)["app_ts"].shift(1)
        df["time_since_last_app"] = df["app_ts"] - df["last_app_ts"]
    
        # Rapid submission flag (< 1h)
        df["rapid_submission_flag"] = (df["time_since_last_app"] < 3600).astype("Int64").fillna(0)
    
        # Merge count features back
        df = df.merge(passport_grp, on="passport_id", how="left")
    
        # --- Groupby per IP (one pass) ---
        df["app_count_per_ip"] = df.groupby("ip_short", observed=True)["ip_short"].transform("count")
    
        # --- Total applications submitted across *_app cols ---
        app_cols = [
            c for c in df.columns
            if "_app" in c and "_increase_chances" not in c and ("24" in c or "25" in c)
        ]
        df["total_apps"] = df[app_cols].astype(int).sum(axis=1) if app_cols else 0
    
        # Drop helpers
        df = df.drop(columns=["app_ts", "last_app_ts", "time_since_last_app"])
    
        return df
    
    
    def geolocation_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Running geolocation_based_features")
    
        # --- Ensure string comparison (avoids categorical mismatch errors) ---
        df["country_mismatch"] = (
            df["country"].astype(str) != df["provider_country"].astype(str)
        ).astype("int8")
    
        # --- Multiple cities per postcode ---
        postcode_city_count = df.groupby("postcode", observed=True)["city"].transform("nunique")
        df["postcode_multiple_cities"] = (postcode_city_count > 1).astype("int8")
    
        return df
 
    
    def fingerprint_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Running fingerprint_based_features")
    
        # --- Aggregate once for IP + Browser ---
        agg = (
            df.groupby(["ip_short", "browser"], observed=True)["passport_id"]
            .nunique()
            .reset_index(name="ip_browser_nunique")
        )
    
        # Separate IP-only and Browser-only counts
        ip_shared = df.groupby("ip_short", observed=True)["passport_id"].nunique().rename("ip_shared")
        browser_shared = df.groupby("browser", observed=True)["passport_id"].nunique().rename("browser_shared")
    
        # Map results back to df
        df = df.join(ip_shared, on="ip_short").join(browser_shared, on="browser")
    
        # --- Suspicious flags ---
        df["ip_suspicious"] = (df["ip_shared"] > 2).astype("int8")
        df["browser_suspicious"] = (df["browser_shared"] > 10).astype("int8")
    
        # Drop intermediates
        df = df.drop(columns=["ip_shared", "browser_shared"])
        return df
 
    
    def country_language_mismatch(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Running country_language_mismatch")
    
        # --- Extract all country codes in one pass ---
        extracted = (
            df["browser_language"]
            .str.extractall(r"-([a-zA-Z]{2})")[0]
            .groupby(level=0)  # group by row
            .apply(lambda x: list(x.str.upper().dropna().unique()))
        )
    
        
    
        # Safer: use .map with a fallback to [] instead of fillna([])
        df["browser_lang_countries"] = df.index.map(lambda i: extracted.get(i, []))
    
        # --- Check mismatch ---
        def is_mismatch(row):
            langs = row["browser_lang_countries"]
            return int(
                all(
                    row["provider_country"] != lang and row["country"] != lang
                    for lang in langs
                )
            )
    
        df["country_language_mismatch"] = df.apply(is_mismatch, axis=1)
    
        # Drop helper
        df = df.drop(columns=["browser_lang_countries"])
        return df
 
    
    def uncommon_browser_language(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Running uncommon_browser_language")
    
        # Extract ALL country codes from browser_language (e.g. en-US, fr-FR → ['US', 'FR'])
        extracted = (
            df["browser_language"]
            .str.extractall(r"-([a-zA-Z]{2})")[0]   # extract all 2-letter country codes
            .groupby(level=0)
            .apply(lambda x: list(x.str.upper().dropna().unique()))  # keep unique codes per row
        )
    
        
    
        # Map extracted codes back to each row (default empty list if missing)
        df["browser_lang_countries"] = df.index.map(lambda i: extracted.get(i, []))
    
        # Flag if provider_country is NOT in any of the extracted country codes
        df["uncommon_browser_language"] = df.apply(
            lambda row: int(row["provider_country"] not in row["browser_lang_countries"]),
            axis=1,
        )
    
        # Drop helper column
        df = df.drop(columns=["browser_lang_countries"])
        return df
 
    
    def unusual_local_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag unusual local submission times based on 'app_date' and 'local_timezone'.
        Requires:
        - df['app_date'] (timestamp or string convertible to datetime)
        - df['local_timezone'] (string with valid timezone)
        """
        logging.info("Running unusual_local_time")
    
        
    
        # 1. Convert to UTC datetime once
        df["app_date"] = pd.to_datetime(df["app_date"], errors="coerce", utc=True)
    
        # 2. Create result column upfront
        df["unusual_local_time"] = 0
    
        # 3. Process each timezone in bulk (avoids per-row .apply())
        for tz in df["local_timezone"].dropna().unique():
            try:
                mask = df["local_timezone"] == tz
                local_times = df.loc[mask, "app_date"].dt.tz_convert(tz)
                local_hours = local_times.dt.hour
                df.loc[mask, "unusual_local_time"] = (
                    (local_hours <= 5) | (local_hours == 23)
                ).astype(int)
            except Exception:
                logging.warning(f"Skipping invalid timezone: {tz}")
    
        return df
 
    
    def city_country_mismatch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag mismatches between city → expected country vs. actual country/provider country.
        Requires:
        - df['city'] (string, already normalized/uppercased if needed)
        - df['expected_country_from_city'] (from CITY_COUNTRY_MAP join)
        """
        logging.info("Running city_country_mismatch")
    
        
    
        if "expected_country_from_city" not in df.columns:
            logging.warning("expected_country_from_city column missing, skipping mismatch check")
            df["city_country_mismatch"] = 0
            return df
    
        df["city_country_mismatch"] = (
            (df["expected_country_from_city"].notna())
    & (df["expected_country_from_city"] != df["country"])
    & (df["expected_country_from_city"] != df["provider_country"])
        ).astype(int)
    
        # Drop helper column to keep dataset clean
        df = df.drop(columns=["expected_country_from_city"])
    
        return df
    

    def prune_columns(self, df: pd.DataFrame, keep: list[str] = None) -> pd.DataFrame:
        """
        Drop intermediate columns not needed after transformations.
        If keep is provided, only retain those columns.
        """
        logging.info("Running prune_columns")
    
        # Default: drop known intermediates
        drop_cols = [
            "email_username", "email_domain", "email_tld", 
            "full_name", "expected_country_from_city"
        ]
    
        if keep is not None:
            # keep only specified columns
            df = df[keep]
        else:
            # drop only known intermediate columns
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    
        return df
    
 
    def compute_username_similarity_features(
        self,
        df: pd.DataFrame,
        *,
        visualize: bool = False,
        num_perm: int = 128,
        thresholds: dict | None = None,
        min_refined_similarity: float = 0.70,
        neighbor_threshold: int = 2,
        cluster_edge_min_sim: float = 0.70,
        cluster_lsh_threshold: float = 0.55,
        cluster_link_min_sim: float = 0.65,
        audit_top_k: int = 20,
        output_dir: str = ".",
    ) -> pd.DataFrame:
        """
        Wrapper so the pipeline can `.pipe(...)` this stage and still get back a DataFrame.
        Also stores useful artifacts as attributes for downstream audit/inspection.
        """
        (out_df,
         pairs_df,
         user_clusters,
         cluster_summary,
         cluster_links) = data_transformation_utils.username_similarity_score_optimized(
            df=df,
            visualize=visualize,
            num_perm=num_perm,
            thresholds=thresholds,
            min_refined_similarity=min_refined_similarity,
            neighbor_threshold=neighbor_threshold,
            cluster_edge_min_sim=cluster_edge_min_sim,
            cluster_lsh_threshold=cluster_lsh_threshold,
            cluster_link_min_sim=cluster_link_min_sim,
            audit_top_k=audit_top_k,
            output_dir=output_dir,
        )
 
        # stash artifacts
        self.username_pairs_df = pairs_df
        self.username_user_clusters = user_clusters
        self.username_cluster_summary = cluster_summary
        self.username_cluster_links = cluster_links
 
        logger.info(
            "Username similarity stage: pairs=%d, clusters=%d (+ isolates bucket), links=%d",
            len(pairs_df),
            int(cluster_summary[cluster_summary['cluster_id'] != -1]['cluster_id'].nunique())
            if not cluster_summary.empty else 0,
            len(cluster_links),
        )
        return out_df
    
 
