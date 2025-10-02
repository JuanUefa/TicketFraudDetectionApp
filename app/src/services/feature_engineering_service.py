import logging
import numpy as np
import pandas as pd
from typing import List

 
 
class FeatureEngineeringService:
    """
    Service for feature engineering transformations (generic ML prep).
    """
 
    def __init__(self):
        self.dropped_cols = []

 
    # 1. Drop single-unique columns
    def drop_unique_vars(self, df: pd.DataFrame, protected_cols=None) -> pd.DataFrame:
        """
        Drops columns with only one unique value (excluding protected columns).
        """
        protected_cols = protected_cols or []
    
        # Compute number of unique values per column
        nunique = df.nunique()
    
        # Find columns with only one unique value
        drop_cols = nunique[nunique == 1].index.tolist()
    
        # Exclude protected columns
        drop_cols = [c for c in drop_cols if c not in protected_cols]
    
        if drop_cols:
            logging.info(f"Dropping unique columns: {drop_cols}")
            df = df.drop(columns=drop_cols)
        else:
            logging.info("No unique columns to drop.")
    
        return df
 
    # 2. Detect highly correlated pairs
 
    def detect_highly_correlated_columns(self, df: pd.DataFrame, threshold: float = 0.75):
        """
        Detect highly correlated column pairs in a DataFrame.
    
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        threshold : float
            Absolute correlation threshold above which pairs are flagged.
    
        Returns
        -------
        list of tuples
            Each tuple is (col1, col2, correlation_value).
        """
        # Compute correlation matrix for numeric columns only
        corr = df.corr(numeric_only=True)
        if corr.empty:
            return []
    
        # Absolute values for filtering
        corr_abs = corr.abs()
    
        # Zero out diagonal and lower triangle to avoid duplicates
        corr_abs.values[np.tril_indices_from(corr_abs, k=0)] = 0.0
    
        # Get all pairs above threshold in one vectorized step
        pairs = [
            (corr_abs.index[i], corr_abs.columns[j], float(corr.iat[i, j]))
            for i, j in zip(*np.where(corr_abs > threshold))
        ]
    
        return pairs
 
    # 3. Drop correlated variables
    def drop_least_informative_correlated(
        self, df: pd.DataFrame, correlated_pairs, protected_columns=None
    ):
        """
        Drop the least informative column from correlated pairs based on:
        - Protected columns (never dropped if both are protected).
        - Higher null count (dropped first).
        - Lower variance (dropped if null counts are equal).
    
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        correlated_pairs : list
            Output from detect_highly_correlated_columns ([(col1, col2, corr), ...])
        protected_columns : list
            Columns to never drop (priority kept)
    
        Returns
        -------
        pd.DataFrame
            DataFrame with selected correlated columns dropped.
        """
        if not correlated_pairs:
            logging.info("No correlated pairs to process.")
            return df
    
        protected = set(protected_columns or [])
    
        # --- Precompute statistics once ---
        null_counts = df.isna().sum()
        variances = df.var(numeric_only=True)
    
        to_drop = set()
    
        for col1, col2, _ in correlated_pairs:
            if col1 in protected and col2 in protected:
                continue
            elif col1 in protected:
                to_drop.add(col2)
                continue
            elif col2 in protected:
                to_drop.add(col1)
                continue
    
            # Compare null counts
            n1, n2 = null_counts.get(col1, 0), null_counts.get(col2, 0)
            if n1 != n2:
                to_drop.add(col1 if n1 > n2 else col2)
                continue
    
            # Compare variances if null counts are equal
            v1, v2 = variances.get(col1, np.nan), variances.get(col2, np.nan)
            if pd.notna(v1) and pd.notna(v2):
                to_drop.add(col1 if v1 < v2 else col2)
            else:
                # If both NaN, drop col2 arbitrarily
                to_drop.add(col2)
    
        if to_drop:
            logging.info(f"Dropping {len(to_drop)} correlated columns: {sorted(to_drop)}")
            df = df.drop(columns=list(to_drop), errors="ignore")
    
        return df
 
    # 4. Imbalanced variable detection
    def get_imbalanced_variables(
        self, df: pd.DataFrame, max_classes: int = 5, imbalance_threshold: float = 0.9, protected_cols=None
    ):
        """
        Detect categorical/low-cardinality columns that are highly imbalanced.
    
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        max_classes : int
            Maximum number of distinct values considered for imbalance check
        imbalance_threshold : float
            Threshold above which the column is considered imbalanced
        protected_cols : list, optional
            Columns to exclude from dropping
    
        Returns
        -------
        list
            List of column names flagged as highly imbalanced
        """
        protected = set(protected_cols or [])
        total = len(df)
        if total == 0:
            return []
    
        to_drop = []
    
        for col in df.columns:
            if col in protected:
                continue
    
            nunq = df[col].nunique(dropna=True)
            # Only check low-cardinality categorical-like variables
            if 0 < nunq <= max_classes:
                vc = df[col].value_counts(dropna=True, normalize=True)  # normalized for ratio directly
                if not vc.empty and vc.iloc[0] > imbalance_threshold:
                    to_drop.append(col)
    
        if to_drop:
            logging.info(f"Highly imbalanced variables detected (>{imbalance_threshold*100:.0f}% dominance): {to_drop}")
    
        return to_drop
 
 
    def drop_imbalanced_columns(self, df: pd.DataFrame, cols_to_drop: List[str]) -> pd.DataFrame:
        """
        Drop columns flagged as highly imbalanced.
    
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        cols_to_drop : list
            List of column names to drop
    
        Returns
        -------
        pd.DataFrame
            DataFrame with specified columns removed
        """
        if not cols_to_drop:
            logging.info("No imbalanced columns to drop.")
            return df
    
        existing = [col for col in cols_to_drop if col in df.columns]
        if existing:
            logging.info(f"Dropping {len(existing)} imbalanced columns: {existing}")
            df = df.drop(columns=existing, errors="ignore")
        else:
            logging.info("No matching imbalanced columns found in DataFrame.")
    
        return df
 
    # 5. Skewness reduction
    def reduce_skew_in_dataframe(self, df: pd.DataFrame, skew_threshold: float = 0.8) -> pd.DataFrame:
        """
        Reduce skewness in numeric columns using log1p or sqrt transformations.
    
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        skew_threshold : float, default=0.8
            Minimum absolute skewness (mean-median/std) to trigger transformation.
    
        Returns
        -------
        pd.DataFrame
            DataFrame with skew-reduced numeric columns.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            logging.info("No numeric columns found for skewness reduction.")
            return df
    
        stats = df[numeric_cols].agg(["mean", "median", "std"])
        std = stats.loc["std"].replace(0, np.nan)  # avoid div/0
        skew_approx = (stats.loc["mean"] - stats.loc["median"]) / std
    
        transformed_cols = []
        for col, skew_val in skew_approx.dropna().items():
            if abs(skew_val) > skew_threshold:
                s = df[col]
                if (s > 0).all():
                    df[col] = np.log1p(s)
                    transformed_cols.append((col, "log1p"))
                elif (s >= 0).all():
                    df[col] = np.sqrt(s)
                    transformed_cols.append((col, "sqrt"))
    
        if transformed_cols:
            logging.info(
                "Skewness reduced for columns: "
                + ", ".join([f"{col} ({method})" for col, method in transformed_cols])
            )
        else:
            logging.info("No columns required skewness reduction.")
    
        return df
 
    # 6. Min-Max scaling
    def apply_min_max_scaling(
        self, df: pd.DataFrame, exclude_columns=None, suffix: str = "_scaled"
    ) -> pd.DataFrame:
        """
        Apply Min-Max scaling to numeric columns (per feature scaling to [0, 1]).
        Excludes specified columns and appends new scaled versions.
    
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        exclude_columns : list, optional
            List of column names to exclude from scaling.
        suffix : str, default "_scaled"
            Suffix added to scaled column names.
    
        Returns
        -------
        pd.DataFrame
            DataFrame with new scaled columns added.
        """
        exclude = set(exclude_columns or [])
    
        # Select only numeric columns (excluding protected)
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
        if not num_cols:
            logging.info("No numeric columns available for scaling.")
            return df
    
        # Compute min, max, and range
        mins = df[num_cols].min()
        maxs = df[num_cols].max()
        ranges = (maxs - mins).replace(0, np.nan)  # avoid division by 0
    
        # Perform scaling (broadcasting ensures vectorization)
        scaled = (df[num_cols] - mins) / ranges
    
        # For constant columns (range=0), keep original values
        scaled = scaled.where(ranges.notna(), df[num_cols])
    
        # Rename columns with suffix
        scaled.columns = [f"{c}{suffix}" for c in scaled.columns]
    
        # Log transformation info
        logging.info(
            f"Applied Min-Max scaling to {len(num_cols)} columns. "
            f"Excluded: {sorted(exclude) if exclude else 'None'}"
        )
    
        # Concatenate scaled columns back to df
        return pd.concat([df, scaled], axis=1)