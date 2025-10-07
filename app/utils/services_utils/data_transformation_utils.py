# utils/services_utils/data_transformation_utils.py
import re
import zlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from difflib import SequenceMatcher
from nltk.util import ngrams
from scipy.stats import entropy
from functools import lru_cache
from datasketch import MinHash, MinHashLSH
import textdistance
 
 
class DataTransformationUtils:
    """
    Optimized utility functions for feature extraction.
    Supports both scalar and vectorized (Pandas Series) inputs automatically.
    """
 
    def __init__(self):
        pass
 
    # --- Internal helpers ---
    def _ensure_series(self, x):
        """Ensures a Pandas Series, normalizes NaNs."""
        if isinstance(x, pd.Series):
            return x.fillna("").astype(str)
        return pd.Series([x]).fillna("").astype(str)
 
    def _is_scalar(self, x):
        """Detects whether the input is scalar or Series."""
        return not isinstance(x, (pd.Series, list, np.ndarray))
 
    # --- Core functions (vectorized + scalar support) ---
    def count_digits(self, text):
        if self._is_scalar(text):
            if not text:
                return 0
            return sum(c.isdigit() for c in str(text))
 
        s = self._ensure_series(text)
        return s.str.count(r"\d")
 
    def count_special_chars(self, text):
        if self._is_scalar(text):
            if not text:
                return 0
            return len(re.findall(r"[^a-zA-Z0-9]", str(text)))
 
        s = self._ensure_series(text)
        return s.str.count(r"[^a-zA-Z0-9]")
 
    def count_unique_chars(self, text):
        if self._is_scalar(text):
            return len(set(str(text))) if text else 0
 
        s = self._ensure_series(text)
        return s.apply(lambda x: len(set(x)) if x else 0)
 
    def digit_ratio(self, text):
        if self._is_scalar(text):
            if not text:
                return 0.0
            text = str(text)
            return sum(c.isdigit() for c in text) / len(text)
 
        s = self._ensure_series(text)
        counts = s.str.count(r"\d")
        lengths = s.str.len().replace(0, np.nan)
        return (counts / lengths).fillna(0)
 
    def special_char_ratio(self, text):
        if self._is_scalar(text):
            if not text:
                return 0.0
            text = str(text)
            return sum(not c.isalnum() for c in text) / len(text)
 
        s = self._ensure_series(text)
        counts = s.str.count(r"[^a-zA-Z0-9]")
        lengths = s.str.len().replace(0, np.nan)
        return (counts / lengths).fillna(0)
 
    @lru_cache(maxsize=10000)
    def _kolmogorov_cached(self, text: str) -> float:
        if not text:
            return 0.0
        try:
            return len(zlib.compress(text.encode())) / len(text)
        except Exception:
            return 0.0
 
    def kolmogorov_complexity(self, text):
        if self._is_scalar(text):
            return self._kolmogorov_cached(str(text))
 
        s = self._ensure_series(text)
        return s.apply(self._kolmogorov_cached)
 
    def text_entropy(self, text):
        if self._is_scalar(text):
            if not text:
                return 0.0
            text = str(text)
            char_counts = np.array([text.count(c) for c in set(text)])
            return entropy(char_counts) / len(text) if len(text) > 0 else 0.0
 
        s = self._ensure_series(text)
 
        def _entropy_calc(txt):
            if not txt:
                return 0.0
            counts = np.fromiter((txt.count(c) for c in set(txt)), dtype=float)
            return entropy(counts) / len(txt)
 
        return s.apply(_entropy_calc)
 
    def count_numeric_sequences(self, text):
        if self._is_scalar(text):
            if not text or not isinstance(text, str):
                return 0
            start = 1 if re.match(r"^\d+", text) else 0
            u_mid = re.sub(r"\d+$", "", text)
            mids = len(re.findall(r"\d+", u_mid))
            return start + mids
 
        s = self._ensure_series(text)
 
        def _count_seq(u):
            if not u:
                return 0
            start = 1 if re.match(r"^\d+", u) else 0
            u_mid = re.sub(r"\d+$", "", u)
            mids = len(re.findall(r"\d+", u_mid))
            return start + mids
 
        return s.apply(_count_seq)
 
    # --- Name matching (same logic as before) ---
    def _generate_ngrams(self, text: str, n: int = 3):
        text = re.sub(r"[^a-zA-Z]", "", str(text).lower())
        return {"".join(g) for g in ngrams(text, n)}
 
    def _fuzzy_match(self, name: str, username: str, threshold: float = 0.8) -> bool:
        if not name or not username:
            return False
        for i in range(len(username) - len(name) + 1):
            substring = username[i : i + len(name)]
            similarity = SequenceMatcher(None, name, substring).ratio()
            if similarity >= threshold:
                return True
        return False
 
    def match_name_in_username(self, first_name: str, last_name: str, email_username: str) -> dict:
        try:
            username = str(email_username).lower()
            first = str(first_name).lower()
            last = str(last_name).lower()
 
            first_exact = first in username
            last_exact = last in username
 
            username_ngrams = self._generate_ngrams(username)
            first_ngrams = self._generate_ngrams(first)
            last_ngrams = self._generate_ngrams(last)
 
            first_ngram = not username_ngrams.isdisjoint(first_ngrams)
            last_ngram = not username_ngrams.isdisjoint(last_ngrams)
 
            first_fuzzy = self._fuzzy_match(first, username)
            last_fuzzy = self._fuzzy_match(last, username)
 
            return {
                "first_name_exact_match": first_exact,
                "last_name_exact_match": last_exact,
                "first_name_ngram_match": first_ngram,
                "last_name_ngram_match": last_ngram,
                "first_name_fuzzy_match": first_fuzzy,
                "last_name_fuzzy_match": last_fuzzy,
            }
        except Exception:
            return {
                "first_name_exact_match": False,
                "last_name_exact_match": False,
                "first_name_ngram_match": False,
                "last_name_ngram_match": False,
                "first_name_fuzzy_match": False,
                "last_name_fuzzy_match": False,
            }
 
    def inverse_semantic_score(self, first_name: str, last_name: str, email_username: str) -> int:
        try:
            username = str(email_username).lower()
            first = str(first_name).lower()
            last = str(last_name).lower()
 
            exact1 = first in username
            exact2 = last in username
 
            grams_user = self._generate_ngrams(username)
            grams_first = self._generate_ngrams(first)
            grams_last = self._generate_ngrams(last)
 
            ngram1 = not grams_user.isdisjoint(grams_first)
            ngram2 = not grams_user.isdisjoint(grams_last)
 
            fuzzy1 = self._fuzzy_match(first, username)
            fuzzy2 = self._fuzzy_match(last, username)
 
            match_flags = [exact1, exact2, ngram1, ngram2, fuzzy1, fuzzy2]
            return len(match_flags) - sum(match_flags)
 
        except Exception:
            return 6
        

    def username_similarity_score(
        self,
        df: pd.DataFrame,
        visualize: bool = True,
        l_bound: float = 0.1,
        u_bound: float = 0.9,
        output_dir: str = "data/output/plots"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute pairwise Damerau-Levenshtein similarity between email usernames.
 
        Args:
            df: DataFrame containing 'email_username' column.
            visualize: Whether to save similarity plots.
            l_bound, u_bound: Lower/upper similarity cutoff for filtering pairs.
            output_dir: Directory where plots will be saved.
 
        Returns:
            (updated DataFrame, similarity matrix DataFrame)
        """
 
        logging.info("Computing username similarity matrix using Damerau-Levenshtein distance...")
 
        if "email_username" not in df.columns:
            logging.warning("'email_username' column not found — skipping username similarity computation.")
            return df, pd.DataFrame()
 
        usernames = df["email_username"].astype(str).fillna("").values
        n = len(usernames)
 
        if n > 2000:
            logging.warning(
                f"Large dataset detected ({n:,} usernames). "
                "Pairwise Damerau-Levenshtein similarity is O(n²). "
                "Consider sampling or approximate methods."
            )
 
        # --- Compute full Damerau-Levenshtein similarity matrix ---
        logging.info(f"Computing {n}×{n} pairwise Damerau-Levenshtein similarity matrix...")
        levenshtein_damerau_matrix = np.array([
            [textdistance.damerau_levenshtein.normalized_similarity(u1, u2) for u2 in usernames]
            for u1 in usernames
        ])
 
        levenshtein_damerau_df = pd.DataFrame(
            levenshtein_damerau_matrix, index=usernames, columns=usernames
        )
 
        # --- Visualization (optional) ---
        if visualize:
            os.makedirs(output_dir, exist_ok=True)
            heatmap_path = os.path.join(output_dir, "username_similarity_heatmap.png")
 
            plt.figure(figsize=(12, 6))
            sns.heatmap(levenshtein_damerau_df, cmap="viridis")
            plt.title("Damerau-Levenshtein Username Similarity Matrix")
            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=300)
            plt.close()
            logging.info(f"Saved username similarity heatmap to: {heatmap_path}")
 
        # --- Filter similarity space ---
        df_tril = levenshtein_damerau_df.stack().reset_index()
        df_tril.columns = ["user_1", "user_2", "similarity"]
        df_filtered = df_tril[
            (df_tril["similarity"] >= l_bound) & (df_tril["similarity"] <= u_bound)
        ]
 
        # Plot similarity distribution
        if visualize and not df_filtered.empty:
            dist_path = os.path.join(output_dir, "username_similarity_distribution.png")
            sns.displot(df_filtered, x="similarity", kde=True)
            plt.title("Filtered Username Similarity Distribution")
            plt.tight_layout()
            plt.savefig(dist_path, dpi=300)
            plt.close()
            logging.info(f"Saved username similarity distribution to: {dist_path}")
 
        # --- Weighted similarity score per user ---
        df_weighted_similarity = (
            df_filtered.groupby("user_1")["similarity"]
            .sum()
            .reset_index(name="username_similarity_score")
        )
 
        # --- Merge back into original DataFrame ---
        df_result = df.merge(df_weighted_similarity, how="left", left_on="email_username", right_on="user_1")
        df_result.drop(columns=["user_1"], inplace=True, errors="ignore")
        df_result.set_index(df.index, inplace=True)
 
        logging.info(f"Username similarity computation completed successfully for {n:,} usernames.")
        return df_result, levenshtein_damerau_df