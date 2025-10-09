# utils/services_utils/data_transformation_utils.py
from __future__ import annotations
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
import textdistance
from typing import List, Tuple, Optional, Dict
import time
 
 
# Reuse the core utils you already have
from utils.model_utils.username_similarity_utils import (
    norm,
    safe_cast_int,
    safe_cast_float,
    user_features_sum_mean_same_cluster,
)
from src.models.username_similarity_model import UsernameSimilarityModel
from utils.model_utils.username_similarity_utils import *
 
logger = logging.getLogger(__name__)
 
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
    
 
    def finalize_username_similarity_outputs(
        self, 
        df: pd.DataFrame,
        usernames: List[str],
        refined_pairs: pd.DataFrame,
        user_clusters_unique: pd.DataFrame,
        *,
        cluster_edge_min_sim: float,
        neighbor_threshold: int,
        visualize: bool = False,
        audit_top_k: int = 20,
        output_dir: str = ".",
        log: Optional[logging.Logger] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Post-processing:
        - annotate refined_pairs with cluster IDs (and tidy columns/dtypes)
        - compute per-user SUM & MEAN features (same-cluster only)
        - build FULL user_clusters mapping (includes isolates as -1)
        - merge cluster_id + user features back into df (out_df)
        - optional audit CSV
        """
        _log = log or logger

        # ---------- annotate pairs with cluster IDs ----------
        refined_pairs_annot = refined_pairs.copy()
        if not refined_pairs.empty and not user_clusters_unique.empty:
            _log.info("Annotating pairs with cluster IDs...")
            _u2c = user_clusters_unique.set_index("email_username")["cluster_id"]
            refined_pairs_annot["cluster_id_1"] = refined_pairs_annot["user_1"].map(_u2c)
            refined_pairs_annot["cluster_id_2"] = refined_pairs_annot["user_2"].map(_u2c)
            refined_pairs_annot["same_cluster"] = (
                refined_pairs_annot["cluster_id_1"] == refined_pairs_annot["cluster_id_2"]
            ).astype(int)
            refined_pairs_annot["pair_cluster_id"] = np.where(
                refined_pairs_annot["same_cluster"] == 1,
                refined_pairs_annot["cluster_id_1"],
                -1,
            )
            # safe casts
            refined_pairs_annot["cluster_id_1"] = safe_cast_int(refined_pairs_annot["cluster_id_1"], default=-1)
            refined_pairs_annot["cluster_id_2"] = safe_cast_int(refined_pairs_annot["cluster_id_2"], default=-1)
            refined_pairs_annot["same_cluster"]  = safe_cast_int(refined_pairs_annot["same_cluster"], 0)
            refined_pairs_annot["pair_cluster_id"] = safe_cast_int(refined_pairs_annot["pair_cluster_id"], -1)

            base_cols = ["user_1","user_2","cluster_id_1","cluster_id_2","same_cluster","pair_cluster_id"]
            other_cols = [c for c in refined_pairs_annot.columns if c not in base_cols]
            refined_pairs_annot = refined_pairs_annot[base_cols + other_cols]

            same_cluster_edges = int((refined_pairs_annot["same_cluster"] == 1).sum()) if not refined_pairs_annot.empty else 0
            _log.info(
                "Pairs annotated with clusters: %s | same-cluster edges: %s",
                f"{len(refined_pairs_annot):,}", f"{same_cluster_edges:,}"
            )
        else:
            _log.info("No refined pairs or no user_clusters - skipping pair annotations.")

        # ---------- per-user SUM & MEAN features ----------
        user_feats = user_features_sum_mean_same_cluster(
            refined_pairs_with_clusters=refined_pairs_annot,
            cluster_edge_min_sim=cluster_edge_min_sim,
            neighbor_threshold=neighbor_threshold,
        )
        _log.info("Per-user features computed for %s users", f"{len(user_feats):,}")

        # ---------- full user_clusters mapping (ALL usernames; isolates = -1) ----------
        all_users_df = pd.DataFrame({"email_username": usernames})
        user_clusters_full = all_users_df.merge(user_clusters_unique, how="left", on="email_username")
        user_clusters_full["cluster_id"] = safe_cast_int(user_clusters_full["cluster_id"], default=-1)
        _log.info(
            "Full user->cluster mapping: %s rows (isolates=%s)",
            f"{len(user_clusters_full):,}",
            f"{int((user_clusters_full['cluster_id'] == -1).sum()):,}",
        )

        # ---------- out_df with cluster_id + user scores ----------
        out_df = df.copy()
        out_df["_uname_norm"] = out_df["email_username"].map(norm)
        out_df = out_df.merge(
            user_clusters_full.rename(columns={"email_username": "_uname_norm"}),
            how="left",
            on="_uname_norm",
        )
        out_df = out_df.merge(
            user_feats.rename(columns={"email_username": "_uname_norm"}),
            how="left",
            on="_uname_norm",
        )
        out_df.drop(columns=["_uname_norm"], inplace=True, errors="ignore")

        # Fill defaults & cast
        if "cluster_id" in out_df.columns:
            out_df["cluster_id"] = safe_cast_int(out_df["cluster_id"], default=-1)
        for col in ["username_similarity_score", "username_similarity_mean_score"]:
            if col in out_df.columns:
                out_df[col] = safe_cast_float(out_df[col], 0.0)
        for col in ["similarity_neighbors_count", "too_many_similar_usernames_flag"]:
            if col in out_df.columns:
                out_df[col] = safe_cast_int(out_df[col], 0)

        flagged = int(out_df.get("too_many_similar_usernames_flag", pd.Series([], dtype=int)).sum())
        _log.info(
            "out_df ready: %s rows x %s cols; flagged users=%s",
            f"{len(out_df):,}", f"{out_df.shape[1]:,}", f"{flagged:,}"
        )

        # ---------- optional audit ----------
        if visualize and not refined_pairs_annot.empty:
            try:
                os.makedirs(output_dir, exist_ok=True)
                path = os.path.join(output_dir, "username_similarity_top_pairs.csv")
                refined_pairs_annot.sort_values("similarity", ascending=False).head(audit_top_k).to_csv(path, index=False)
                _log.info("Saved top %d pairs to %s", audit_top_k, path)
            except Exception as e:
                _log.warning("Failed to save audit CSV: %s", e)

        return out_df, refined_pairs_annot, user_clusters_full, user_feats


    def username_similarity_score_optimized(
        self, 
        df: pd.DataFrame,
        visualize: bool = False,
        num_perm: int = 128,
        thresholds: Dict[str, float] | None = None,
        min_refined_similarity: float = 0.70,
        neighbor_threshold: int = 2,
        cluster_edge_min_sim: float = 0.70,
        cluster_lsh_threshold: float = 0.55,
        cluster_link_min_sim: float = 0.65,
        audit_top_k: int = 20,
        output_dir: str = ".",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Orchestrator: LSH -> candidates -> exact refinement -> clusters (+isolates=-1)
        -> per-user scores -> out_df merge -> cluster links (isolates excluded by default)
        """
        t_start = time.perf_counter()
        logger.info("username_similarity_score_optimized: start")

        if "email_username" not in df.columns:
            logger.error("Input DataFrame is missing 'email_username' column.")
            raise ValueError("df must contain 'email_username'")

        logger.info(
            "Params | num_perm=%s | min_refined_similarity=%s | cluster_edge_min_sim=%s | "
            "cluster_lsh_threshold=%s | cluster_link_min_sim=%s | neighbor_threshold=%s",
            num_perm, min_refined_similarity, cluster_edge_min_sim,
            cluster_lsh_threshold, cluster_link_min_sim, neighbor_threshold
        )
        if thresholds is not None:
            logger.info("Per-view LSH thresholds: %s", thresholds)

        # 1) Unique normalized usernames
        usernames = (df["email_username"].map(norm).replace("", np.nan).dropna().unique().tolist())
        n_users = len(usernames)
        logger.info("Unique normalized usernames: %s", f"{n_users:,}")

        # 2) Build multi-view LSH indices
        t0 = time.perf_counter()
        indices = build_multi_view_indices(usernames, num_perm=num_perm, thresholds=thresholds)
        logger.info(
            "Built LSH indices in %.2fs (raw=%s, sep=%s, shape=%s, deleet=%s)",
            time.perf_counter() - t0,
            f"{len(indices['raw'][1]):,}",
            f"{len(indices['sep'][1]):,}",
            f"{len(indices['shape'][1]):,}",
            f"{len(indices['deleet'][1]):,}",
        )

        # 3) Candidate pairs (union across views)
        t1 = time.perf_counter()
        candidates = multi_view_candidates(indices)
        cand_time = time.perf_counter() - t1
        avg_cand_per_user = (2 * len(candidates) / n_users) if n_users else 0.0
        logger.info(
            "LSH candidates: %s (avg ~ %.2f per user) in %.2fs",
            f"{len(candidates):,}", avg_cand_per_user, cand_time
        )

        # 4) Exact refinement
        t2 = time.perf_counter()
        refined_pairs = refine_pairs(candidates, min_sim=min_refined_similarity)
        logger.info(
            "Refined pairs kept (>= %.2f): %s (from %s) in %.2fs",
            min_refined_similarity, f"{len(refined_pairs):,}", f"{len(candidates):,}", time.perf_counter() - t2
        )

        # 5) Clusters (+ isolates = -1) & summary
        t3 = time.perf_counter()
        user_clusters_unique, cluster_summary = extract_clusters_with_labels(
            refined_pairs,
            min_sim=cluster_edge_min_sim,
            include_singletons=True,
            all_usernames=usernames,
            include_isolates_summary=True
        )
        multi_ct = int(cluster_summary.loc[cluster_summary["cluster_id"] != -1, "cluster_id"].nunique()) if not cluster_summary.empty else 0
        iso_ct   = int(cluster_summary.loc[cluster_summary["cluster_id"] == -1, "size"].sum()) if not cluster_summary.empty else 0
        logger.info(
            "Cluster extraction in %.2fs | multi-user clusters: %s | isolates summary size: %s",
            time.perf_counter() - t3,
            f"{multi_ct:,}", f"{iso_ct:,}"
        )

        # 6) Post-process (annotate pairs, user scores, full mapping, out_df)
        out_df, pairs_df, user_clusters_full, user_feats = self.finalize_username_similarity_outputs(
            df=df,
            usernames=usernames,
            refined_pairs=refined_pairs,
            user_clusters_unique=user_clusters_unique,
            cluster_edge_min_sim=cluster_edge_min_sim,
            neighbor_threshold=neighbor_threshold,
            visualize=visualize,
            audit_top_k=audit_top_k,
            output_dir=output_dir,
            log=logger
        )

        # 7) Inter-cluster links (exclude isolates by default)
        t4 = time.perf_counter()
        cluster_links = cluster_links_via_signatures(
            cluster_summary=cluster_summary,
            user_clusters=user_clusters_full,
            num_perm=num_perm,
            lsh_threshold=cluster_lsh_threshold,
            min_sim=cluster_link_min_sim,
            include_isolates=False
        )
        logger.info(
            "Inter-cluster links: %s in %.2fs (isolates excluded)",
            f"{len(cluster_links):,}", time.perf_counter() - t4
        )

        logger.info("username_similarity_score_optimized: done in %.2fs", time.perf_counter() - t_start)
        return out_df, pairs_df, user_clusters_full, cluster_summary, cluster_links


