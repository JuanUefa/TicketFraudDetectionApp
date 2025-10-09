# src/models/username_similarity_model.py
# Model wrapper around the utils: builds indices, generates candidates, refines, clusters, links
from __future__ import annotations
 
import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple
 
import pandas as pd
 
from utils.model_utils.username_similarity_utils import *
 
logger = logging.getLogger(__name__)
 
@dataclass
class UsernameSimilarityModel:
    num_perm: int = 128
    thresholds: Dict[str, float] | None = None
    min_refined_similarity: float = 0.70
    cluster_edge_min_sim: float = 0.70
    cluster_lsh_threshold: float = 0.55
    cluster_link_min_sim: float = 0.65
 
    def run(self, usernames: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Args:
            usernames: list of raw usernames (strings)
 
        Returns:
            refined_pairs   : candidate user-user pairs with exact similarity
            user_clusters   : username -> cluster_id (isolates = -1)
            cluster_summary : cluster-level summary (includes isolates row)
            cluster_links   : inter-cluster similarity links (isolates excluded)
        """
        t0 = time.perf_counter()
        users = [u for u in (norm(x) for x in usernames) if u]
        logger.info("Model.run: %d unique normalized usernames", len(users))
 
        # LSH indices & candidates
        indices = build_multi_view_indices(users, num_perm=self.num_perm, thresholds=self.thresholds)
        candidates = multi_view_candidates(indices)
        logger.info("Model.run: LSH candidates = %d", len(candidates))
 
        # Exact refinement
        refined_pairs = refine_pairs(candidates, min_sim=self.min_refined_similarity)
        logger.info("Model.run: refined pairs kept (>= %.2f) = %d",
                    self.min_refined_similarity, len(refined_pairs))
 
        # Clustering (with singletons labeled -1) + summary (with isolates summary row)
        user_clusters, cluster_summary = extract_clusters_with_labels(
            refined_pairs,
            min_sim=self.cluster_edge_min_sim,
            include_singletons=True,
            all_usernames=users,
            include_isolates_summary=True
        )
        logger.info("Model.run: clusters (multi-user) = %d | isolates = %d",
                    int((cluster_summary["cluster_id"] != -1).sum()),
                    int(cluster_summary.loc[cluster_summary["cluster_id"] == -1, "size"].sum() if not cluster_summary.empty else 0))
 
        # Inter-cluster links (exclude isolates)
        cluster_links = cluster_links_via_signatures(
            cluster_summary=cluster_summary,
            user_clusters=user_clusters,
            num_perm=self.num_perm,
            lsh_threshold=self.cluster_lsh_threshold,
            min_sim=self.cluster_link_min_sim,
            include_isolates=False
        )
        logger.info("Model.run: cluster links = %d | total time = %.2fs",
                    len(cluster_links), time.perf_counter() - t0)
 
        return refined_pairs, user_clusters, cluster_summary, cluster_links