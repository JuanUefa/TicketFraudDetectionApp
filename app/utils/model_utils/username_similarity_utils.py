# utils/model_utils/username_similarity_utils.py
# Core helpers: normalization, MinHash/LSH, refinement, clustering, links, scoring
from __future__ import annotations

import re
import logging
from typing import Iterable, Dict, Tuple, Set, Sequence, List, Optional

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from rapidfuzz.distance import DamerauLevenshtein as DL
from rapidfuzz.distance import JaroWinkler as JW
from rapidfuzz import fuzz


logger = logging.getLogger(__name__)

# ---------------------------
# Safe casting helpers
# ---------------------------

def safe_cast_int(s: pd.Series, default: int = 0) -> pd.Series:
    return (pd.to_numeric(s, errors="coerce")
              .replace([np.inf, -np.inf], np.nan)
              .fillna(default)
              .astype(int))

def safe_cast_float(s: pd.Series, default: float = 0.0) -> pd.Series:
    return (pd.to_numeric(s, errors="coerce")
              .replace([np.inf, -np.inf], np.nan)
              .fillna(default)
              .astype(float))

# ---------------------------
# Normalization & transforms
# ---------------------------

def norm(u: str) -> str:
    """Lowercase + trim; keep digits & punctuation."""
    return "" if u is None else str(u).strip().lower()

def sep_norm(u: str) -> str:
    """Unify separators: non-alnum -> '_' (collapse runs)."""
    u = norm(u)
    u = re.sub(r"[^a-z0-9]", "_", u)
    u = re.sub(r"_+", "_", u).strip("_")
    return u

_LEET = str.maketrans({"0":"o","1":"i","3":"e","4":"a","5":"s","7":"t","8":"b","@":"a","$":"s","!":"i"})
def deleet_norm(u: str) -> str:
    """Map common leetspeak to letters; keep punctuation."""
    return norm(u).translate(_LEET)

def shape(u: str) -> str:
    """letters->a, digits->d, specials->s; collapse runs."""
    u = "" if u is None else str(u)
    out, last = [], None
    for ch in u:
        t = "a" if ch.isalpha() else ("d" if ch.isdigit() else "s")
        if t != last:
            out.append(t); last = t
    return "".join(out)

def letters_only(u: str) -> str:
    return re.sub(r"[^a-zA-Z]+", "", str(u)).lower()

def tokens_for_ratio(u: str) -> str:
    """Separators -> spaces for token_set_ratio; keeps digits as tokens."""
    u = norm(u)
    u = re.sub(r"[^a-z0-9]", " ", u)
    u = re.sub(r"\s+", " ", u).strip()
    return u

def canonical_label(u: str) -> str:
    """Separator-unified & digit-runs collapsed → stable cluster label."""
    s = sep_norm(u)
    s = re.sub(r"\d+", "0", s)
    return s

# ---------------------------
# Shingles & MinHash / LSH
# ---------------------------

def make_shingles(s: str, n_values: Sequence[int] = (2,3,4)) -> Set[str]:
    s = "" if s is None else str(s)
    S = set()
    for n in n_values:
        if len(s) < n:
            if s: S.add(s)
        else:
            for i in range(len(s)-n+1):
                S.add(s[i:i+n])
    return S

def build_minhash(shingles: Set[str], num_perm: int = 128) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for sh in shingles:
        mh.update(sh.encode("utf-8"))
    return mh

def build_view_index(
    items: Iterable[str], transform, n_values=(2,3,4), num_perm=128, threshold=0.65
) -> Tuple[MinHashLSH, Dict[str, MinHash]]:
    mhs: Dict[str, MinHash] = {}
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for u in items:
        t = transform(u)
        mh = build_minhash(make_shingles(t, n_values), num_perm=num_perm)
        mhs[u] = mh
        lsh.insert(u, mh)
    return lsh, mhs

def build_multi_view_indices(
    usernames: Iterable[str], num_perm=128, thresholds: Dict[str, float] | None = None
) -> Dict[str, Tuple[MinHashLSH, Dict[str, MinHash]]]:
    if thresholds is None:
        thresholds = {"raw":0.65,"sep":0.60,"shape":0.85,"deleet":0.65}
    users = [u for u in (norm(x) for x in usernames) if u]
    idx_raw    = build_view_index(users, transform=norm,       n_values=(2,3,4), num_perm=num_perm, threshold=thresholds["raw"])
    idx_sep    = build_view_index(users, transform=sep_norm,   n_values=(2,3,4), num_perm=num_perm, threshold=thresholds["sep"])
    idx_shape  = build_view_index(users, transform=shape,      n_values=(3,),    num_perm=num_perm, threshold=thresholds["shape"])
    idx_deleet = build_view_index(users, transform=deleet_norm,n_values=(2,3,4), num_perm=num_perm, threshold=thresholds["deleet"])
    return {"raw":idx_raw,"sep":idx_sep,"shape":idx_shape,"deleet":idx_deleet}

def multi_view_candidates(indices: Dict[str, Tuple[MinHashLSH, Dict[str, MinHash]]]) -> pd.DataFrame:
    users = list(indices["raw"][1].keys())
    seen, pairs = set(), []
    for u in users:
        cand_union = set()
        for _, (lsh, mhs) in indices.items():
            cand_union.update(lsh.query(mhs[u]))
        for v in cand_union:
            if u >= v: continue
            key = (u,v)
            if key in seen: continue
            seen.add(key)
            jacc = indices["raw"][1][u].jaccard(indices["raw"][1][v])
            pairs.append((u, v, float(jacc)))
    return pd.DataFrame(pairs, columns=["user_1","user_2","jaccard_raw_view"])

# ---------------------------
# Refinement (exact & robust)
# ---------------------------

def combined_similarity(u: str, v: str) -> float:
    u_raw, v_raw = u, v
    u_sep, v_sep = sep_norm(u), sep_norm(v)
    u_del, v_del = deleet_norm(u), deleet_norm(v)
    u_tok, v_tok = tokens_for_ratio(u), tokens_for_ratio(v)

    s1 = DL.normalized_similarity(u_raw, v_raw)
    s2 = DL.normalized_similarity(u_sep, v_sep)
    s3 = DL.normalized_similarity(u_del, v_del)
    s4 = JW.normalized_similarity(u_raw, v_raw)
    s5 = fuzz.token_set_ratio(u_tok, v_tok) / 100.0

    u_let, v_let = letters_only(u_raw), letters_only(v_raw)
    boost = 0.05 if (u_let and v_let and DL.normalized_similarity(u_let, v_let) >= 0.90) else 0.0
    return min(1.0, max(s1, s2, s3, s4, s5) + boost)

def refine_pairs(pairs_df: pd.DataFrame, min_sim: float = 0.70) -> pd.DataFrame:
    if pairs_df.empty:
        out = pairs_df.copy(); out["similarity"] = []
        return out
    sims = [combined_similarity(u, v) for u, v in pairs_df[["user_1","user_2"]].itertuples(index=False)]
    out = pairs_df.copy()
    out["similarity"] = sims
    return out[out["similarity"] >= min_sim].reset_index(drop=True)

# ---------------------------
# Clustering (+ isolates = -1) with labels & centroid metrics
# ---------------------------

def extract_clusters_with_labels(
    pairs_df: pd.DataFrame,
    min_sim: float = 0.70,
    include_singletons: bool = True,
    all_usernames: Optional[List[str]] = None,
    include_isolates_summary: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build clusters from edges >= min_sim. Users with no edges get cluster_id = -1.
    cluster_summary includes multi-user clusters (id 1..K) + one isolates row (id = -1).
    """
    pf = pairs_df[pairs_df["similarity"] >= min_sim].copy()

    def _build_adj_local(pdf: pd.DataFrame) -> Dict[str, Set[str]]:
        adj: Dict[str, Set[str]] = {}
        for u, v in pdf[["user_1","user_2"]].itertuples(index=False):
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        return adj

    def _connected_components(adj: Dict[str, Set[str]]) -> List[List[str]]:
        seen, comps = set(), []
        for start in adj.keys():
            if start in seen: continue
            stack = [start]; seen.add(start); comp=[]
            while stack:
                x = stack.pop(); comp.append(x)
                for y in adj.get(x, ()):
                    if y not in seen:
                        seen.add(y); stack.append(y)
            comps.append(comp)
        return comps

    adj = _build_adj_local(pf)
    comps = _connected_components(adj)
    multi_comps = [c for c in comps if len(c) > 1]

    # singletons (no qualified edges)
    singleton_usernames: List[str] = []
    if include_singletons:
        nodes_seen = set(all_usernames) if all_usernames is not None \
                     else set(pairs_df["user_1"]).union(set(pairs_df["user_2"]))
        singleton_usernames = [u for u in nodes_seen if u not in adj]

    user_rows, cluster_rows = [], []
    cid = 1  # positive ids for multi-user clusters

    # multi-user clusters
    for nodes in multi_comps:
        sub = pf[(pf["user_1"].isin(nodes)) & (pf["user_2"].isin(nodes))]

        # node weights (incident similarity)
        weights = {u:0.0 for u in nodes}
        if not sub.empty:
            for u, s in sub.groupby("user_1")["similarity"].sum().items(): weights[u] += float(s)
            for v, s in sub.groupby("user_2")["similarity"].sum().items(): weights[v] += float(s)

        medoid = max(weights.items(), key=lambda kv: kv[1])[0] if nodes else None

        if medoid is not None and not sub.empty:
            med_edges = pd.concat([
                sub[sub["user_1"] == medoid][["user_2","similarity"]].rename(columns={"user_2":"neighbor"}),
                sub[sub["user_2"] == medoid][["user_1","similarity"]].rename(columns={"user_1":"neighbor"})
            ], ignore_index=True)
            centroid_sum   = float(med_edges["similarity"].sum()) if not med_edges.empty else 0.0
            centroid_mean  = float(med_edges["similarity"].mean()) if not med_edges.empty else 0.0
            centroid_neigh = int(med_edges["neighbor"].nunique()) if not med_edges.empty else 0
            avg_sim = float(sub["similarity"].mean())
        else:
            centroid_sum = centroid_mean = 0.0
            centroid_neigh = 0
            avg_sim = 0.0

        label = canonical_label(medoid) if medoid else ""

        cluster_rows.append({
            "cluster_id": cid,
            "size": len(nodes),
            "medoid_username": medoid,
            "canonical_label": label,
            "avg_similarity": avg_sim,
            "centroid_sum_similarity": centroid_sum,
            "centroid_mean_similarity": centroid_mean,
            "centroid_neighbor_count": centroid_neigh
        })
        for u in nodes:
            user_rows.append({"email_username": u, "cluster_id": cid})
        cid += 1

    # isolates => cluster_id = -1
    for u in singleton_usernames:
        user_rows.append({"email_username": u, "cluster_id": -1})

    user_clusters = pd.DataFrame(user_rows)

    cluster_summary = pd.DataFrame(cluster_rows)
    if include_isolates_summary:
        iso_count = len(singleton_usernames)
        if iso_count > 0:
            iso_row = {
                "cluster_id": -1,
                "size": int(iso_count),
                "medoid_username": None,
                "canonical_label": "__isolates__",
                "avg_similarity": 0.0,
                "centroid_sum_similarity": 0.0,
                "centroid_mean_similarity": 0.0,
                "centroid_neighbor_count": 0
            }
            cluster_summary = pd.concat([cluster_summary, pd.DataFrame([iso_row])], ignore_index=True)

    if not cluster_summary.empty:
        cluster_summary["__is_iso__"] = (cluster_summary["cluster_id"] == -1).astype(int)
        cluster_summary = (cluster_summary
                            .sort_values(["__is_iso__","size","avg_similarity"], ascending=[True, False, False])
                            .drop(columns="__is_iso__")
                            .reset_index(drop=True))

    # ensure dtypes
    if not user_clusters.empty:
        user_clusters["cluster_id"] = safe_cast_int(user_clusters["cluster_id"], default=-1)
    for col in ["cluster_id","size","centroid_neighbor_count"]:
        if col in cluster_summary.columns:
            cluster_summary[col] = safe_cast_int(cluster_summary[col], default=0)
    for col in ["avg_similarity","centroid_sum_similarity","centroid_mean_similarity"]:
        if col in cluster_summary.columns:
            cluster_summary[col] = safe_cast_float(cluster_summary[col], default=0.0)

    return user_clusters, cluster_summary

# ---------------------------
# Inter-cluster links (exclude isolates by default)
# ---------------------------

def _cluster_member_map(user_clusters: pd.DataFrame) -> Dict[int, Set[str]]:
    m: Dict[int, Set[str]] = {}
    for cid, grp in user_clusters.groupby("cluster_id"):
        m[int(cid)] = set(grp["email_username"].astype(str))
    return m

def build_cluster_signatures(user_clusters: pd.DataFrame, n_values=(2,3,4)) -> Dict[int, Set[str]]:
    sigs: Dict[int, Set[str]] = {}
    member_map = _cluster_member_map(user_clusters)
    for cid, members in member_map.items():
        if cid == -1:  # skip isolates bucket
            continue
        S: Set[str] = set()
        for u in members:
            S.update(make_shingles(sep_norm(u), n_values))
            S.update(make_shingles(deleet_norm(u), n_values))
        sigs[int(cid)] = S
    return sigs

def cluster_links_via_signatures(
    cluster_summary: pd.DataFrame,
    user_clusters: pd.DataFrame,
    num_perm: int = 128,
    lsh_threshold: float = 0.55,
    min_sim: float = 0.65,
    include_isolates: bool = False
) -> pd.DataFrame:
    # Exclude isolates by default
    if not include_isolates:
        cluster_summary = cluster_summary[cluster_summary["cluster_id"] != -1].copy()
        user_clusters   = user_clusters[user_clusters["cluster_id"] != -1].copy()

    if cluster_summary.empty or cluster_summary["cluster_id"].nunique() <= 1:
        return pd.DataFrame(columns=["cluster_id_1","cluster_id_2","cluster_similarity"])

    sigs = build_cluster_signatures(user_clusters, n_values=(2,3,4))
    if not sigs:
        return pd.DataFrame(columns=["cluster_id_1","cluster_id_2","cluster_similarity"])

    mh: Dict[int, MinHash] = {}
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
    for cid, S in sigs.items():
        mh_c = build_minhash(S, num_perm=num_perm)
        mh[cid] = mh_c
        lsh.insert(str(cid), mh_c)

    labels = cluster_summary.set_index("cluster_id")["canonical_label"].astype(str).to_dict()

    links = []
    for cid in sigs.keys():
        cands = lsh.query(mh[cid])
        for other_str in cands:
            other = int(other_str)
            if cid >= other: continue
            jacc_est  = mh[cid].jaccard(mh[other])
            label_sim = combined_similarity(labels.get(cid, ""), labels.get(other, ""))
            sim = max(float(jacc_est), float(label_sim))
            if sim >= min_sim:
                links.append((int(cid), int(other), float(sim)))

    links_df = pd.DataFrame(links, columns=["cluster_id_1","cluster_id_2","cluster_similarity"])
    if not links_df.empty:
        links_df["cluster_id_1"] = safe_cast_int(links_df["cluster_id_1"], 0)
        links_df["cluster_id_2"] = safe_cast_int(links_df["cluster_id_2"], 0)
        links_df["cluster_similarity"] = safe_cast_float(links_df["cluster_similarity"], 0.0)
    return links_df

# ---------------------------
# Per-user features (SUM & MEAN over same-cluster neighbors)
# ---------------------------

def user_features_sum_mean_same_cluster(
    refined_pairs_with_clusters: pd.DataFrame,
    cluster_edge_min_sim: float,
    neighbor_threshold: int
) -> pd.DataFrame:
    cols = [
        "email_username","username_similarity_score","username_similarity_mean_score",
        "similarity_neighbors_count","too_many_similar_usernames_flag"
    ]
    if refined_pairs_with_clusters.empty:
        return pd.DataFrame(columns=cols)

    sc = refined_pairs_with_clusters[
        (refined_pairs_with_clusters["same_cluster"] == 1) &
        (refined_pairs_with_clusters["similarity"] >= cluster_edge_min_sim)
    ]
    if sc.empty:
        return pd.DataFrame(columns=cols)

    a = sc[["user_1","similarity"]].rename(columns={"user_1":"email_username"})
    b = sc[["user_2","similarity"]].rename(columns={"user_2":"email_username"})
    stacked = pd.concat([a,b], ignore_index=True)

    agg = stacked.groupby("email_username")["similarity"].agg(sum="sum", mean="mean", count="count").reset_index()
    agg.rename(columns={
        "sum":"username_similarity_score",
        "mean":"username_similarity_mean_score",
        "count":"similarity_neighbors_count"
    }, inplace=True)
    agg["too_many_similar_usernames_flag"] = (agg["similarity_neighbors_count"] > neighbor_threshold).astype(int)

    for col in ["username_similarity_score","username_similarity_mean_score"]:
        agg[col] = safe_cast_float(agg[col], 0.0)
    for col in ["similarity_neighbors_count","too_many_similar_usernames_flag"]:
        agg[col] = safe_cast_int(agg[col], 0)
    return agg