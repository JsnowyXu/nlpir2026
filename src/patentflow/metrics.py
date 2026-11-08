"""Full-catalog multi-label ranking metrics with explicit empty-target policy."""
import numpy as np


def ranking_metrics(scores, targets, ks=(10, 20, 30, 40), candidate_ids=None):
    """Rows are org-cutoff queries; target IDs can lie outside the catalog.

    AP@K divides by min(K_eff, |ground truth|); MRR@K is first relevant rank.
    PHR follows EDGPAT: fraction of queries with >=1 relevant prediction.
    Active metrics exclude empty baskets; all-query metrics zero-fill them.
    Out-of-catalog positives remain in Recall/AP/NDCG denominators.
    A stable column order breaks score ties, independent of random seeds.
    """
    scores = np.asarray(scores)
    if scores.ndim != 2 or len(targets) != len(scores):
        raise ValueError("Expected scores [queries,candidates] and one target set per query")
    if not np.isfinite(scores).all():
        raise ValueError("All candidate scores must be finite")
    if scores.shape[1] == 0:
        raise ValueError("Empty candidate catalog")
    if not ks or any(k < 1 or int(k) != k for k in ks):
        raise ValueError("Ranking cutoffs must be positive integers")
    candidate_ids = np.arange(scores.shape[1]) if candidate_ids is None else np.asarray(candidate_ids)
    if len(candidate_ids) != scores.shape[1] or len(set(candidate_ids.tolist())) != len(candidate_ids):
        raise ValueError("Candidate IDs must be unique and match score columns")
    targets = [set(t) for t in targets]
    target_size = np.asarray([len(t) for t in targets], dtype=np.int64)
    active = target_size > 0
    max_k = min(max(ks), scores.shape[1])
    rank = np.argsort(-scores, axis=1, kind="stable")[:, :max_k]
    ranked_ids = candidate_ids[rank]
    hits = np.asarray([[int(c in targets[i]) for c in row] for i, row in enumerate(ranked_ids)], dtype=float)
    if not len(scores):
        hits = np.zeros((0, max_k))
    catalog = set(candidate_ids.tolist())
    result = {"n_queries": len(targets), "n_active": int(active.sum()),
              "n_empty": int((~active).sum()), "n_candidates": scores.shape[1],
              "oov_positive_count": sum(len(t - catalog) for t in targets),
              "positive_count": int(target_size.sum()), "active": {}, "all_zero_filled": {},
              "effective_k": {}, "per_query": {}}
    for k in ks:
        kk = min(k, scores.shape[1])
        h = hits[:, :kk]
        cumulative = h.cumsum(axis=1)
        ranks = np.arange(1, kk + 1)
        dcg = (h / np.log2(ranks + 1)).sum(axis=1)
        ideal_n = np.minimum(target_size, kk)
        discounts = np.r_[0., (1. / np.log2(ranks + 1)).cumsum()]
        denom = np.maximum(target_size, 1)
        values = {
            "recall": h.sum(axis=1) / denom,
            "precision": h.sum(axis=1) / kk,
            "mrr": (h / ranks).max(axis=1),
            "ap": ((cumulative / ranks) * h).sum(axis=1) / np.maximum(ideal_n, 1),
            "ndcg": dcg / np.maximum(discounts[ideal_n], 1e-30),
            "phr": (h.sum(axis=1) > 0).astype(float),
        }
        result["effective_k"][str(k)] = kk
        for metric, value in values.items():
            key = "%s_%d" % (metric, k)
            result["per_query"][key] = value
            result["active"][key] = float(value[active].mean()) if active.any() else None
            result["all_zero_filled"][key] = float(value.mean()) if len(value) else None
    return result
