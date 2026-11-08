"""Causal history features used by the main model repetition branch."""

import numpy as np
from scipy import sparse


FEATURE_NAMES = [
    "log_pair_count",
    "log_recent_count",
    "log_previous_count",
    "log_decayed_count",
    "personal_share",
    "recent_personal_share",
    "recent_change",
    "years_since_pair",
    "global_share",
    "global_recent_share",
    "global_change",
    "log_org_events",
    "org_distinct_labels",
    "org_age_years",
    "class_personal_share",
    "section_personal_share",
]


class PairFeatures:
    def __init__(self, dataset, cutoff):
        self.data = dataset
        self.cutoff = cutoff
        self.history = dataset.history(cutoff)
        self.recent = dataset.counts(cutoff)
        self.previous = dataset.counts(cutoff - 1)
        self.decayed = dataset.history(cutoff, half_life=2.0)
        self.total = np.asarray(self.history.sum(axis=1)).ravel()
        self.recent_total = np.asarray(self.recent.sum(axis=1)).ravel()
        self.global_counts = np.asarray(self.history.sum(axis=0)).ravel()
        self.global_recent = np.asarray(self.recent.sum(axis=0)).ravel()
        self.global_previous = np.asarray(self.previous.sum(axis=0)).ravel()
        self.distinct = np.diff(self.history.indptr)
        self.lag = sparse.csr_matrix(self.history.shape, dtype=np.float32)
        # Store last year relative to a nonzero origin, no future target-year data.
        self.origin = min(dataset.manifest["years"]) - 1
        for year in dataset.manifest["years"]:
            if year <= cutoff:
                current = dataset.counts(year).sign() * (year - self.origin)
                self.lag = self.lag.maximum(current)
        self.parents = {}
        for width in [1, 3]:
            names = sorted({c[:width] for c in dataset.codes})
            ids = {c: i for i, c in enumerate(names)}
            mapping = np.array([ids[c[:width]] for c in dataset.codes])
            projection = sparse.coo_matrix(
                (np.ones(len(mapping)), (np.arange(len(mapping)), mapping)),
                shape=(len(mapping), len(names)),
            ).tocsr()
            self.parents[width] = (self.history @ projection, mapping)

    def pairs(self, orgs, labels):
        orgs, labels = np.asarray(orgs), np.asarray(labels)

        def pair(matrix):
            return np.asarray(matrix[orgs, labels]).ravel()

        hist, recent, previous, decay = [
            pair(m) for m in [self.history, self.recent, self.previous, self.decayed]
        ]
        total = np.maximum(self.total[orgs], 1.0)
        rtotal = np.maximum(self.recent_total[orgs], 1.0)
        last = pair(self.lag)
        lag = np.where(last > 0, self.cutoff - self.origin - last, 30.0)
        values = [
            np.log1p(hist),
            np.log1p(recent),
            np.log1p(previous),
            np.log1p(decay),
            hist / total,
            recent / rtotal,
            np.log1p(recent) - np.log1p(previous),
            lag,
            self.global_counts[labels] / max(self.global_counts.sum(), 1.0),
            self.global_recent[labels] / max(self.global_recent.sum(), 1.0),
            np.log1p(self.global_recent[labels])
            - np.log1p(self.global_previous[labels]),
            np.log1p(total),
            self.distinct[orgs],
            self.cutoff - self.data.first_year[orgs] + 1,
        ]
        for width in [3, 1]:
            counts, mapping = self.parents[width]
            values.append(np.asarray(counts[orgs, mapping[labels]]).ravel() / total)
        return np.stack(values, axis=1).astype(np.float32)
