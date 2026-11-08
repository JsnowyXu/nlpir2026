"""Shared annual query generation for every forecasting method."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from .data import ROOT


class DatasetView:
    def __init__(self, name, level=3):
        self.spec = json.loads((ROOT / 'data/processed/views' / (name + '.json')).read_text())
        self.path = ROOT / 'data/processed' / self.spec['country']
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        self.level = level
        if level not in (1, 2, 3):
            raise ValueError('IPC level must be 1, 2 or 3')
        self.all_leaf_codes = json.loads((self.path / 'subclasses.json').read_text())
        self.first_year = np.load(self.path / 'organization_first_seen.npz')['year']
        width = {1: 1, 2: 3, 3: 4}[level]
        selected = self.spec['subclass_ids']
        self.codes = sorted({self.all_leaf_codes[i][:width] for i in selected})
        code_to_id = {c: i for i, c in enumerate(self.codes)}
        self.candidate_ids = np.array(sorted({code_to_id[self.all_leaf_codes[i][:width]]
                                              for i in self.spec['train_candidate_ids']}), dtype=np.int32)
        self.projection = sparse.coo_matrix((np.ones(len(selected), dtype=np.float32),
                            (selected, [code_to_id[self.all_leaf_codes[i][:width]] for i in selected])),
                            shape=(len(self.all_leaf_codes), len(self.codes))).tocsr()
        self._annual = {}

    def counts(self, year):
        if year not in self._annual:
            path = self.path / ('counts_%d.npz' % year)
            if path.exists():
                self._annual[year] = (sparse.load_npz(path) @ self.projection).tocsr()
            else:
                self._annual[year] = sparse.csr_matrix((len(self.first_year), len(self.codes)), dtype=np.float32)
        return self._annual[year]

    def history(self, cutoff_year, half_life=None):
        result = sparse.csr_matrix((len(self.first_year), len(self.codes)), dtype=np.float32)
        for year in self.manifest['years']:
            if year > cutoff_year:
                break
            weight = 1. if half_life is None else 2. ** (-(cutoff_year - year) / half_life)
            result = result + weight * self.counts(year)
        return result.tocsr()

    def queries(self, target_year):
        # Global previously observed applicants; no filtering by future activity.
        orgs = np.flatnonzero(self.first_year < target_year)
        target = self.counts(target_year)[orgs].tocsr()
        return orgs, [target.indices[target.indptr[i]:target.indptr[i+1]] for i in range(len(orgs))]

    def split_years(self, split):
        if split == 'train':
            return [y for y in self.manifest['supervised_target_years'] if y <= self.spec['train_end_year']]
        return [self.spec['validation_year'] if split == 'validation' else self.spec['test_year']]

    def collaborators(self, cutoff_year):
        file = self.path / ('collaborators_%d.npz' % cutoff_year)
        if not file.exists():
            return np.full((len(self.first_year), 3), -1, dtype=np.int32)
        partners = np.load(file)['partners']
        padding = np.full((len(self.first_year) - len(partners), 3), -1, dtype=np.int32)
        return np.vstack([partners, padding])
