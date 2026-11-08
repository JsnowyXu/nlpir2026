"""Selected-view raw event access for original-model adapters."""
import numpy as np
import torch
from .data import ROOT


class EventStream:
    def __init__(self, data):
        self.data = data
        events = np.load(data.path / 'events.npz')
        mask = np.isin(events['dst'], data.spec['subclass_ids'])
        self.org_count = len(data.first_year)
        code_ids = {code: i for i, code in enumerate(data.codes)}
        width = {1: 1, 2: 3, 3: 4}[data.level]
        mapping = np.array([code_ids.get(c[:width], -1) for c in data.all_leaf_codes])
        self.src = events['src'][mask].astype(np.int64)
        self.dst = mapping[events['dst'][mask]].astype(np.int64) + self.org_count
        self.day = events['day'][mask].astype(np.int64)
        self.origin_day = int(self.day.min())
        self.day = self.day - self.day.min() + 1
        self.year = events['year'][mask]
        self.rows = events['patent_row'][mask]
        self.cuts = np.r_[0, np.flatnonzero(np.diff(self.day)) + 1, len(self.day)]
        self.vectors = np.load(ROOT / 'data/text' / data.spec['country'] / 'patent_vectors.npy', mmap_mode='r')
        self.types = np.array([ord(data.all_leaf_codes[i][0]) - ord('A') for i in events['dst'][mask]])
        self.node_count = self.org_count + len(data.codes)

    def messages(self, indices, device, edge_types=False):
        values = self.vectors[self.rows[indices]]
        if edge_types:
            values = np.concatenate([values, np.eye(8, dtype=np.float32)[self.types[indices]]], axis=1)
        return torch.as_tensor(values, device=device)

    def groups(self, max_year):
        for left, right in zip(self.cuts[:-1], self.cuts[1:]):
            if self.year[left] > max_year:
                break
            yield int(left), int(right), int(self.year[left])
