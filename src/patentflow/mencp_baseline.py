"""Original MENCP TextCNN + hierarchical attention, with historical text input."""
import json
import sys
import numpy as np
import torch
from torch import nn
from .data import ROOT, file_hash

SOURCE = ROOT/'baseline/MENCP'
sys.path.insert(0, str(SOURCE))
from models.dhpt import Embedding, TextCNNModel
from models.hier_ipc_emb import Hierarchy_Embedding


class HistoricalTokens:
    def __init__(self, data, tokens=64, history=4):
        self.path = ROOT/'data/sequences'/data.spec['country']/('tokens%d_history%d' % (tokens, history))
        self.data, self.tokens, self.history = data, tokens, history
        self.ids = np.load(self.path/'tokens.npy', mmap_mode='r')
        self.indices = {}

    def batch(self, orgs, cutoff):
        if cutoff not in self.indices:
            name = self.data.spec['country'].upper()+'_'+self.data.spec['section']
            self.indices[cutoff] = np.load(self.path/name/('%d.npy' % cutoff), mmap_mode='r')
        rows = self.indices[cutoff][orgs]
        ids = self.ids[np.maximum(rows,0)].copy()
        ids[rows < 0] = 0
        ids = ids.reshape(len(orgs), -1)
        empty = (ids != 0).sum(axis=1) == 0
        ids[empty,0] = 2  # real mask position prevents all-masked attention NaNs.
        return ids.astype(np.int64)


class MENCPForecast(nn.Module):
    def __init__(self, data, initial_embeddings, device='cpu', filters=32, dropout=.1):
        super().__init__()
        self.data, self.device_name = data, device
        self.embedding = Embedding(device, torch.as_tensor(initial_embeddings), 0, dropout)
        self.embedding.emb.padding_idx = 0
        self.cnn = TextCNNModel(filters, len(data.candidate_ids), 100, device, dropout=dropout)
        self.hierarchy = None
        if data.level > 1:
            leaves = [data.codes[i] for i in data.candidate_ids]
            codes = [sorted({c[0] for c in leaves}), sorted({c[:3] for c in leaves})]
            if data.level == 3:
                codes.append(leaves)
            trees = []
            for parents, children in zip(codes[:-1], codes[1:]):
                index = {c: i for i,c in enumerate(parents)}
                tree = torch.zeros(len(parents), len(children))
                for j, child in enumerate(children):
                    tree[index[child[:len(parents[0])]],j] = 1
                trees.append(tree)
            anchors = np.load(ROOT/'data/text'/data.spec['country']/'taxonomy_anchors.npz')
            lookup = dict(zip(anchors['codes'], anchors['anchors']))
            # Constructor slices 8/124 hard-coded groups. Replace those embeddings
            # with exact country/view sizes; keep all original message/attention code.
            dummy = np.zeros((132+len(leaves),100),dtype=np.float32)
            self.hierarchy = Hierarchy_Embedding(100,100,trees,[len(c) for c in codes],
                device,data.level,[dummy],dropout=dropout)
            self.hierarchy.NodeEmb = nn.ModuleList([
                nn.Embedding.from_pretrained(torch.as_tensor(np.stack([lookup[c] for c in level])),freeze=False)
                for level in codes])
        self.to(device)

    def forward(self, ids):
        text, _, mask = self.embedding(ids)
        _, logits = self.cnn(text)
        if self.hierarchy is not None:
            logits = logits + self.hierarchy(text, mask)
        return logits

    def provenance(self):
        return {str((SOURCE/'models'/name).relative_to(ROOT)):file_hash(SOURCE/'models'/name)
                for name in ['dhpt.py','hier_ipc_emb.py']}
