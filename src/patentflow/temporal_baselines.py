"""Forecast adapters retaining the vendored TGB TGN and TAMI computations.

Date transactions replace TGN's event-by-event delayed message store. All
messages in one date read the same old state; a single mean/GRU update follows.
"""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .data import ROOT, file_hash

TGB = ROOT / 'baseline/vendor/TGB2-main/TGB2-main'
sys.path.insert(0, str(TGB))
from modules.memory_module import TGNMemory
from modules.msg_func import IdentityMessage
from modules.msg_agg import MeanAggregator
from modules.emb_module import GraphAttentionEmbedding
from modules.decoder import LinkPredictor
from modules.neighbor_loader import LastNeighborLoader


class TransactionMemory(TGNMemory):
    def reset_state(self):
        self.memory.zero_()
        self.last_update.zero_()
        self.msg_s_store, self.msg_d_store = {}, {}

    def train(self, mode=True):
        # No pending messages: switching mode must not flush empty stores through
        # the original GRU, which would silently change every node's memory.
        return nn.Module.train(self, mode)


class DeterministicNeighbors(LastNeighborLoader):
    def insert(self, src, dst):
        # Original dense_id assignment can collide when a node has >size edges
        # on a date. Select its last size canonical event IDs explicitly.
        s, d = src.cpu().numpy(), dst.cpu().numpy()
        event = np.arange(self.cur_e_id, self.cur_e_id + len(s))
        nodes, neighbors, ids = np.r_[s, d], np.r_[d, s], np.r_[event, event]
        order = np.lexsort((-ids, nodes))
        sorted_nodes = nodes[order]
        starts = np.maximum.accumulate(np.where(np.r_[True, np.diff(sorted_nodes) != 0],
                                               np.arange(len(order)), 0))
        rank = np.arange(len(order)) - starts
        keep = rank < self.size
        unique, group = np.unique(sorted_nodes[keep], return_inverse=True)
        event_table = np.full((len(unique), self.size), -1, dtype=np.int64)
        neighbor_table = np.zeros_like(event_table)
        event_table[group, rank[keep]] = ids[order[keep]]
        neighbor_table[group, rank[keep]] = neighbors[order[keep]]
        uid = torch.as_tensor(unique, device=src.device)
        new_ids = torch.as_tensor(event_table, device=src.device)
        new_neighbors = torch.as_tensor(neighbor_table, device=src.device)
        merged = torch.cat([self.e_id[uid], new_ids], dim=1)
        selected, perm = merged.topk(self.size, dim=1)
        self.e_id[uid] = selected
        self.neighbors[uid] = torch.gather(torch.cat([self.neighbors[uid], new_neighbors], dim=1), 1, perm)
        self.cur_e_id += len(s)


def tami_module():
    sys.path.insert(0, str(ROOT / 'baseline/vendor/python_dependencies'))
    path = ROOT / 'baseline/vendor/TAMI_temporal_graph-main/TAMI_temporal_graph-main/models/modules.py'
    spec = importlib.util.spec_from_file_location('patentflow_original_tami_modules', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LogTimeAdapter(nn.Module):
    def __init__(self, encoder, dim):
        super().__init__()
        self.encoder, self.out_channels = encoder, dim

    @property
    def lin(self):
        return self.encoder.w

    def forward(self, times):
        return self.encoder(times.reshape(-1, 1)).squeeze(1)


from .event_stream import EventStream


class TemporalForecast(nn.Module):
    def __init__(self, stream, method='TGN', dim=100, time_dim=32, neighbors=10, device='cpu', gamma=.9):
        super().__init__()
        self.stream, self.method, self.device_name = stream, method, device
        self.edge_types = method == 'TGN_edgetype'
        raw_dim = 108 if self.edge_types else 100
        self.memory = TransactionMemory(stream.node_count, raw_dim, dim, time_dim,
                                       IdentityMessage(raw_dim, dim, time_dim), MeanAggregator())
        if method == 'TAMI':
            original = tami_module()
            self.memory.time_enc = LogTimeAdapter(original.TimeEncoder(time_dim), time_dim)
            self.decoder = original.HistoricalDecoder(dim, dim, dim, 1, device=device, gamma=gamma)
        else:
            self.decoder = LinkPredictor(dim)
        self.embedding = GraphAttentionEmbedding(dim, dim, raw_dim, self.memory.time_enc)
        self.to(device)
        self.neighbors = DeterministicNeighbors(stream.node_count, neighbors, device=device)

    def reset(self):
        self.memory.reset_state()
        self.neighbors.reset_state()
        if self.method == 'TAMI':
            self.decoder.historical_interaction_memory.reset_memory()

    def propose(self, left, right):
        """Read-before-write differentiable update for one complete date."""
        device = self.memory.memory.device
        src = torch.as_tensor(self.stream.src[left:right], device=device)
        dst = torch.as_tensor(self.stream.dst[left:right], device=device)
        day = int(self.stream.day[left])
        msg = self.stream.messages(slice(left, right), device, self.edge_types)
        source, dest = torch.cat([src, dst]), torch.cat([dst, src])
        unique, inverse = source.unique(return_inverse=True)
        time = self.memory.time_enc((day - self.memory.last_update[source]).float())
        raw = self.memory.msg_s_module(self.memory.memory[source].detach(),
                   self.memory.memory[dest].detach(), torch.cat([msg, msg]), time)
        aggregate = self.memory.aggr_module(raw, inverse, source.new_full(source.shape, day), len(unique))
        updated = self.memory.memory_updater(aggregate, self.memory.memory[unique].detach())
        self.neighbors.insert(src, dst)
        return unique, updated, day

    def encode(self, nodes, pending=None):
        expanded, edge_index, event_ids = self.neighbors(nodes)
        states = self.memory.memory[expanded].detach()
        last = self.memory.last_update[expanded].clone()
        if pending is not None:
            ids, values, day = pending
            location = torch.searchsorted(ids, expanded)
            valid = (location < len(ids)) & (ids[location.clamp(max=len(ids)-1)] == expanded)
            states = states.clone()
            states[valid] = values[location[valid]]
            last[valid] = day
        event_np = event_ids.cpu().numpy()
        timestamps = torch.as_tensor(self.stream.day[event_np], device=nodes.device)
        messages = self.stream.messages(event_np, nodes.device, self.edge_types)
        embedding = self.embedding(states, last, edge_index, timestamps, messages)
        return embedding[torch.searchsorted(expanded, nodes)]

    @torch.no_grad()
    def commit(self, pending, left, right):
        ids, values, day = pending
        self.memory.memory[ids] = values.detach()
        self.memory.last_update[ids] = day
        if self.method == 'TAMI':
            # One actual pair per date. Duplicate patent edges at the same date
            # cannot introduce an arbitrary intra-date EMA order.
            pairs = np.unique(np.stack([self.stream.src[left:right], self.stream.dst[left:right]], axis=1), axis=0)
            device = self.memory.memory.device
            nodes = torch.as_tensor(np.unique(pairs), device=device)
            embeddings = self.encode(nodes)
            for offset in range(0, len(pairs), 2048):
                part = pairs[offset:offset+2048]
                s, d = part[:, 0], part[:, 1]
                src = embeddings[torch.searchsorted(nodes, torch.as_tensor(s.copy(), device=device))]
                dst = embeddings[torch.searchsorted(nodes, torch.as_tensor(d.copy(), device=device))]
                current = self.decoder.act(self.decoder.fc1(torch.cat([src, dst], dim=1)))
                historical = self._history(s, d, current.device)
                updated = self.decoder.aggregate_hist_emb(current, historical).cpu()
                memory = self.decoder.historical_interaction_memory.most_recent_hist_emb
                for key, value in zip(zip(s.tolist(), d.tolist()), updated):
                    memory[key] = value.clone()

    def _history(self, src, dst, device):
        memory = self.decoder.historical_interaction_memory.most_recent_hist_emb
        dim = self.decoder.fc1.out_features
        values = np.zeros((len(src), dim), dtype=np.float32)
        for i, key in enumerate(zip(src.tolist(), dst.tolist())):
            old = memory.get(key)
            if old is not None:
                values[i] = old.numpy()
        return torch.as_tensor(values, device=device)

    def score_embeddings(self, orgs, candidates, src, dst):
        if self.method != 'TAMI':
            hidden = (self.decoder.lin_src(src)[:, None, :] + self.decoder.lin_dst(dst)[None, :, :]).relu()
            return self.decoder.lin_final(hidden).squeeze(-1)
        dim = src.shape[1]
        fc = self.decoder.fc1
        current = (nn.functional.linear(src, fc.weight[:, :dim], fc.bias)[:, None, :] +
                   nn.functional.linear(dst, fc.weight[:, dim:])[None, :, :]).relu()
        s = np.repeat(np.asarray(orgs), len(candidates))
        d = np.tile(np.asarray(candidates), len(orgs))
        historical = self._history(s, d, src.device).reshape(len(orgs), len(candidates), dim)
        return self.decoder.fc2(torch.cat([current, historical], dim=-1)).squeeze(-1)

    def score(self, orgs, candidates, pending=None):
        device = self.memory.memory.device
        nodes = torch.as_tensor(np.r_[orgs, candidates], device=device)
        values = self.encode(nodes, pending)
        return self.score_embeddings(orgs, candidates, values[:len(orgs)], values[len(orgs):])

    @torch.no_grad()
    def replay(self, max_year):
        self.reset()
        for left, right, _ in self.stream.groups(max_year):
            self.commit(self.propose(left, right), left, right)

    def provenance(self):
        names = ['memory_module.py', 'msg_func.py', 'msg_agg.py', 'emb_module.py', 'decoder.py', 'neighbor_loader.py']
        paths = [TGB / 'modules' / name for name in names]
        if self.method == 'TAMI':
            paths.append(ROOT / 'baseline/vendor/TAMI_temporal_graph-main/TAMI_temporal_graph-main/models/modules.py')
        return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}
