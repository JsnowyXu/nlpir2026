"""Vendored TGAT, DyGFormer and RepeatMixer with strict historical queries."""
from datetime import date
from functools import lru_cache
import importlib.util
import sys
from types import MethodType
import numpy as np
import torch
from torch import nn
from .data import ROOT, file_hash
from .event_stream import EventStream

SOURCE = ROOT/'baseline/vendor/RepeatMixer-main/RepeatMixer-main'
sys.path.insert(0, str(SOURCE))
# Load the installed original metric file, avoiding unrelated audio backends.
metric_path = ROOT/'baseline/vendor/python_dependencies/audtorch/metrics/functional.py'
spec = importlib.util.spec_from_file_location('audtorch.metrics.functional', str(metric_path))
metric_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metric_module)
sys.modules['audtorch.metrics.functional'] = metric_module
from utils.utils import NeighborSampler, HistoricalNeighborSampler
from models.TGAT import TGAT
from models.DyGFormer import DyGFormer
from models.RepeatMixer import RepeatMixer
from models.modules import MergeLayer, MergeSingleLayer


def vectorized_appearances(self, src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids):
    """Exactly the original integer counts, without per-node GPU transfers."""
    src=torch.as_tensor(src_padded_nodes_neighbor_ids,device=self.device)
    dst=torch.as_tensor(dst_padded_nodes_neighbor_ids,device=self.device)
    ss=(src[:,:,None]==src[:,None,:]).sum(dim=2)
    dd=(dst[:,:,None]==dst[:,None,:]).sum(dim=2)
    cross=src[:,:,None]==dst[:,None,:]
    source=torch.stack([ss,cross.sum(dim=2)],dim=-1).float()
    destination=torch.stack([cross.sum(dim=1),dd],dim=-1).float()
    source[src==0]=0.;destination[dst==0]=0.
    return source,destination


class CachedNeighbors(NeighborSampler):
    @lru_cache(maxsize=100000)
    def find_neighbors_before(self, node_id, interact_time, return_sampled_probabilities=False):
        return super().find_neighbors_before(node_id, interact_time, return_sampled_probabilities)


class CorrectedRepeatNeighbors(HistoricalNeighborSampler):
    @lru_cache(maxsize=50000)
    def find_neighbors_before(self, node_id, interact_node, interact_time, return_sampled_probabilities=False):
        if return_sampled_probabilities:
            return super().find_neighbors_before(node_id, interact_node, interact_time, True)
        stop = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time, side='left')
        matches = np.flatnonzero(self.nodes_neighbor_ids[node_id][:stop] == interact_node)[-10:]
        if not len(matches):
            selected = slice(0, stop)
        else:
            expand = min(stop//len(matches), 16)
            selected = (matches[:, None] + np.arange(-expand, 1)[None, :]).ravel()
            # Vendored code used total>=0, unintentionally discarding every
            # preceding context offset. The node positions must be nonnegative.
            selected = np.unique(selected[selected >= 0])
        return (self.nodes_neighbor_ids[node_id][selected], self.nodes_edge_ids[node_id][selected],
                self.nodes_neighbor_times[node_id][selected], None)


class IndexedRepeatNeighbors(CorrectedRepeatNeighbors):
    """Exact recent sampling, with indexed matches and bounded two-hop tails."""
    @lru_cache(maxsize=8192)
    def _neighbor_positions(self, node):
        ids = self.nodes_neighbor_ids[node]
        order = np.argsort(ids, kind='stable')
        unique, starts = np.unique(ids[order], return_index=True)
        ends = np.r_[starts[1:], len(order)]
        return {int(key): order[left:right] for key, left, right in zip(unique, starts, ends)}

    @lru_cache(maxsize=100000)
    def _stop(self, node, cutoff):
        return int(np.searchsorted(self.nodes_neighbor_times[node], cutoff, side='left'))

    def _context_tail(self, node, scope, cutoff, size):
        stop = self._stop(int(node), float(cutoff))
        if not stop:
            return np.empty(0, dtype=np.int64)
        if len(self.nodes_neighbor_ids[node]) <= 64:
            matches = np.flatnonzero(np.isin(self.nodes_neighbor_ids[node][:stop], scope))[-10:]
        else:
            positions = self._neighbor_positions(int(node))
            matches = []
            for other in np.unique(scope):
                found = positions.get(int(other))
                if found is not None:
                    end = np.searchsorted(found, stop)
                    matches.extend(found[max(end-10, 0):end])
            matches = np.sort(np.asarray(matches, dtype=np.int64))[-10:]
        if not len(matches):
            return np.arange(max(stop-size, 0), stop)
        expand = min(stop // len(matches), 16)
        selected = (matches[:, None] + np.arange(-expand, 1)[None, :]).ravel()
        return np.unique(selected[selected >= 0])[-size:]

    def _gather(self, node, positions):
        return tuple(values[node][positions] for values in
                     [self.nodes_neighbor_ids, self.nodes_edge_ids, self.nodes_neighbor_times])

    @lru_cache(maxsize=200000)
    def _two_hop_tail(self, node, scope, cutoff, size):
        # The same IPC/context scope recurs for many organization candidates
        # at an annual cutoff. Cache only this reusable second-order context.
        return self._context_tail(node, scope, cutoff, size)

    def get_historical_neighbors(self, node_ids, interact_nodes, node_interact_times, num_neighbors=20):
        if self.sample_neighbor_strategy != 'recent':
            return super().get_historical_neighbors(node_ids, interact_nodes, node_interact_times, num_neighbors)
        output = [np.zeros((len(node_ids), num_neighbors), dtype=dtype)
                  for dtype in [np.int64, np.int64, np.float32]]
        for i, (node, other, cutoff) in enumerate(zip(node_ids, interact_nodes, node_interact_times)):
            selected = self._context_tail(node, [other], cutoff, num_neighbors)
            if len(selected):
                for result, values in zip(output, self._gather(node, selected)):
                    result[i, -len(selected):] = values
        return tuple(output)

    def get_two_hop_historical_neighbors(self, node_ids, scopes, node_interact_times, num_neighbors=20):
        if self.sample_neighbor_strategy != 'recent':
            return super().get_two_hop_historical_neighbors(node_ids, scopes, node_interact_times, num_neighbors)
        output = [np.zeros((len(node_ids), num_neighbors), dtype=dtype)
                  for dtype in [np.int64, np.int64, np.float32]]
        for i, (node, scope, cutoff) in enumerate(zip(node_ids, scopes, node_interact_times)):
            stop = self._stop(int(node), float(cutoff))
            chunks, remaining = [], num_neighbors
            scope_key = tuple(sorted(set(scope.tolist())))
            # The source implementation concatenates five histories then keeps
            # the tail. Traverse backwards to avoid constructing discarded data.
            # Preserve duplicates and original concatenation order exactly.
            for neighbor in self.nodes_neighbor_ids[node][max(stop-5, 0):stop][::-1]:
                selected = self._two_hop_tail(int(neighbor), scope_key, float(cutoff), num_neighbors)[-remaining:]
                if len(selected):
                    chunks.append(self._gather(neighbor, selected))
                    remaining -= len(selected)
                if remaining == 0:
                    break
            if chunks:
                for j, result in enumerate(output):
                    values = np.concatenate([chunk[j] for chunk in chunks[::-1]])
                    result[i, -len(values):] = values
        return tuple(output)


class FixedPaddingDyGFormer(DyGFormer):
    def pad_sequences(self, node_ids, node_interact_times, nodes_neighbor_ids_list, nodes_edge_ids_list,
                      nodes_neighbor_times_list, patch_size=1, max_input_sequence_length=512):
        length = max_input_sequence_length
        length += (-length) % patch_size
        nodes = np.zeros((len(node_ids), length), dtype=np.int64)
        edges = np.zeros_like(nodes)
        times = np.zeros((len(node_ids), length), dtype=np.float32)
        nodes[:,0] = node_ids; times[:,0] = node_interact_times
        for i, (ns, es, ts) in enumerate(zip(nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list)):
            take = min(len(ns), max_input_sequence_length-1)
            if take:
                nodes[i,1:take+1] = ns[-take:]
                edges[i,1:take+1] = es[-take:]
                times[i,1:take+1] = ts[-take:]
        return nodes, edges, times


class LazyEdgeText:
    """Exact feature gather with one batched transfer instead of a full GPU copy."""
    def __init__(self, stream, device):
        self.stream, self.device = stream, device
        self.shape = (len(stream.rows)+1, 100)

    def __getitem__(self, indices):
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        indices = np.asarray(indices)
        result = self.stream.vectors[self.stream.rows[np.maximum(indices-1,0)]].copy()
        result[indices == 0] = 0
        return torch.as_tensor(result, device=self.device)


class SamplingForecast(nn.Module):
    def __init__(self, data, method, device='cpu', neighbors=10, layers=2, time_dim=32,
                 sequence_length=32, channels=32, patch=2, high_order=True, seed=42, dropout=.1):
        super().__init__()
        self.stream = EventStream(data)
        self.data, self.method, self.device_name, self.neighbors = data, method, device, neighbors
        stream = self.stream
        self.src = stream.src+1; self.dst = stream.dst+1
        self.times = (stream.day+stream.origin_day-1).astype(np.float64)*86400.
        adj = [[] for _ in range(stream.node_count+1)]
        for eid, (src, dst, timestamp) in enumerate(zip(self.src, self.dst, self.times), 1):
            adj[src].append((dst,eid,timestamp));adj[dst].append((src,eid,timestamp))
        sampler = CachedNeighbors(adj, sample_neighbor_strategy='recent', seed=seed)
        del adj
        historical = IndexedRepeatNeighbors.__new__(IndexedRepeatNeighbors)
        historical.__dict__.update(sampler.__dict__)
        self.sampler, self.historical_sampler = sampler, historical
        features = np.zeros((stream.node_count+1,100), dtype=np.float32)
        anchors = np.load(ROOT/'data/text'/data.spec['country']/'taxonomy_anchors.npz')
        lookup = dict(zip(anchors['codes'], anchors['anchors']))
        for i, code in enumerate(data.codes):
            features[stream.org_count+1+i] = lookup[code]
        common = dict(node_raw_features=features, edge_raw_features=np.zeros((1,100),dtype=np.float32),
                      neighbor_sampler=sampler, time_feat_dim=time_dim, num_layers=layers, dropout=dropout, device=device)
        if method == 'TGAT':
            self.backbone = TGAT(**common, num_heads=2)
        elif method == 'DyGFormer':
            self.backbone = FixedPaddingDyGFormer(**common, channel_embedding_dim=channels, patch_size=patch,
                                                 max_input_sequence_length=sequence_length, num_heads=2)
            encoder=self.backbone.neighbor_co_occurrence_encoder
            encoder.count_nodes_appearances=MethodType(vectorized_appearances,encoder)
        elif method == 'RepeatMixer':
            self.backbone = RepeatMixer(**common, edge_neighbor_sampler=historical,
                                        num_tokens=neighbors, high_order=high_order)
        else:
            raise ValueError(method)
        self.backbone.edge_raw_features = LazyEdgeText(stream, device)
        self.decoder = (MergeSingleLayer(100,50,1) if method=='RepeatMixer' else MergeLayer(100,100,0,100,1))
        self.to(device)

    @staticmethod
    def cutoff(target_year):
        return float((date(target_year,1,1)-date(1970,1,1)).days*86400)

    def pairs(self, orgs, fields, target_year):
        src = np.asarray(orgs,dtype=np.int64)+1
        dst = np.asarray(fields,dtype=np.int64)+self.stream.org_count+1
        times = np.full(len(src),self.cutoff(target_year),dtype=np.float64)
        if self.method == 'RepeatMixer':
            pair = self.backbone.compute_src_dst_node_temporal_embeddings(src,dst,times,num_neighbors=self.neighbors)
            return self.decoder(pair,None,None).squeeze(-1)
        if self.method == 'TGAT':
            source, dest = self.backbone.compute_src_dst_node_temporal_embeddings(src,dst,times,num_neighbors=self.neighbors)
        else:
            source, dest = self.backbone.compute_src_dst_node_temporal_embeddings(src,dst,times)
        return self.decoder(source,dest,None).squeeze(-1)

    @torch.no_grad()
    def prepare_tgat_evaluation(self, year, batch_size=128):
        orgs = self.data.queries(year)[0]
        candidates = self.data.candidate_ids
        nodes = np.r_[orgs+1,candidates+self.stream.org_count+1]
        embeddings=[]
        for offset in range(0,len(nodes),batch_size):
            batch=nodes[offset:offset+batch_size]
            embeddings.append(self.backbone.compute_node_temporal_embeddings(batch,
                np.full(len(batch),self.cutoff(year)),self.backbone.num_layers,self.neighbors).cpu())
        embeddings=torch.cat(embeddings)
        dst=embeddings[len(orgs):].to(self.device_name)
        fc=self.decoder.fc1
        right=nn.functional.linear(dst,fc.weight[:,100:])
        def score(query):
            with torch.no_grad():
                src=embeddings[np.searchsorted(orgs,query)].to(self.device_name)
                left=nn.functional.linear(src,fc.weight[:,:100],fc.bias)
                return self.decoder.fc2((left[:,None,:]+right[None,:,:]).relu()).squeeze(-1).cpu().numpy()
        return score

    def provenance(self):
        paths=[SOURCE/'models'/(self.method+'.py'),SOURCE/'models/modules.py',SOURCE/'utils/utils.py']
        if self.method=='RepeatMixer':paths.append(metric_path)
        return {str(p.relative_to(ROOT)):file_hash(p) for p in paths}
