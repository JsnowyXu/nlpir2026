"""Three-level EDGPAT adapter using original trainable modules and decoder algebra.

Run in its own process: the original EDGPAT and TGB projects both use `modules`.
"""
import sys
import numpy as np
import torch
from torch import nn
from .data import ROOT, file_hash

SOURCE = ROOT / 'baseline/EDGPAT-main'
sys.path.insert(0, str(SOURCE))
from model.EDGPAT import EDGPAT
from modules.message_function import get_message_function


class EDGPATForecast(nn.Module):
    def __init__(self, stream, dim=100, time_dim=32, device='cpu'):
        super().__init__()
        self.stream, self.device_name = stream, device
        data = stream.data
        leaf = data.codes
        self.types = ['company', 'field']
        levels = [leaf]
        if data.level == 3:
            levels.append(sorted({c[:3] for c in leaf})); self.types.append('class')
        if data.level >= 2:
            levels.append(sorted({c[0] for c in leaf})); self.types.append('section')
        self.reflect = []
        self.leaf_to_level = [np.arange(len(leaf))]
        self.parent_projection = []
        for i in range(1, len(levels)):
            lookup = {c: n for n, c in enumerate(levels[i])}
            width = len(levels[i][0])
            parent = np.array([lookup[c[:width]] for c in levels[i-1]])
            self.reflect.append(torch.as_tensor(np.c_[np.arange(len(parent)), parent]))
            leaf_parent = np.array([lookup[c[:width]] for c in leaf])
            self.leaf_to_level.append(leaf_parent)
            projection = np.zeros((len(leaf), len(levels[i])), dtype=np.float32)
            projection[np.arange(len(leaf)), leaf_parent] = 1
            self.parent_projection.append(torch.as_tensor(projection, device=device))
        counts = {'company': stream.org_count}
        counts.update({t: len(codes) for t, codes in zip(self.types[1:], levels)})
        self.original = EDGPAT(device, n_nodes=counts, type=self.types,
                              reflect=self.reflect, num_hier=counts,
                              memory_dimension=dim, message_dimension=dim,
                              time_dim=time_dim, use_time=True, embedding_module_type='identity')
        # Original has a five-level hard-coded arity list. Replace only the
        # top-level MLP, whose messages have no parent; other modules are unchanged.
        top = self.types[-1]
        self.original.message_function[top] = get_message_function('mlp', 2*dim+time_dim, dim, device)
        self.to(device)
        self.ever = np.zeros((stream.org_count, len(leaf)), dtype=bool)
        self.recent = np.zeros_like(self.ever)
        self.current_year = None

    @torch.no_grad()
    def reset(self):
        for value in self.original.memory.memory.values():
            value.zero_()
        for value in self.original.memory.last_update.values():
            value.zero_()
        self.ever.fill(False); self.recent.fill(False); self.current_year = None

    def propose(self, left, right):
        stream, original = self.stream, self.original
        device = self.device_name
        org = stream.src[left:right]
        field = stream.dst[left:right] - stream.org_count
        year, day = int(stream.year[left]), int(stream.day[left])
        if year != self.current_year:
            self.recent.fill(False); self.current_year = year
        self.ever[org, field] = True; self.recent[org, field] = True
        ids = [torch.as_tensor(org, device=device)]
        ids += [torch.as_tensor(mapping[field], device=device) for mapping in self.leaf_to_level]
        old = [original.memory.memory[t][idx].detach() for t, idx in zip(self.types, ids)]
        # Equal patent weight for company messages: first average its categories,
        # then average its patent messages, as in get_company_messages.
        _, inverse, count = np.unique(np.c_[org, stream.rows[left:right]], axis=0,
                                      return_inverse=True, return_counts=True)
        patent_weight = torch.as_tensor(1./count[inverse], dtype=torch.float32, device=device)
        result = {}
        for i, (kind, nodes) in enumerate(zip(self.types, ids)):
            parts = [old[i]]
            if i == 0:
                parts.append(old[1])
            else:
                parts.append(old[i-1])
                if i+1 < len(ids):
                    parts.append(old[i+1])
            delta = day-original.memory.last_update[kind][nodes]
            parts.append(original.time_encoder(delta[:, None]))
            messages = torch.cat(parts, dim=1)
            unique, group = nodes.unique(return_inverse=True)
            weights = patent_weight if i == 0 else torch.ones_like(patent_weight)
            sums = messages.new_zeros((len(unique), messages.shape[1]))
            sums.index_add_(0, group, messages*weights[:, None])
            mass = messages.new_zeros(len(unique)).index_add_(0, group, weights)
            message = original.message_function[kind].compute_message(sums/mass[:, None])
            updated = original.memory_updater.memory_updater(message, original.memory.memory[kind][unique].detach())
            result[kind] = (unique, updated)
        return result, day

    @torch.no_grad()
    def commit(self, pending, left, right):
        changes, day = pending
        for kind, (ids, values) in changes.items():
            self.original.memory.memory[kind][ids] = values.detach()
            self.original.memory.last_update[kind][ids] = day

    def read(self, kind, nodes, pending):
        values = self.original.memory.memory[kind][nodes].detach()
        if pending is not None:
            ids, updated = pending[0][kind]
            position = torch.searchsorted(ids, nodes)
            valid = (position < len(ids)) & (ids[position.clamp(max=len(ids)-1)] == nodes)
            values = values.clone(); values[valid] = updated[position[valid]]
        return values

    def score(self, orgs, candidates, pending=None):
        decoder = self.original.predict_patent
        device = self.device_name
        fields = candidates-self.stream.org_count
        fid = torch.as_tensor(fields, device=device)
        uid = torch.as_tensor(orgs, device=device)
        recent_all = torch.as_tensor(self.recent[orgs], device=device)
        now = recent_all[:, fid]
        historical = torch.as_tensor(self.ever[orgs][:, fields], device=device) & ~now
        seen = now | historical
        alpha = decoder.alpha_fields[fid]
        static = decoder.projection_layer(decoder.field_embedding(fid))
        current = self.read('field', fid, pending)
        old = decoder.mlp_his(self.original.memory.memory['field'][fid].detach())
        embedding = (1-seen[:, :, None]*alpha)*static
        embedding = embedding + now[:, :, None]*alpha*current + historical[:, :, None]*alpha*old
        if len(self.types) > 2:
            hierarchy = embedding.new_zeros(embedding.shape)
            for i, kind in enumerate(self.types[2:], 1):
                projection = self.parent_projection[i-1]
                recent_parents = (recent_all.float() @ projection) > 0
                parent = torch.as_tensor(self.leaf_to_level[i][fields], device=device)
                parent_now = recent_parents[:, parent]
                updated_parent = self.read(kind, parent, pending)
                old_parent = decoder.mlp[kind](self.original.memory.memory[kind][parent].detach())
                hierarchy = hierarchy + torch.where(parent_now[:, :, None], updated_parent[None], old_parent[None])
            gamma = decoder.gamma[fid]
            embedding = torch.where(seen[:, :, None], gamma*embedding+(1-gamma)*hierarchy, embedding)
        org_memory = self.read('company', uid, pending)
        theta = decoder.theta[uid]
        org_embedding = decoder.dropout((1-theta)*org_memory + theta*decoder.company_embedding(uid))
        return decoder.fc_field(embedding).squeeze(-1) + decoder.fc_company(org_embedding)

    @torch.no_grad()
    def replay(self, max_year):
        self.reset()
        for left, right, _ in self.stream.groups(max_year):
            self.commit(self.propose(left, right), left, right)
        # If this view has no interaction during the final calendar year, its
        # recent-year basket is empty, not the last active year's basket.
        if self.current_year != max_year:
            self.recent.fill(False)

    def provenance(self):
        paths = [SOURCE/'model/EDGPAT.py', SOURCE/'model/time_encoding.py']
        paths += [SOURCE/'modules'/name for name in ['memory.py', 'message_function.py', 'memory_updater.py']]
        return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}
