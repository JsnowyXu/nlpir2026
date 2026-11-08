"""Modular anchored temporal memory forecaster. Not a baseline impersonation."""

import math
import torch
from torch import nn
from torch.nn import functional as F


def projection(input_dim, dim, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Dropout(dropout)
    )


class PatentFlow(nn.Module):
    def __init__(
        self,
        anchors,
        dim=100,
        communities=32,
        dropout=0.1,
        disabled=(),
        temperature=0.5,
    ):
        super().__init__()
        self.disabled = set(disabled)
        self.temperature = temperature
        self.register_buffer(
            "raw_anchors", torch.as_tensor(anchors, dtype=torch.float32)
        )
        self.semantic = projection(200, dim, dropout)
        self.structural = projection(200, dim, dropout)
        self.ipc_dynamic = projection(400, dim, dropout)
        self.anchor = nn.Linear(100, dim, bias=False)
        self.ipc_structural = nn.Parameter(torch.randn(len(anchors), dim) * 0.02)
        self.community_query = nn.Linear(200, dim, bias=False)
        self.prototypes = nn.Parameter(
            F.normalize(torch.randn(communities, dim), dim=-1)
        )
        self.register_buffer("community_bank", torch.zeros(communities, 400))
        self.community_encode = projection(400, dim, dropout)
        self.fusion = nn.Linear(8, 3)
        self.cooperation_gate = nn.Linear(8, 1)
        self.community_gate = nn.Linear(8, 1)
        self.repeat = nn.Sequential(
            nn.LayerNorm(16), nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.repeat_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.ipc_bias = nn.Parameter(torch.zeros(len(anchors)))

    @staticmethod
    def channels(raw):
        # raw=[fast(text100,structure100), slow(text100,structure100)]
        text = torch.cat([raw[:, :100], raw[:, 200:300]], dim=-1)
        structure = torch.cat([raw[:, 100:200], raw[:, 300:400]], dim=-1)
        return text, structure

    def route(self, structure):
        query = F.normalize(self.community_query(structure), dim=-1)
        logits = (
            query @ F.normalize(self.prototypes, dim=-1).T / max(self.temperature, 1e-3)
        )
        values, indices = logits.topk(min(3, logits.shape[1]), dim=-1)
        return indices, F.softmax(values, dim=-1)

    def mask_channels(self, raw):
        mask = torch.ones(400, device=raw.device)
        if "text" in self.disabled:
            mask[:100] = 0
            mask[200:300] = 0
        if "structure" in self.disabled:
            mask[100:200] = 0
            mask[300:400] = 0
        return raw * mask

    @torch.no_grad()
    def refresh_communities(self, raw):
        """Population context from strictly past node states, never target labels.

        Refresh once at a year boundary/epoch; query router remains differentiable.
        This avoids an org-by-org all-pairs graph and dense one-hot routing matrix.
        """
        raw = self.mask_channels(raw)
        _, structure = self.channels(raw)
        ids, weights = self.route(structure)
        valid = (structure.norm(dim=-1) > 0).float()
        weights = weights * valid[:, None]
        sums = torch.zeros_like(self.community_bank)
        masses = torch.zeros(len(sums), device=raw.device)
        sums.index_add_(
            0,
            ids.flatten(),
            (raw[:, None, :] * weights[:, :, None]).reshape(-1, raw.shape[1]),
        )
        masses.index_add_(0, ids.flatten(), weights.flatten())
        self.community_bank.copy_(sums / masses.clamp_min(1e-8)[:, None])

    def forward(
        self, org_raw, ipc_raw, peer_raw, peer_mask, reliability, pair_features
    ):
        org_raw, ipc_raw, peer_raw = [
            self.mask_channels(v) for v in [org_raw, ipc_raw, peer_raw]
        ]
        reliability = reliability.clone()
        if "text" in self.disabled:
            reliability[:, 3] = 0
        if "structure" in self.disabled:
            reliability[:, 4] = 0
        if "cooperation" in self.disabled:
            reliability[:, 5] = 0
        text, structure = self.channels(org_raw)
        sem = F.normalize(self.semantic(text), dim=-1)
        structural = F.normalize(self.structural(structure), dim=-1)
        static = F.normalize(self.anchor(self.raw_anchors), dim=-1)
        dynamic = F.normalize(self.ipc_dynamic(ipc_raw), dim=-1)
        if "hierarchy" in self.disabled:
            static = static * 0
        ipc = (
            dynamic
            if "anchor" in self.disabled
            else F.normalize(static + 0.5 * dynamic, dim=-1)
        )
        scale = self.logit_scale.exp().clamp(max=30)
        semantic_score = sem @ ipc.T
        structural_score = structural @ F.normalize(self.ipc_structural, dim=-1).T
        hierarchy_score = F.normalize(sem + structural, dim=-1) @ static.T
        branches = torch.stack(
            [semantic_score, structural_score, hierarchy_score], dim=-1
        )
        mask = torch.ones(3, device=org_raw.device)
        for index, name in enumerate(["text", "structure", "hierarchy"]):
            if name in self.disabled:
                mask[index] = 0
        weights = F.softmax(
            self.fusion(reliability).masked_fill(mask[None, :] == 0, -1e9), dim=-1
        )
        weights = weights * mask[None, :]
        logits = (branches * weights[:, None, :]).sum(dim=-1) * scale
        if "cooperation" not in self.disabled:
            # Gate is exactly zero without a valid historical partner state.
            peer_text, peer_structure = self.channels(peer_raw.reshape(-1, 400))
            peers = F.normalize(
                self.semantic(peer_text) + self.structural(peer_structure), dim=-1
            )
            peers = peers.reshape(len(org_raw), 3, -1)
            peer_mean = (peers * peer_mask[:, :, None]).sum(dim=1) / peer_mask.sum(
                dim=1
            ).clamp_min(1.0)[:, None]
            gate = (
                torch.sigmoid(self.cooperation_gate(reliability))
                * (peer_mask.sum(dim=1) > 0)[:, None]
            )
            logits = logits + gate * (F.normalize(peer_mean, dim=-1) @ ipc.T) * scale
        if "community" not in self.disabled and "structure" not in self.disabled:
            ids, route_weights = self.route(structure)
            context = (self.community_bank[ids] * route_weights[:, :, None]).sum(dim=1)
            community = F.normalize(self.community_encode(context), dim=-1)
            available = (context.norm(dim=-1) > 0)[:, None]
            logits = (
                logits
                + torch.sigmoid(self.community_gate(reliability))
                * available
                * (community @ ipc.T)
                * scale
            )
        if "repeat" not in self.disabled:
            logits = logits + self.repeat_scale * self.repeat(pair_features).squeeze(-1)
        return logits + self.ipc_bias[None, :]

    def anchor_regularization(self, ipc_raw):
        if {"alignment", "anchor", "hierarchy"} & self.disabled:
            return self.ipc_structural.sum() * 0
        ipc_raw = self.mask_channels(ipc_raw)
        raw = F.normalize(self.raw_anchors, dim=-1)
        static = F.normalize(self.anchor(self.raw_anchors), dim=-1)
        dynamic = F.normalize(static + 0.5 * self.ipc_dynamic(ipc_raw), dim=-1)
        # Match pairwise geometry, not identical vectors for all taxonomy members.
        return F.mse_loss(dynamic @ dynamic.T, raw @ raw.T)

    def export_embeddings(self, org_raw, ipc_raw):
        org_raw, ipc_raw = self.mask_channels(org_raw), self.mask_channels(ipc_raw)
        text, structure = self.channels(org_raw)
        org = F.normalize(self.semantic(text) + self.structural(structure), dim=-1)
        static = self.anchor(self.raw_anchors)
        if {"hierarchy", "anchor"} & self.disabled:
            static = static * 0
        ipc = F.normalize(static + 0.5 * self.ipc_dynamic(ipc_raw), dim=-1)
        return org, ipc
