import json
import numpy as np
import torch
from .features import PairFeatures
from .io import ROOT


class MemoryFeatures:
    def __init__(self, data, cutoff, device, fast=180, slow=1095, novelty=1.0):
        self.data, self.device = data, device
        country = data.spec["country"]
        slug = (
            country
            if data.spec["section"] == "All"
            else country + "_" + data.spec["section"]
        )
        self.path = (
            ROOT / "data/memory" / slug / ("f%d_s%d_n%g" % (fast, slow, novelty))
        )
        snap = np.load(self.path / ("%d.npz" % cutoff))
        self.org_raw = np.concatenate([snap["org_fast"], snap["org_slow"]], axis=1)
        catalog = json.loads((data.path / "subclasses.json").read_text())
        selected = set(data.spec["subclass_ids"])
        code_to_id = {c: i for i, c in enumerate(catalog)}
        field_raw = np.concatenate([snap["ipc_fast"], snap["ipc_slow"]], axis=1)
        ipc = []
        width = {1: 1, 2: 3, 3: 4}[data.level]
        for code in data.codes:
            members = [
                i for i, c in enumerate(catalog) if i in selected and c[:width] == code
            ]
            weights = snap["ipc_slow_mass"][members]
            if weights.sum() > 0:
                ipc.append(
                    (field_raw[members] * weights[:, None]).sum(axis=0) / weights.sum()
                )
            else:
                ipc.append(np.zeros(400, dtype=np.float32))
        self.ipc_raw = torch.as_tensor(np.stack(ipc)[data.candidate_ids], device=device)
        self.peers = data.collaborators(cutoff)
        f, s = snap["org_fast"], snap["org_slow"]

        def cosine(a, b):
            return (a * b).sum(axis=1) / (
                np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
            )

        self.pairs = PairFeatures(data, cutoff)
        self.reliability = np.stack(
            [
                np.log1p(snap["org_fast_mass"]) / 10,
                np.log1p(snap["org_slow_mass"]) / 10,
                np.minimum(snap["org_slow_age"] / 3650.0, 3.0),
                cosine(f[:, :100], s[:, :100]),
                cosine(f[:, 100:], s[:, 100:]),
                (self.peers >= 0).sum(axis=1) / 3.0,
                np.log1p(self.pairs.total) / 10.0,
                np.maximum(cutoff - data.first_year + 1, 0) / 20.0,
            ],
            axis=1,
        ).astype(np.float32)

    def batch(self, orgs):
        peers = self.peers[orgs]
        peer_raw = self.org_raw[np.maximum(peers, 0)]
        peer_mask = (peers >= 0) & (np.linalg.norm(peer_raw, axis=-1) > 0)
        candidates = self.data.candidate_ids
        pair = self.pairs.pairs(
            np.repeat(orgs, len(candidates)), np.tile(candidates, len(orgs))
        )
        pair = pair.reshape(len(orgs), len(candidates), -1)
        return tuple(
            torch.as_tensor(x, device=self.device)
            for x in [
                self.org_raw[orgs],
                self.ipc_raw,
                peer_raw,
                peer_mask.astype(np.float32),
                self.reliability[orgs],
                pair,
            ]
        )

    def refresh(self, model):
        # Unknown/future organizations have exactly zero states; router masks them.
        model.refresh_communities(torch.as_tensor(self.org_raw, device=self.device))


def anchors_for(data):
    arr = np.load(ROOT / "data/text" / data.spec["country"] / "taxonomy_anchors.npz")
    lookup = dict(zip(arr["codes"], arr["anchors"]))
    return np.stack([lookup[data.codes[i]] for i in data.candidate_ids])
