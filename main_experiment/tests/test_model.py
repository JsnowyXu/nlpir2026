import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from patentflow_main.model import PatentFlow


def test_real_gradient_paths_and_empty_collaboration_mask():
    torch.manual_seed(42)
    model = PatentFlow(
        np.random.RandomState(1).randn(8, 100).astype("float32"),
        communities=4,
        dropout=0,
    )
    org = torch.randn(5, 400)
    ipc = torch.randn(8, 400)
    peers = torch.randn(5, 3, 400)
    mask = torch.zeros(5, 3)
    reliability = torch.randn(5, 8)
    pairs = torch.randn(5, 8, 16)
    model.refresh_communities(org)
    logits = model(org, ipc, peers, mask, reliability, pairs)
    changed = model(org, ipc, peers * 1000, mask, reliability, pairs)
    assert torch.allclose(logits, changed)
    loss = logits.square().mean() + model.anchor_regularization(ipc)
    loss.backward()
    for parameter in [
        model.semantic[0].weight,
        model.prototypes,
        model.community_query.weight,
        model.repeat[1].weight,
        model.anchor.weight,
    ]:
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert logits.shape == (5, 8) and torch.isfinite(logits).all()


def test_disabled_cooperation_removes_peer_vectors_and_count():
    torch.manual_seed(7)
    model = PatentFlow(
        np.random.RandomState(4).randn(8, 100).astype("float32"),
        communities=4,
        dropout=0,
        disabled=["cooperation"],
    )
    model.eval()
    org = torch.randn(5, 400)
    ipc = torch.randn(8, 400)
    peers = torch.randn(5, 3, 400)
    mask = torch.ones(5, 3)
    reliability = torch.randn(5, 8)
    pairs = torch.randn(5, 8, 16)
    model.refresh_communities(org)
    before = model(org, ipc, peers, mask, reliability, pairs)
    changed = reliability.clone()
    changed[:, 5] = 1000
    after = model(org, ipc, peers * 1000, mask * 0, changed, pairs)
    assert torch.equal(before, after)
