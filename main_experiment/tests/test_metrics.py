import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from patentflow_main.metrics import ranking_metrics


def test_hand_computed_ranking_and_oov():
    # relevant ranks 2,4; one additional true label outside candidate catalog.
    r = ranking_metrics([[4.0, 3.0, 2.0, 1.0]], [{1, 3, 99}], ks=(2, 4, 10))
    assert r["active"]["mrr_4"] == 0.5
    assert r["active"]["recall_4"] == 2 / 3
    assert r["active"]["precision_4"] == 0.5
    assert r["active"]["ap_4"] == (1 / 2 + 2 / 4) / 3
    ideal = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert np.isclose(r["active"]["ndcg_4"], (1 / np.log2(3) + 1 / np.log2(5)) / ideal)
    assert r["effective_k"]["10"] == 4
    assert r["oov_positive_count"] == 1


def test_empty_targets_do_not_silently_change_active_denominator():
    r = ranking_metrics([[1.0, 0.0], [1.0, 0.0]], [{0}, set()], ks=(1,))
    assert r["n_empty"] == 1
    assert r["active"]["phr_1"] == 1.0
    assert r["all_zero_filled"]["phr_1"] == 0.5


def test_tie_break_is_stable_and_ids_can_be_noncontiguous():
    r = ranking_metrics([[0.0, 0.0]], [{17}], ks=(1,), candidate_ids=[17, 29])
    assert r["active"]["recall_1"] == 1


def test_no_queries():
    r = ranking_metrics(np.zeros((0, 3)), [], ks=(1,))
    assert r["active"]["ap_1"] is None
