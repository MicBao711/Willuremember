import json

import numpy as np
import pytest

from vcm.eskin import drt, synth


@pytest.fixture(scope="module")
def specimens(tmp_path_factory):
    root = tmp_path_factory.mktemp("eskin")
    out = {}
    for k, mode in enumerate(("none", "gain", "nonlinear")):
        synth.make_specimen(root / mode, f"syn-{mode}", days=(0, 30), drift=mode, seed=k)
        out[mode] = drt.load_specimen(root / mode)
    return out


def test_reference_order_is_balanced_and_each_round_covers_the_array():
    keys, counts = np.unique(drt.REF_ORDER, axis=0, return_counts=True)
    assert len(drt.REF_ORDER) == 96 and len(keys) == 48 and (counts == 2).all()
    for r in range(6):
        rnd = drt.REF_ORDER[16 * r:16 * (r + 1)]
        assert sorted(rnd[:, 0]) == list(range(16)) and len(set(rnd[:, 1])) == 1


def test_synthetic_sessions_conform(specimens):
    for sp in specimens.values():
        assert all(drt.validate(s) == [] for s in sp)


def test_validator_catches_broken_files(specimens, tmp_path):
    s = specimens["none"][0]
    a = dict(s.a)
    ref = s.rows("ref")
    a["pos"] = a["pos"].copy()
    a["pos"][ref[[0, 1]]] = a["pos"][ref[[1, 0]]]            # two reference presses swapped
    meta = {k: v for k, v in s.meta.items() if k != "temp_C"}
    errs = drt.validate(drt.Session(s.path, meta, a))
    assert any("temp_C" in e for e in errs) and any("Appendix A" in e for e in errs)
    a2 = dict(s.a, x=s.a["x"][:, :, :3])
    assert any("[N, F, 4, 4, T]" in e for e in drt.validate(drt.Session(s.path, s.meta, a2)))


def test_recal_map_recovers_an_affine_change_without_ridge():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(40, 5))
    g, b = rng.uniform(0.5, 2, 5), rng.normal(size=5)
    gh, bh = drt.recal_map((v - b) / g, v, ridge=0.0)
    assert np.allclose(gh, g) and np.allclose(bh, b)
    g0, b0 = drt.recal_map(v[:0], v[:0])
    assert np.allclose(g0, 1) and np.allclose(b0, 0)


def test_shrinkage_lda_separates_clear_classes():
    rng = np.random.default_rng(1)
    y = np.repeat(np.arange(3), 30)
    X = rng.normal(size=(90, 4)) * 0.3 + np.eye(3, 4)[y] * 3
    assert np.mean(drt.ShrinkageLDA(X, y, 3).predict(X) == y) == 1.0


def test_no_drift_keeps_the_decoder(specimens):
    rep = drt.evaluate_specimen(specimens["none"])
    assert rep["decodable"]
    s = rep["sessions"][0]
    assert abs(s["R"][0] - 1) < 0.05 and s["n_star"] == 0


def test_gain_drift_is_lost_without_and_recovered_with_reference_presses(specimens):
    s = drt.evaluate_specimen(specimens["gain"])["sessions"][0]
    assert s["R"][0] < 0.85 and s["R"][96] >= drt.RETAIN and 0 < s["n_star"] <= 96


def test_nonlinear_drift_is_not_recovered_by_affine_recalibration(specimens):
    s = drt.evaluate_specimen(specimens["nonlinear"])["sessions"][0]
    assert s["R"][96] < drt.RETAIN and s["n_star"] is None


def test_evaluation_is_deterministic_and_serialisable(specimens):
    a, b = drt.evaluate_specimen(specimens["gain"]), drt.evaluate_specimen(specimens["gain"])
    assert json.dumps(a) == json.dumps(b)
    cross = drt.evaluate_cross([specimens["none"], specimens["gain"]])
    assert cross["pairs"] == 2 and set(cross["R"]) == set(drt.RECAL_N)
    assert "syn-gain" in drt.report_md([a], cross)
