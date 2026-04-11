# -*- coding: utf-8 -*-
"""
test_determinism.py -- reproducibility guarantee.

The styxx centroid file (atlas_v0.3.json) is sha256-pinned. The
classifier is deterministic (z-score + nearest-centroid, no
randomness). Given identical input trajectories, styxx MUST return
identical vitals on every machine, every Python version, every run.

If any of these assertions fail, styxx has a reproducibility bug
and should not ship in that state.

These tests are the product's defense against "oh it worked on my
machine but not theirs." Run them in CI on every push.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the parent dir importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from styxx import Raw
from styxx.vitals import (
    CATEGORIES,
    EXPECTED_CENTROIDS_SHA256,
    _compute_sha256,
    _default_centroids_path,
    load_centroids,
    extract_features,
)


# ══════════════════════════════════════════════════════════════════
# 1. The centroids file exists and matches its pinned sha256
# ══════════════════════════════════════════════════════════════════

def test_centroids_file_exists():
    path = _default_centroids_path()
    assert path.exists(), f"centroids file missing at {path}"


def test_centroids_sha256_matches():
    path = _default_centroids_path()
    actual = _compute_sha256(path)
    assert actual == EXPECTED_CENTROIDS_SHA256, (
        f"centroid sha256 mismatch.\n"
        f"  expected: {EXPECTED_CENTROIDS_SHA256}\n"
        f"  actual:   {actual}\n"
        "The shipped calibration data has been modified."
    )


# ══════════════════════════════════════════════════════════════════
# 2. load_centroids refuses to load a tampered file
# ══════════════════════════════════════════════════════════════════

def test_load_centroids_verifies_sha(tmp_path):
    path = tmp_path / "fake_centroids.json"
    with open(path, "w") as f:
        json.dump({"bogus": True}, f)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_centroids(path=path, verify_sha=True)


def test_load_centroids_accepts_verify_off(tmp_path):
    path = tmp_path / "fake_centroids.json"
    with open(path, "w") as f:
        json.dump({
            "categories": list(CATEGORIES),
            "phases": {},
        }, f)
    # With verify_sha=False, we bypass the check (used only in tests)
    result = load_centroids(path=path, verify_sha=False)
    assert result["categories"] == list(CATEGORIES)


# ══════════════════════════════════════════════════════════════════
# 3. Feature extraction is deterministic and shape-stable
# ══════════════════════════════════════════════════════════════════

def _fake_trajectory(n: int) -> dict:
    return {
        "entropy":     [1.5 + (i % 3) * 0.1 for i in range(n)],
        "logprob":     [-0.5 - (i % 4) * 0.05 for i in range(n)],
        "top2_margin": [0.5 + (i % 2) * 0.05 for i in range(n)],
    }


def test_feature_vector_shape():
    traj = _fake_trajectory(30)
    feats = extract_features(traj, n_tokens=25)
    # 4 stats (mean, std, min, max) × 3 signals = 12 features
    assert feats.shape == (12,), f"expected 12 features, got {feats.shape}"


def test_feature_extraction_deterministic():
    traj = _fake_trajectory(30)
    a = extract_features(traj, n_tokens=25)
    b = extract_features(traj, n_tokens=25)
    assert np.array_equal(a, b)


def test_feature_extraction_window_respected():
    traj = _fake_trajectory(30)
    f_all = extract_features(traj, n_tokens=30)
    f_first = extract_features(traj, n_tokens=1)
    # Window of 1 should have std = 0 for every signal
    # Layout: [mean_e, std_e, min_e, max_e, mean_l, std_l, ...]
    assert f_first[1] == 0.0, "std with n=1 must be 0"
    assert f_first[5] == 0.0
    assert f_first[9] == 0.0
    # Different windows should give different features
    assert not np.array_equal(f_all, f_first)


# ══════════════════════════════════════════════════════════════════
# 4. End-to-end determinism via the Raw adapter
# ══════════════════════════════════════════════════════════════════

def test_raw_adapter_deterministic():
    adapter = Raw()
    traj = _fake_trajectory(25)
    v1 = adapter.read(**traj)
    v2 = adapter.read(**traj)
    # Phase 1 classifier output must match bit-for-bit
    assert v1.phase1_pre.predicted_category == v2.phase1_pre.predicted_category
    assert v1.phase1_pre.confidence == v2.phase1_pre.confidence
    # Phase 4 too
    assert v1.phase4_late is not None
    assert v1.phase4_late.predicted_category == v2.phase4_late.predicted_category
    assert v1.phase4_late.confidence == v2.phase4_late.confidence


def test_raw_adapter_returns_tier_0():
    """v0.1 uses the tier 0 classifier regardless of environment."""
    adapter = Raw()
    vitals = adapter.read(**_fake_trajectory(25))
    assert vitals.tier_active == 0, (
        f"v0.1 runtime must report tier 0 in vitals, got {vitals.tier_active}"
    )


def test_raw_adapter_requires_aligned_lengths():
    adapter = Raw()
    with pytest.raises(ValueError, match="aligned trajectories"):
        adapter.read(
            entropy=[1.0, 2.0, 3.0],
            logprob=[-0.5, -0.4],
            top2_margin=[0.5, 0.5, 0.5],
        )


def test_raw_adapter_phase_progression():
    """Phases should light up as trajectory grows past each cutoff."""
    adapter = Raw()

    # Only phase 1 can be computed with 1 token
    short = _fake_trajectory(1)
    v_short = adapter.read(**short)
    assert v_short.phase1_pre is not None
    assert v_short.phase2_early is None
    assert v_short.phase3_mid is None
    assert v_short.phase4_late is None

    # Phase 2 lights up at 2+ tokens
    med = _fake_trajectory(6)
    v_med = adapter.read(**med)
    assert v_med.phase1_pre is not None
    assert v_med.phase2_early is not None
    assert v_med.phase3_mid is None
    assert v_med.phase4_late is None

    # Full phase coverage at 16+ tokens
    full = _fake_trajectory(25)
    v_full = adapter.read(**full)
    assert v_full.phase1_pre is not None
    assert v_full.phase2_early is not None
    assert v_full.phase3_mid is not None
    assert v_full.phase4_late is not None


# ══════════════════════════════════════════════════════════════════
# 5. Classifier outputs sum to 1 and pick a valid category
# ══════════════════════════════════════════════════════════════════

def test_phase_reading_probs_sum_to_one():
    adapter = Raw()
    vitals = adapter.read(**_fake_trajectory(25))
    for reading in (vitals.phase1_pre, vitals.phase2_early,
                    vitals.phase3_mid, vitals.phase4_late):
        if reading is None:
            continue
        total = sum(reading.probs.values())
        assert abs(total - 1.0) < 1e-9, f"probs sum = {total}, expected 1.0"


def test_phase_reading_prediction_is_valid_category():
    adapter = Raw()
    vitals = adapter.read(**_fake_trajectory(25))
    for reading in (vitals.phase1_pre, vitals.phase2_early,
                    vitals.phase3_mid, vitals.phase4_late):
        if reading is None:
            continue
        assert reading.predicted_category in CATEGORIES


# ══════════════════════════════════════════════════════════════════
# 6. Vitals.as_dict is json-serializable (for the agent)
# ══════════════════════════════════════════════════════════════════

def test_vitals_as_dict_json_roundtrip():
    adapter = Raw()
    vitals = adapter.read(**_fake_trajectory(25))
    as_dict = vitals.as_dict()
    # Should serialize and round-trip cleanly
    text = json.dumps(as_dict)
    back = json.loads(text)
    assert back["phase1_pre"]["predicted_category"] == vitals.phase1_pre.predicted_category
