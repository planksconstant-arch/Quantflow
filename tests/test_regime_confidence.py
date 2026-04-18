"""Tests for HMM regime confidence calibration."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from models.ml.regime_detector import RegimeDetector


def test_calibrated_confidence_penalizes_synthetic_data():
    probs = np.array([0.9, 0.05, 0.03, 0.02])
    clean = RegimeDetector._calibrated_confidence(
        state_probs=probs,
        state_idx=0,
        is_synthetic=False,
        realized_vol=0.20,
        regime_label="High Vol Bull",
    )
    synthetic = RegimeDetector._calibrated_confidence(
        state_probs=probs,
        state_idx=0,
        is_synthetic=True,
        realized_vol=0.20,
        regime_label="High Vol Bull",
    )
    assert synthetic < clean


def test_calibrated_confidence_penalizes_low_vol_label_with_high_realized_vol():
    probs = np.array([0.85, 0.10, 0.03, 0.02])
    normal = RegimeDetector._calibrated_confidence(
        state_probs=probs,
        state_idx=0,
        is_synthetic=False,
        realized_vol=0.22,
        regime_label="Low Vol Bull",
    )
    contradictory = RegimeDetector._calibrated_confidence(
        state_probs=probs,
        state_idx=0,
        is_synthetic=False,
        realized_vol=0.55,
        regime_label="Low Vol Bull",
    )
    assert contradictory < normal
    assert 0.05 <= contradictory <= 0.95
