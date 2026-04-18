"""
Signal quality and execution-aware validation utilities.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.stats import norm


@dataclass
class SignalInputs:
    market_price: float
    forecast_fair_value: float
    bid: float
    ask: float
    mc_ci: Tuple[float, float]
    confidence: float = 0.5
    commission_per_contract: float = 0.65
    contract_multiplier: int = 100


class SignalQualityEngine:
    """
    Converts raw mispricing into execution-aware, statistically interpretable diagnostics.
    """

    def evaluate(self, inputs: SignalInputs) -> Dict:
        if inputs.market_price <= 0:
            return self._empty("invalid_market_price")

        divergence = inputs.forecast_fair_value - inputs.market_price
        divergence_pct = (divergence / inputs.market_price) * 100.0

        spread = max(0.0, inputs.ask - inputs.bid)
        half_spread = spread / 2.0
        mc_sigma = self._mc_sigma(inputs.mc_ci)

        model_uncertainty = max(0.10, half_spread, mc_sigma)
        z_score = divergence / model_uncertainty
        p_value_two_sided = float(2.0 * (1.0 - norm.cdf(abs(z_score))))

        expected_edge_per_share = divergence - half_spread
        expected_edge_per_contract = expected_edge_per_share * inputs.contract_multiplier - inputs.commission_per_contract
        notional_per_contract = max(inputs.market_price * inputs.contract_multiplier, 0.01)
        expected_return_pct = (expected_edge_per_contract / notional_per_contract) * 100.0

        confidence_weight = float(np.clip(inputs.confidence, 0.05, 0.95))
        prob_edge_real = float(np.clip((1.0 - p_value_two_sided) * confidence_weight, 0.0, 1.0))
        edge_ratio = expected_edge_per_contract / notional_per_contract

        # Fractional Kelly with conservative clip for options
        raw_kelly = prob_edge_real - (1.0 - prob_edge_real) / max(abs(edge_ratio), 1e-6)
        kelly_fraction = float(np.clip(raw_kelly * 0.25, 0.0, 0.05))

        quality_tier = self._quality_tier(prob_edge_real, expected_return_pct, p_value_two_sided)
        actionable = bool(prob_edge_real >= 0.55 and expected_return_pct > 0.0 and p_value_two_sided < 0.20)

        return {
            "divergence_dollars": divergence,
            "divergence_pct": divergence_pct,
            "spread": spread,
            "uncertainty_sigma": model_uncertainty,
            "z_score": z_score,
            "p_value": p_value_two_sided,
            "expected_edge_per_contract": expected_edge_per_contract,
            "expected_return_pct": expected_return_pct,
            "prob_edge_real": prob_edge_real,
            "kelly_fraction": kelly_fraction,
            "quality_tier": quality_tier,
            "actionable": actionable,
        }

    @staticmethod
    def _mc_sigma(mc_ci: Tuple[float, float]) -> float:
        if not mc_ci or len(mc_ci) != 2:
            return 0.0
        lo, hi = mc_ci
        return max(0.0, (hi - lo) / 4.0)

    @staticmethod
    def _quality_tier(prob_edge_real: float, expected_return_pct: float, p_value: float) -> str:
        if prob_edge_real >= 0.70 and expected_return_pct >= 3.0 and p_value <= 0.10:
            return "A"
        if prob_edge_real >= 0.60 and expected_return_pct >= 1.0 and p_value <= 0.20:
            return "B"
        if expected_return_pct > 0:
            return "C"
        return "D"

    @staticmethod
    def _empty(reason: str) -> Dict:
        return {
            "divergence_dollars": 0.0,
            "divergence_pct": 0.0,
            "spread": 0.0,
            "uncertainty_sigma": 0.0,
            "z_score": 0.0,
            "p_value": 1.0,
            "expected_edge_per_contract": 0.0,
            "expected_return_pct": 0.0,
            "prob_edge_real": 0.0,
            "kelly_fraction": 0.0,
            "quality_tier": "D",
            "actionable": False,
            "reason": reason,
        }
