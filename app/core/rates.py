from typing import Tuple
import numpy as np


def calculate_rates(
    bids: np.ndarray,
    asks: np.ndarray,
    fee_rate: float,
    slippage_bps: float,
    latency_ms: float,
    drift_bps_per_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    spread = np.maximum(asks - bids, 1e-8)
    mid = (bids + asks) / 2.0
    slippage = mid * (slippage_bps / 10000.0)
    drift = mid * (drift_bps_per_sec / 10000.0) * (latency_ms / 1000.0)
    effective_bid = bids - slippage - drift - spread * 0.25
    effective_ask = asks + slippage + drift + spread * 0.25
    ij_rates = effective_bid * (1.0 - fee_rate)
    ji_rates = 1 / effective_ask * (1.0 - fee_rate)
    return ij_rates, ji_rates
