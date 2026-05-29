"""Differential Privacy Mechanisms Module.
"""

import math
from typing import Any, Dict, Optional
import numpy as np


# ==============================================================================
# COMPOSITION BOUNDS
# ==============================================================================

def get_best_eps_0(
    eps_target: float,
    delta_target: float,
    k: int,
) -> float:
    """Calculates the optimal base privacy cost per step under composition rules.
    """
    eps_basic = eps_target / k
    if not delta_target:
        return eps_basic

    log_term = math.log(1.0 / delta_target)
    term1 = (2.0 * log_term) / k
    eps_adv = math.sqrt(term1 + (2.0 * eps_target / k)) - math.sqrt(term1)

    return max(eps_basic, eps_adv)


# ==============================================================================
#  MECHANISMS
# ==============================================================================

def laplace_mechanism(
    data: Any,
    eps: float,
    sensitivity: float,
    private: bool
) -> Any:
    """Adds Laplace noise calibrated to a given epsilon and global sensitivity. """
    if not private:
        return data
    scale = sensitivity / eps
    return data + np.random.laplace(0.0, scale)


def report_noisy_argmax(
    candidates_with_scores: Dict[Any, float],
    eps: float,
    sensitivity: float,
    private: bool
) -> Any:
    """Identifies the optimal candidate element utilizing standard Noisy Argmax. """
    if not private:
        return max(candidates_with_scores.keys(), key=candidates_with_scores.get)

    scale = (2.0 * sensitivity) / eps

    exp_scores = {
        candidate: score + np.random.exponential(scale)
        for candidate, score in candidates_with_scores.items()
    }

    return max(exp_scores.keys(), key=exp_scores.get)


def generalized_report_noisy_argmax(
    candidates_with_scores: Dict[Any, float],
    candidates_with_sensitivities: Dict[Any, float],
    eps: float,
    beta: float,
    private: bool,
    upperbound_num_cands: Optional[int] = None,
) -> Any:
    """Generalized Noisy Argmax for handling heterogeneous sensitivities. """

    if not private:
        return max(candidates_with_scores.keys(), key=candidates_with_scores.get)

    if len(candidates_with_scores) == 1:
        return next(iter(candidates_with_scores))

    num_cand = upperbound_num_cands or len(candidates_with_sensitivities)
    t = (2.0 * np.log(num_cand / beta)) / eps

    aux_scores = {}

    for u in candidates_with_scores.keys():
        margins = []
        for v in candidates_with_sensitivities.keys():
            if u == v:
                continue

            score_u_adj = candidates_with_scores[u] - t * candidates_with_sensitivities[u]
            score_v_adj = candidates_with_scores[v] - t * candidates_with_sensitivities[v]

            denom = candidates_with_sensitivities[u] + candidates_with_sensitivities[v]
            margin = (score_u_adj - score_v_adj) / denom
            margins.append(margin)

        worst_margin = min(margins)

        # Can be implemented with Laplace/Exponential noise
        aux_scores[u] = worst_margin + np.random.laplace(0.0, 1.0 / eps)
        # aux_scores[u] = worst_margin + np.random.exponential(2 / eps)

    return max(aux_scores.keys(), key=aux_scores.get)