import itertools
import math
import random
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

from classes import GroundSet
from dp_mechanisms import (
    generalized_report_noisy_argmax,
    get_best_eps_0,
    laplace_mechanism,
    report_noisy_argmax,
)
from experiments.uber.objectives import UberMonotoneObjective
from feature_selection import NaiveBayesMutualInfo


# ==============================================================================
# 1. MONOTONE
# ==============================================================================

def dp_density_greedy(
        f: Any,
        gs: GroundSet,
        eps_0: float,
        beta: float,
        private: bool = True,
) -> Set[Any]:

    S = set()
    current_val = 0.0
    C = {v for v in gs.elements if gs.is_feasible({v})}
    i = 0

    _, aux = f.evaluate(S)

    while len(C) > 0:
        # 1. Calculate density (marginal gain / cost)
        candidates_with_scores = {
            e: f.marginal_gain(e, S, auxiliary=aux)[0] / gs.get_cost(e) for e in C
        }

        # 2. Calculate density sensitivity
        candidates_with_sensitivities = {
            e: 2 * f.sensitivity / gs.get_cost(e) for e in C
        }

        # 3. Select the next element privately
        # We divide beta by gs.k for composition over the iterations
        u_next = generalized_report_noisy_argmax(
            candidates_with_scores,
            candidates_with_sensitivities,
            eps_0,
            beta / gs.k,
            private
        )

        # Calculate the specific gain this element provides
        gain, aux = f.marginal_gain(u_next, S, aux, charge=False)
        aux = f.add_one_element(u_next, S, aux)
        S.add(u_next)

        # Update current_val so the NEXT marginal_gain calculation is correct
        current_val += gain

        # 5. Update candidates
        C.remove(u_next)  # Remove the one we just processed

        # Remove any candidates that can no longer fit in the budget
        C = {v for v in C if gs.is_feasible(S | {v})}
        i += 1

    return S


def dp_two_guess_greedy(
        f: Any,
        gs: GroundSet,
        eps: float,
        delta: float,
        beta: float,
        private: bool = True,
        mi_cache: Optional[Dict] = None,
        model: Optional[Any] = None,
        exp: str = 'UBER'
) -> Tuple[Set[Any], int]:
    """DP Two-Guess Greedy framework for processing pairs and greedy extensions."""
    f.num_queries = 0
    p = random.random()
    candidates = {}  # Store as (set_tuple, iteration_id): noisy_value
    delta_prime = np.ceil(3 * beta / (gs.nb_elements ** 2)) * delta

    eps_0 = get_best_eps_0(eps, delta_prime, gs.k)

    # 1. Evaluate Pairs + Greedy Extensions
    for Y_pair in itertools.combinations(sorted(gs.elements), 2):
        seed_set = set(Y_pair)
        if not gs.is_feasible(seed_set):
            continue

        num_guesses = int(np.ceil(3 / beta)) if private else 1
        for j in range(num_guesses):
            if random.random() <= p or not private:
                # Residual problem setup
                remaining_indices = gs.elements - seed_set
                remaining_costs = {e: c for e, c in gs.costs.items() if e not in seed_set}
                indices_to_keep = sorted(list(remaining_indices))
                gs_res = GroundSet(
                    remaining_indices,
                    remaining_costs,
                    capacity=gs.capacity - gs.get_total_cost(seed_set),
                    aux_data=gs.aux_data
                )
                if gs_res.nb_elements == 0:
                    continue

                # Define residual instance.
                if exp == 'MI':
                    f_res = NaiveBayesMutualInfo(
                        f.df, model, residual=seed_set, mi_cache=mi_cache, sensitivity=f.sensitivity
                    )
                elif exp == 'UBER':
                    f_res = UberMonotoneObjective(
                        passenger_coords=f.passengers,
                        grid_coords=gs_res.aux_data,
                        sensitivity=f.sensitivity,
                        residual=seed_set
                    )

                s_greedy = dp_density_greedy(
                    f_res, gs_res, eps_0=eps_0 / 6, beta=beta / 3, private=private
                )
                s_candidate = s_greedy | seed_set
                f.num_queries += f_res.num_queries

                # Use a tuple of (set, iteration_index) to prevent overwriting
                score = laplace_mechanism(
                    f.evaluate(s_candidate)[0], eps_0 / 6, f.sensitivity, private=private
                )
                candidates[(tuple(sorted(s_candidate)), j)] = score

    # 2. Evaluate Singletons
    for e in gs.elements:
        singleton = {e}
        if gs.is_feasible(singleton):
            score = laplace_mechanism(
                f.evaluate(singleton)[0], eps_0 / 6, f.sensitivity, private=private
            )
            # Use a unique identifier for singletons to avoid collisions
            candidates[(tuple(singleton), 'singleton')] = score

    if not candidates:
        return set(), f.num_queries

    # Max returns the (tuple, id) pair; we take the first element (the set)
    best_candidate_key = max(candidates, key=candidates.get)
    return set(best_candidate_key[0]), f.num_queries


def dp_density_greedy_plus(
        f: Any,
        gs: GroundSet,
        eps: float,
        delta: float,
        beta: float,
        private: bool = True,
        mi_cache: Optional[Dict] = None
) -> Tuple[Set[Any], int]:

    f.num_queries = 0
    S = set()
    current_val = 0.0
    C = {v for v in gs.elements if gs.is_feasible({v})}
    i = 0
    observed = dict()
    eps_0 = get_best_eps_0(eps / 2, delta, gs.k)

    _, aux = f.evaluate(S)

    while len(C) > 0:
        # 1. Calculate density (marginal gain / cost)
        candidates_with_densities = {
            e: f.marginal_gain(e, S, aux)[0] / gs.get_cost(e) for e in C
        }

        # 2. Calculate density sensitivity
        candidates_with_sensitivities = {
            e: 2 * f.sensitivity / gs.get_cost(e) for e in C
        }

        # 3. Select the next element privately
        # We divide beta by gs.k for composition over the iterations
        u_next = generalized_report_noisy_argmax(
            candidates_with_densities,
            candidates_with_sensitivities,
            eps_0,
            beta / gs.k,
            private
        )

        # Calculate the specific gain this element provides
        gain, _ = f.marginal_gain(u_next, S, aux, charge=False)
        aux = f.add_one_element(u_next, S, aux)
        S.add(u_next)

        # Update current_val so the NEXT marginal_gain calculation is correct
        current_val += gain
        observed[tuple(S)] = current_val

        # 5. Update candidates
        C.remove(u_next)  # Remove the one we just processed

        # Remove any candidates that can no longer fit in the budget
        C = {v for v in C if gs.is_feasible(S | {v})}
        i += 1

    # In Greedy+: Add all singletons and single element extensions
    final_cands = {tuple(S): observed[tuple(S)]}

    for u in gs.elements:
        # 1. Evaluate the single element if not already seen
        if tuple({u}) not in observed:
            final_cands[tuple({u})] = f.evaluate({u})

        # 2. Capture current keys to avoid "dictionary changed size" error
        for s in observed.keys():
            s_set = set(s)
            if u in s_set:
                continue

            s_u = s_set | {u}
            s_u_tuple = tuple(sorted(s_u))  # Sorting ensures consistent keys

            if s_u_tuple not in observed and gs.is_feasible(s_u):
                final_cands[s_u_tuple] = f.evaluate(s_u)

    # Select best observed
    S_out = report_noisy_argmax(
        observed,
        eps / 2,
        f.sensitivity,
        private
    )

    return set(S_out), f.num_queries



# ==============================================================================
# 3. NON-MONOTONE
# ==============================================================================

def dp_density_sample_greedy(
        f: Any,
        gs: GroundSet,
        eps: float,
        delta: float,
        beta: float,
        private: bool = True,
        mi_cache: Optional[Dict] = None
) -> Tuple[Set[Any], int]:

    f.num_queries = 0
    S = set()
    current_val = 0.0
    C = {v for v in gs.elements if gs.is_feasible({v})}
    i = 0
    observed = dict()

    log_1_delta = np.log(1 / delta)
    log_2_delta = np.log(2 / delta)

    # Calculate composition denominators
    k1 = 2 * log_1_delta + np.sqrt(4 * log_1_delta ** 2 + 8 * gs.k * log_1_delta)
    k2 = 2 * log_2_delta + np.sqrt(4 * log_2_delta ** 2 + 8 * gs.k * log_2_delta)

    # Calculate the two candidate bounds for per-query epsilon
    splt = 0.9
    eps_01 = splt * eps / k1

    term1 = (2 * log_2_delta) / k2
    eps_02 = np.sqrt(term1 + (2 * splt * eps / k2)) - np.sqrt(term1)

    # Select the maximum allowable per-query epsilon
    eps_0 = max(eps_01, eps_02)

    _, aux = f.evaluate(S)

    while len(C) > 0:
        # 1. Calculate density (marginal gain / cost)
        candidates_with_densities = {
            e: f.marginal_gain(e, S, aux)[0] / gs.get_cost(e) for e in C
        }

        # 2. Calculate density sensitivity
        candidates_with_sensitivities = {
            e: 2 * f.sensitivity / gs.get_cost(e) for e in C
        }

        # 3. Select the next element privately
        # We divide beta by gs.k for composition over the iterations
        u_next = generalized_report_noisy_argmax(
            candidates_with_densities,
            candidates_with_sensitivities,
            eps_0,
            beta / gs.k,
            private
        )

        if random.random() < 0.5:
            # Calculate the specific gain this element provides
            gain, _ = f.marginal_gain(u_next, S, aux, charge=False)
            aux = f.add_one_element(u_next, S, aux)
            S.add(u_next)
            # Update current_val so the NEXT marginal_gain calculation is correct
            current_val += gain
            observed[tuple(S)] = current_val

        # 5. Update candidates
        C.remove(u_next)  # Remove the one we just processed

        # Remove any candidates that can no longer fit in the budget
        C = {v for v in C if gs.is_feasible(S | {v})}
        i += 1

    # Single element extensions
    final_cands = {**observed}

    for u in gs.elements:
        # 1. Evaluate the single element if not already seen
        # 2. Capture current keys to avoid "dictionary changed size" error
        for s in observed.keys():
            s_set = set(s)
            if u in s_set:
                continue

            s_u = s_set | {u}
            s_u_tuple = tuple(sorted(s_u))  # Sorting ensures consistent keys

            if s_u_tuple not in observed and gs.is_feasible(s_u):
                final_cands[s_u_tuple] = f.evaluate(s_u)

    # Select best observed
    S_out = report_noisy_argmax(
        observed,
        eps * (1 - splt),
        f.sensitivity,
        private
    )

    return set(S_out), f.num_queries