#!/usr/bin/env python3
"""
Hill-Climbing Key Recovery for ML-DSA via Verification Fitness Oracle (v6)

Solves the Leaky-Signature-LWE problem (Damm et al., "One Bit to Rule Them
All") by hill-climbing over a verification fitness oracle.  Phase 1 performs
relation extraction and the j-independence transformation to produce Integer
LWE relations; Phase 2 uses OLS regression for a warm start, then iteratively
minimises a fitness function over the coefficient space.

New in v6 -- Optional MOSEK ILP fallback (--mosek):
  When hill climbing exhausts --max-iter without recovering the key, the best
  intermediate solution x' is handed to MOSEK's mixed-integer optimizer as a
  warm start for an Integer Linear Program.  The ILP encodes *all* verification
  relations as hard constraints (guaranteed to hold in the deterministic leakage
  model) and minimises the L1 distance to x', guiding the solver to fix the
  remaining incorrect positions.  Specifically:

    minimise   sum_j |x_j - x'_j|          (linearised via auxiliary vars)
    subject to lb_i <= <c_i, x> <= ub_i    for every informative relation i
               x_j in {-eta, ..., eta}      for j = 1, ..., n

  Requires the MOSEK Python package (``pip install Mosek``) and a valid license.

Fitness modes (--fitness):

  excess (default):
    S(x) = sum of constraint excesses over all equations (L1).
    Provides gradient information on F-plateaus.

  count:
    F(x) = number of violated verification equations (L0).

  combined:
    M(x) = lambda * F(x) + S(x).
    Fixed penalty lambda per violated equation plus its excess.
    Interpolates between pure excess (lambda=0) and count-dominated
    (lambda >> 1). Default lambda = beta = eta * tau.

Optimization strategies (all independently toggleable):

  Tier 1 -- Score-guided sampling (--score-guided):
    Uses regression residuals to bias position selection toward uncertain
    positions, dramatically increasing the hit rate on wrong positions.

  Tier 2 -- Adaptive block size / VNS (--adaptive-w):
    Variable Neighborhood Search: increases w when stuck, resets on progress.

  Tier 3a -- Accept lateral moves with tabu (--lateral-moves):
    Accepts F' == F to drift along plateaus; lightweight tabu prevents cycling.

  Tier 3b -- Frequency-based diversification (--diversify):
    Biases selection toward under-explored positions; periodic forced sweeps
    guarantee full coverage.

  Tier 4 -- Iterated Local Search / perturbation restart (--perturb-restart):
    When stuck at maximum w (or base w if adaptive-w is off), randomly perturbs
    p positions to break compensating error clusters, then restarts local search.
    Best-ever solution is tracked across all perturbation rounds.

  Tier 5 -- Sequential position selection (--sequential-w):
    Instead of random position sampling, maintains a pool of available
    positions that is used for w_base only to find all easily findable improvements.
    This way low hanging fruit is systematically harvested at low w before moving on to higher w.

    --all-optimizations: Enables all of the above.

Graceful interruption:
  Ctrl+C during execution will finish the current iteration, then print
  summary statistics for all keys processed so far and write partial CSV
  output if --output was specified.  Ctrl+C also interrupts an active MOSEK
  solve gracefully via Model.breakSolver().

Usage:
  python hillclimb_mldsa_v6.py --params 44 --inf-rels 25000 --block-size 5
  python hillclimb_mldsa_v6.py --params 44 --inf-rels 25000 --block-size 5 \\
      --all-optimizations --mosek --mosek-timeout 120
"""

import argparse
import csv
import signal
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import product as cartesian_product
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr


# ===================================================================
# Global state for signal handling
# ===================================================================
_interrupt_event = threading.Event()
_active_mosek_model = None          # Set while MOSEK is solving
_active_mosek_lock = threading.Lock()


def _sigint_handler(signum, frame):
    """Handle Ctrl+C: set interrupt flag and break MOSEK if active."""
    if _interrupt_event.is_set():
        sys.exit(1)  # Second Ctrl+C: hard exit

    _interrupt_event.set()

    # If MOSEK is currently solving, break it immediately
    with _active_mosek_lock:
        if _active_mosek_model is not None:
            print("\n  [INTERRUPT] Ctrl+C -- breaking MOSEK solver...",
                  flush=True)
            _active_mosek_model.breakSolver()
            return

    print("\n  [INTERRUPT] Ctrl+C -- finishing current iteration, "
          "then printing summary...", flush=True)


# ===================================================================
# ML-DSA parameter sets (FIPS 204)
# ===================================================================
MLDSA_PARAMS = {
    44: dict(n=256, eta=2, gamma=2**17, tau=39, name="ML-DSA-44"),
    65: dict(n=256, eta=4, gamma=2**19, tau=49, name="ML-DSA-65"),
    87: dict(n=256, eta=2, gamma=2**19, tau=60, name="ML-DSA-87"),
}


def compute_beta_eff(params, leakage_index):
    """Effective error bound: min(beta, 2^{ell-1}).

    When 2^{ell-1} < beta the j-independence transformation was not applied,
    so the actual LWE error lives in [-2^{ell-1}, 2^{ell-1}] rather than
    [-beta, beta].
    """
    beta = params["eta"] * params["tau"]
    return min(beta, 2 ** (leakage_index - 1))


# ===================================================================
# Modular arithmetic and relation extraction
# ===================================================================
def get_bit(value, position):
    """Return the bit at the given position in the binary representation."""
    return (value >> position) & 1


def mod_centered(value, modulus):
    """Compute the centered reduction of value modulo modulus."""
    out = value % modulus
    if out > modulus / 2:
        out -= modulus
    return out


def extract_lwe_relation(z, j, y_j):
    """
    Relation extraction for the Leaky-Signature-LWE problem.

    Given a signature coefficient z = <c, x> + y (after rejection sampling)
    and the leaked bit y_j, compute z_bar via:

      1. Normal-form computation  (Damm et al., Eq. 6 & 7)
      2. Simplified LZS+ extraction  (Damm et al., Eq. 10)

    Parameters:
        z:    signature coefficient (integer)
        j:    leakage bit index
        y_j:  the leaked bit of the randomness y at position j

    Returns:
        z_bar: extracted LWE relation value (integer)
    """
    z_mod = mod_centered(z, 2 ** (j + 1))

    if get_bit(z_mod, j) == 1:
        z_up = z + 2 ** (j - 1)         # Eq. 6: rotate right
        b_j = y_j                        # Eq. 7: bit unchanged
    else:
        z_up = z - 2 ** (j - 1)         # Eq. 6: rotate left
        b_j = y_j ^ 1                   # Eq. 7: bit flipped

    if b_j == 1:
        z_bar = mod_centered(z_up, 2 ** (j + 1))
    elif get_bit(z_up, j) == 1:
        z_bar = mod_centered(z_up, 2 ** (j + 1)) + 2**j
    else:
        z_bar = mod_centered(z_up, 2 ** (j + 1)) - 2**j

    return z_bar


def format_key(x):
    """Format a key vector as a compact string for display."""
    return "[" + " ".join(f"{v:+d}" for v in x) + "]"


# ===================================================================
# Phase 1: Data generation
# ===================================================================
def generate_informative_relations(rng, x_true, r_target, params,
                                   leakage_index):
    """
    Simulate signature generation with bit leakage and extract Integer LWE
    relations via the j-independence transformation (Damm et al., Eq. 3).

    The leaked bit y_j is *observed* (not forced to a fixed value), modelling
    a realistic side channel where the attacker reads but does not control
    the randomness.

    Parameters:
        rng:            numpy random generator
        x_true:         (n,) true partial key (int8)
        r_target:       number of informative relations to collect
        params:         ML-DSA parameter dict
        leakage_index:  bit position j of the leaked randomness bit

    Returns:
        z_tilde: (r_target,) transformed relation values (float64)
        C:       (r_target, n) dense challenge matrix (int8)
        total_signatures: number of signatures processed
    """
    n = params["n"]
    eta = params["eta"]
    gamma = params["gamma"]
    tau = params["tau"]
    beta = eta * tau
    ell = leakage_index

    z_values = []
    challenge_rows = []
    total_signatures = 0
    batch_size = max(r_target, 1000)

    while len(z_values) < r_target:
        remaining = r_target - len(z_values)
        n_batch = max(remaining * 5, batch_size)
        y_batch = rng.integers(-(gamma - 1), gamma, size=n_batch,
                               dtype=np.int64)

        for y_raw in y_batch:
            if len(z_values) >= r_target:
                break

            y = int(y_raw)
            y_j = get_bit(y, ell)

            c_idx = rng.choice(n, size=tau, replace=False)
            c_signs = rng.choice([-1, 1], size=tau).astype(np.int8)

            cx = int(np.dot(x_true[c_idx], c_signs))
            z = y + cx

            # Mimic rejection sampling
            if not (-(gamma - beta) < z < (gamma - beta)):
                continue

            total_signatures += 1

            z_bar = extract_lwe_relation(z, ell, y_j)

            # Informativity filter
            if abs(z_bar) <= 2 ** (ell - 1) - beta:
                continue

            # j-independence transformation (Eq. 3)
            if 2 ** (ell - 1) > beta:
                if z_bar > 2 ** (ell - 1) - beta:
                    z_bar -= 2 ** (ell - 1) - beta
                else:
                    z_bar += 2 ** (ell - 1) - beta

            c_dense = np.zeros(n, dtype=np.int8)
            c_dense[c_idx] = c_signs
            challenge_rows.append(c_dense)
            z_values.append(z_bar)

    z_tilde = np.array(z_values[:r_target], dtype=np.float64)
    C = np.array(challenge_rows[:r_target], dtype=np.int8)
    return z_tilde, C, total_signatures


# ===================================================================
# Phase 1b: Regression warm start
# ===================================================================
def regression_warm_start(C, z_tilde, n, eta):
    """
    Run LSQR regression on the Integer LWE system C @ x ~ z_tilde.

    Returns:
        x_hat_float:   (n,) raw OLS estimate (float64)
        x_hat_rounded: (n,) rounded and clipped to [-eta, eta] (int8)
    """
    C_csr = csr_matrix(C)
    x_hat_float = lsqr(C_csr, z_tilde)[0]
    x_hat_rounded = np.round(x_hat_float).astype(np.int8)
    x_hat_rounded = np.clip(x_hat_rounded, -eta, eta)
    return x_hat_float, x_hat_rounded


# ===================================================================
# Sampling weight helpers
# ===================================================================
def compute_score_weights(x_hat_float, eta, temperature=2.0):
    """
    Compute per-position sampling weights from regression residuals.

    Positions where the regression estimate is far from any integer in
    [-eta, eta] (high residual = low confidence) get higher weight.

    Parameters:
        x_hat_float: (n,) raw regression estimates (float)
        eta:         coefficient bound
        temperature: controls sharpness; higher = more concentrated on
                     uncertain positions, lower = more uniform

    Returns:
        weights: (n,) non-negative sampling weights (sum to 1)
    """
    valid_ints = np.arange(-eta, eta + 1)
    # Vectorised distance: min |x_hat_j - k| over k in {-eta, ..., eta}
    distances = np.min(
        np.abs(x_hat_float[:, np.newaxis] - valid_ints), axis=1
    )

    logits = temperature * distances
    logits -= np.max(logits)   # numerically stable softmax
    weights = np.exp(logits)
    weights /= weights.sum()
    return weights


def compute_diversified_weights(base_weights, freq_counts,
                                diversify_strength=1.0):
    """
    Combine base weights (uniform or score-guided) with frequency-based
    anti-repetition bias.
    """
    penalty = 1.0 + diversify_strength * freq_counts
    weights = base_weights / penalty
    total = weights.sum()
    if total > 0:
        weights /= total
    else:
        weights = np.ones_like(weights) / len(weights)
    return weights


# ===================================================================
# Fitness computation
# ===================================================================
def _compute_fitness_scalar(ip, lb, ub, fitness_mode, fitness_lambda):
    """
    Compute scalar fitness value and violation count for a single candidate.

    Parameters:
        ip: (R,) inner product vector C @ x
        lb, ub: (R,) constraint bounds

    Returns:
        (fitness_value, F_count)
    """
    violated = (ip < lb) | (ip > ub)
    F = int(np.count_nonzero(violated))
    if fitness_mode == "count":
        return float(F), F
    excess_total = float(np.sum(
        np.maximum(lb - ip, 0.0) + np.maximum(ip - ub, 0.0)
    ))
    if fitness_mode == "excess":
        return excess_total, F
    # combined
    return fitness_lambda * F + excess_total, F


def _compute_fitness_batch(ip_batch, lb, ub, fitness_mode, fitness_lambda):
    """
    Compute fitness values and violation counts for a batch of candidates.

    Parameters:
        ip_batch: (R, num_candidates) inner products
        lb, ub:   (R,) bounds

    Returns:
        fitness:  (num_candidates,) float64
        F_counts: (num_candidates,) int
    """
    violated = (ip_batch < lb[:, np.newaxis]) | (ip_batch > ub[:, np.newaxis])
    F_counts = np.count_nonzero(violated, axis=0)

    if fitness_mode == "count":
        return F_counts.astype(np.float64), F_counts

    excess = (np.maximum(lb[:, np.newaxis] - ip_batch, 0.0)
              + np.maximum(ip_batch - ub[:, np.newaxis], 0.0))
    S_vals = np.sum(excess, axis=0)

    if fitness_mode == "excess":
        return S_vals, F_counts
    # combined
    return fitness_lambda * F_counts + S_vals, F_counts


# ===================================================================
# Phase 2: Hill-climbing with optional optimizations
# ===================================================================
def _evaluate_candidate_chunk(C_block, ip_base, lb, ub, candidate_chunk,
                              chunk_offset, fitness_mode, fitness_lambda):
    """Evaluate fitness for a chunk of candidate assignments (thread worker)."""
    new_contrib = C_block @ candidate_chunk.astype(np.int32).T
    ip_new = ip_base[:, np.newaxis] + new_contrib
    fitness, F_counts = _compute_fitness_batch(
        ip_new, lb, ub, fitness_mode, fitness_lambda
    )
    best_local = int(np.argmin(fitness))
    return (chunk_offset + best_local,
            float(fitness[best_local]),
            int(F_counts[best_local]))


def _precompute_candidates(values, w):
    """Precompute all (2*eta+1)^w candidate tuples for a given block size."""
    return np.array(list(cartesian_product(values, repeat=w)), dtype=np.int8)


def hillclimb(C, z_tilde, x_init, params, rng, w, T,
              leakage_index,
              true_key=None, verbose=True, print_keys=False, num_workers=1,
              # Fitness mode
              fitness_mode="excess", fitness_lambda=None,
              # Tier 1: Score-guided sampling
              score_weights=None,
              # Tier 2: Adaptive block size / VNS
              use_adaptive_w=False, adaptive_w_max=6, adaptive_w_patience=50,
              # Tier 3a: Lateral moves
              use_lateral_moves=False, lateral_tabu_size=20,
              # Tier 3b: Frequency diversification
              use_diversify=False, diversify_strength=1.0, sweep_interval=0,
              # Tier 4: Iterated Local Search
              use_perturb_restart=False, perturb_strength=30,
              perturb_patience=50, perturb_max=50,
              perturb_score_guided=False,
              # Sequential position selection
              use_sequential_w=False):
    """
    Hill-climbing key recovery using configurable fitness function,
    with optional optimization strategies including ILS.

    Sequential position selection mode (--sequential-w):
      When enabled, the algorithm maintains a pool of available positions for
      each w size. Each iteration, w random positions are selected from the
      available pool and removed. For example, iteration 1 could pick [5, 2],
      iteration 2 could pick [104, 12], etc. When all n positions are exhausted,
      w is incremented and the pool is reset to all positions.

    Returns:
        x_final:           (n,) recovered key estimate
        F_final:           final violation count (L0)
        iterations:        number of iterations used
        history:           list of (iteration, F, D_from_true) tuples
        num_perturbations: number of ILS perturbations applied
    """
    n = params["n"]
    eta = params["eta"]
    beta_eff = compute_beta_eff(params, leakage_index)

    if fitness_lambda is None:
        fitness_lambda = float(beta_eff)

    values = np.arange(-eta, eta + 1, dtype=np.int8)
    w_base = w
    w_curr = w

    # Candidate tuple cache (keyed by block size)
    candidate_cache = {w_curr: _precompute_candidates(values, w_curr)}

    def get_candidates(block_size):
        if block_size not in candidate_cache:
            candidate_cache[block_size] = _precompute_candidates(
                values, block_size)
        return candidate_cache[block_size]

    # ---------------------------------------------------------------
    # Sampling weights
    # ---------------------------------------------------------------
    base_weights = (score_weights.copy() if score_weights is not None
                    else np.ones(n, dtype=np.float64) / n)

    perturb_weights = (score_weights.copy()
                       if (perturb_score_guided and score_weights is not None)
                       else None)

    # ---------------------------------------------------------------
    # Mutable exploration state (reset on perturbation)
    # ---------------------------------------------------------------
    freq_counts = np.zeros(n, dtype=np.int64)
    tabu_set = set()
    tabu_queue = deque()
    sweep_permutation = None
    sweep_offset = 0
    seq_w_available = {}  # For sequential-w: remaining positions per w size
    seq_w_tried = {}     # Track positions that have been tried for each w
    iters_since_improvement = 0

    def _reset_soft_state():
        """Reset exploration state after a perturbation."""
        nonlocal freq_counts, tabu_set, tabu_queue
        nonlocal sweep_permutation, sweep_offset, w_curr
        nonlocal seq_w_available, seq_w_tried
        nonlocal iters_since_improvement
        freq_counts[:] = 0
        tabu_set.clear()
        tabu_queue.clear()
        sweep_permutation = None
        sweep_offset = 0
        seq_w_available.clear()
        seq_w_tried.clear()
        w_curr = w_base
        iters_since_improvement = 0
        # Reinitialize sequential-w pool for base w
        if use_sequential_w:
            seq_w_available[w_base] = set(range(n))
            seq_w_tried[w_base] = 0

    # ---------------------------------------------------------------
    # Precompute constraint bounds (constant throughout)
    # ---------------------------------------------------------------
    C_i32 = C.astype(np.int32)
    lb = z_tilde - beta_eff
    ub = z_tilde + beta_eff

    # Current solution state
    x_curr = x_init.copy()
    ip = C_i32 @ x_curr.astype(np.int32)
    fitness_curr, F_curr = _compute_fitness_scalar(
        ip, lb, ub, fitness_mode, fitness_lambda)

    # History and logging
    history = []
    D_init = int(np.sum(x_curr != true_key)) if true_key is not None else -1
    history.append((0, F_curr, D_init))

    if verbose:
        extra = f", fitness={fitness_curr:.1f}" if fitness_mode != "count" else ""
        print(f"  Iter 0: F={F_curr}, D={D_init}{extra}")
        if print_keys:
            print(f"    x* = {format_key(x_curr)}")

    # Best-ever tracking (survives perturbations)
    fitness_best_ever = fitness_curr
    F_best_ever = F_curr
    x_best_ever = x_curr.copy()
    ip_best_ever = ip.copy()

    num_perturbations = 0

    # Thread pool for parallel candidate evaluation
    use_parallel = num_workers > 1
    executor = (ThreadPoolExecutor(max_workers=num_workers)
                if use_parallel else None)

    iters_used = 0
    
    # Initialize sequential-w pool for base w
    if use_sequential_w:
        seq_w_available[w_base] = set(range(n))
        seq_w_tried[w_base] = 0
    
    try:
        for t in range(1, T + 1):
            if F_curr == 0 or _interrupt_event.is_set():
                break
            iters_used = t

            # ===== Tier 4: ILS perturbation check =====
            if (use_perturb_restart
                    and iters_since_improvement >= perturb_patience):
                adaptive_exhausted = ((not use_adaptive_w)
                                      or (w_curr >= adaptive_w_max))
                if adaptive_exhausted and num_perturbations < perturb_max:
                    num_perturbations += 1
                    p = min(perturb_strength, n)

                    if perturb_weights is not None:
                        perturb_pos = rng.choice(
                            n, size=p, replace=False, p=perturb_weights)
                    else:
                        perturb_pos = rng.choice(n, size=p, replace=False)

                    # Save pre-perturbation best
                    if fitness_curr < fitness_best_ever:
                        fitness_best_ever = fitness_curr
                        F_best_ever = F_curr
                        x_best_ever = x_curr.copy()
                        ip_best_ever = ip.copy()
                    else:
                        # Restore best-ever before perturbing, to avoid drifting too far
                        x_curr = x_best_ever.copy()
                        ip = ip_best_ever.copy()
                        fitness_curr = fitness_best_ever
                        F_curr = F_best_ever

                    # Apply perturbation
                    x_curr[perturb_pos] = rng.integers(
                        -eta, eta + 1, size=p, dtype=np.int8)
                    ip = C_i32 @ x_curr.astype(np.int32)
                    fitness_curr, F_curr = _compute_fitness_scalar(
                        ip, lb, ub, fitness_mode, fitness_lambda)
                    _reset_soft_state()

                    D_now = (int(np.sum(x_curr != true_key))
                             if true_key is not None else -1)
                    if verbose:
                        extra = (f", fitness={fitness_curr:.1f}"
                                 if fitness_mode != "count" else "")
                        print(f"    [ILS] Perturbation #{num_perturbations}: "
                              f"perturbed {p} positions -> F={F_curr}, "
                              f"D={D_now}{extra}  (best-ever F={F_best_ever})")
                        if print_keys:
                            print(f"    x* = {format_key(x_curr)}")

                    if F_curr == 0:
                        break

            # ===== Position selection =====
            in_sweep = False

            if use_sequential_w and (w_curr == w_base or not use_adaptive_w):
               
                # Check if all positions have been tried for current w (and w wasn't just adapted)
                if (not seq_w_available[w_curr]):
                    
                    # Try to expand w to the next level
                    new_w = min(w_curr + 1, adaptive_w_max, n)
                    
                    if new_w != w_curr:
                        # Successfully expanded
                        w_curr = new_w
                        seq_w_available[w_curr] = set(range(n))
                        seq_w_tried[w_curr] = 0
                        iters_since_improvement = 0
                        if verbose:
                            print(f"    [sequential-w] w expanded to {w_curr}  "
                                  f"({(2*eta+1)**w_curr} candidates/step)")
                        # Reinitialize for new w (will be done at start of next iteration)
                        continue
                    else:
                        # w is already at max (can't expand), trigger perturbation if enabled
                        if use_perturb_restart and num_perturbations < perturb_max:
                            # Force ILS perturbation on next iteration
                            iters_since_improvement = perturb_patience
                        else:
                            # Otherwise reset and continue exploring at current w
                            seq_w_available[w_curr] = set(range(n))
                            seq_w_tried[w_curr] = 0
                
                # Pick w random positions from available (not contiguous blocks)
                available_list = list(seq_w_available[w_curr])
                
                num_to_pick = min(w_curr, len(available_list))
                selected_indices = rng.choice(len(available_list), size=num_to_pick, 
                                             replace=False)
                positions = np.array([available_list[i] for i in selected_indices], 
                                    dtype=int)
                
                # Remove selected positions from available and track tried count
                for pos in positions:
                    seq_w_available[w_curr].remove(pos)
                seq_w_tried[w_curr] += len(positions)
            elif sweep_interval > 0 and (t % sweep_interval) == 0:
                # Tier 3b: forced deterministic sweep round
                if sweep_permutation is None or sweep_offset >= n:
                    sweep_permutation = rng.permutation(n)
                    sweep_offset = 0
                end = min(sweep_offset + w_curr, n)
                positions = sweep_permutation[sweep_offset:end]
                sweep_offset = end
                in_sweep = True
            else:
                # Weighted sampling (with optional diversification)
                if use_diversify:
                    weights = compute_diversified_weights(
                        base_weights, freq_counts, diversify_strength)
                else:
                    weights = base_weights
                positions = rng.choice(n, size=w_curr, replace=False,
                                       p=weights)

            freq_counts[positions] += 1

            # ===== Candidate evaluation (core inner loop) =====
            actual_w = len(positions)
            candidate_tuples = get_candidates(actual_w)
            num_candidates = candidate_tuples.shape[0]

            C_block = C[:, positions].astype(np.int32)
            x_block_curr = x_curr[positions].astype(np.int32)
            ip_base = ip - C_block @ x_block_curr

            if use_parallel and num_candidates > num_workers:
                chunks = np.array_split(candidate_tuples, num_workers)
                chunk_offsets = [0]
                for ch in chunks[:-1]:
                    chunk_offsets.append(chunk_offsets[-1] + len(ch))
                futures = [
                    executor.submit(
                        _evaluate_candidate_chunk,
                        C_block, ip_base, lb, ub, chunk, offset,
                        fitness_mode, fitness_lambda)
                    for chunk, offset in zip(chunks, chunk_offsets)
                ]
                best_idx, fitness_best, F_best = min(
                    (f.result() for f in futures), key=lambda r: r[1])
            else:
                ip_new = (ip_base[:, np.newaxis]
                          + C_block @ candidate_tuples.astype(np.int32).T)
                fitness_vals, F_counts = _compute_fitness_batch(
                    ip_new, lb, ub, fitness_mode, fitness_lambda)
                best_idx = int(np.argmin(fitness_vals))
                fitness_best = float(fitness_vals[best_idx])
                F_best = int(F_counts[best_idx])

            # ===== Acceptance logic =====
            strict_improvement = fitness_best < fitness_curr
            lateral_move = (not strict_improvement
                            and use_lateral_moves
                            and fitness_best == fitness_curr)

            if lateral_move:
                pos_key = frozenset(positions.tolist())
                if pos_key in tabu_set:
                    lateral_move = False

            accepted = strict_improvement or lateral_move

            if accepted:
                best_values = candidate_tuples[best_idx]
                if not np.array_equal(best_values, x_curr[positions]):
                    x_curr[positions] = best_values
                    ip = ip_base + C_block @ best_values.astype(np.int32)
                    fitness_curr = fitness_best
                    F_curr = F_best
                else:
                    accepted = False

            # Tier 3a: update tabu list
            if use_lateral_moves:
                pos_key = frozenset(positions.tolist())
                if pos_key not in tabu_set:
                    tabu_set.add(pos_key)
                    tabu_queue.append(pos_key)
                    if len(tabu_queue) > lateral_tabu_size:
                        tabu_set.discard(tabu_queue.popleft())

            # ===== Adaptive block size + stagnation tracking =====
            if strict_improvement:
                iters_since_improvement = 0
                if fitness_curr < fitness_best_ever:
                    fitness_best_ever = fitness_curr
                    F_best_ever = F_curr
                    x_best_ever = x_curr.copy()
                    ip_best_ever = ip.copy()
                if use_adaptive_w and w_curr != w_base:
                    w_curr = w_base
                    if verbose:
                        print(f"    [VNS] w reset to {w_curr}")
                # Reset sequential-w available positions when w resets
                if use_sequential_w:
                    w_curr = w_base
                    seq_w_available[w_base] = set(range(n))
                    seq_w_tried[w_base] = 0
                    if verbose and use_adaptive_w:  # Only print if also using adaptive-w, to avoid duplicate messages
                        print(f"    [sequential-w] w reset to {w_curr}")
            else:
                iters_since_improvement += 1

            if (use_adaptive_w
                    and iters_since_improvement >= adaptive_w_patience
                    and (w_curr > w_base or not use_sequential_w)):
                new_w = min(w_curr + 1, adaptive_w_max, n)
                if new_w != w_curr:
                    w_curr = new_w
                    iters_since_improvement = 0
                    # Initialize sequential-w pool for new w if using sequential-w
                    if use_sequential_w:
                        seq_w_available[w_curr] = set(range(n))
                        seq_w_tried[w_curr] = 0
                    if verbose:
                        print(f"    [VNS] w expanded to {w_curr}  "
                              f"({(2*eta+1)**w_curr} candidates/step)")
                    # Skip position selection for this iteration; start fresh with new w next iteration
                    continue

            # ===== Logging =====
            D_now = (int(np.sum(x_curr != true_key))
                     if true_key is not None else -1)
            history.append((t, F_curr, D_now))

            if verbose:
                tag = " *" if strict_improvement else (
                    " ~" if (lateral_move and accepted) else "")
                extra_parts = []
                if fitness_mode != "count":
                    extra_parts.append(f"fitness={fitness_curr:.1f}")
                if in_sweep:
                    extra_parts.append("[sweep]")
                if use_adaptive_w and w_curr != w_base:
                    extra_parts.append(f"[w={w_curr}]")
                extra = ("  " + ", ".join(extra_parts)) if extra_parts else ""
                print(f"  Iter {t}: F={F_curr}, D={D_now}{tag}"
                      f"  pos={sorted(positions.tolist())}{extra}")
                if accepted and print_keys:
                    print(f"    x* = {format_key(x_curr)}")

    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    # Revert to best-ever if current is worse
    if fitness_curr > fitness_best_ever:
        x_curr = x_best_ever
        ip = ip_best_ever
        F_curr = F_best_ever
        fitness_curr = fitness_best_ever

    D_final = int(np.sum(x_curr != true_key)) if true_key is not None else -1
    if F_curr == 0:
        iters_used = t if t <= T else T

    if verbose:
        extra = f", fitness={fitness_curr:.1f}" if fitness_mode != "count" else ""
        perturb_info = (f", perturbations={num_perturbations}"
                        if num_perturbations > 0 else "")
        print(f"  --- Final: F={F_curr}, D={D_final}{extra}, "
              f"iters={iters_used}{perturb_info} ---")
        if print_keys:
            print(f"    x* = {format_key(x_curr)}")
            if true_key is not None:
                print(f"    x  = {format_key(true_key)}")

    return x_curr, F_curr, iters_used, history, num_perturbations


# ===================================================================
# MOSEK ILP fallback: exact key recovery via integer feasibility
# ===================================================================
def mosek_ilp_recovery(C, z_tilde, x_warm, params, leakage_index,
                       mosek_timeout=300.0, verbose=True):
    """
    Attempt exact key recovery by solving an Integer Linear Program (ILP)
    encoding the verification relations as hard constraints.

    The formulation exploits the deterministic side-channel model where every
    informative relation induces a *guaranteed* pair of bounds on <c_i, x*>.
    Hill climbing may fail to find x* due to compensating errors, but the
    constraints themselves are always satisfied by the true key.

    ILP formulation:
        minimise   sum_j d_j                         (L1 distance to warm start)
        subject to lb_i <= sum_j C[i,j] * x_j <= ub_i   for all relations i
                   d_j >= x_j - x'_j                    for all j
                   d_j >= -(x_j - x'_j)                 for all j
                   d_j >= 0                              for all j
                   x_j in {-eta, ..., eta}               for all j

    The auxiliary variables d_j linearise |x_j - x'_j|.  The L1 objective
    guides MOSEK's branch-and-bound toward solutions close to the warm start,
    enabling effective pruning and fast convergence.

    Parameters:
        C:              (R, n) int8 challenge matrix
        z_tilde:        (R,) transformed relation values
        x_warm:         (n,) best key estimate from hill climbing (int8)
        params:         ML-DSA parameter dict
        leakage_index:  bit position j of the leaked bit
        mosek_timeout:  solver time limit in seconds (default: 300)
        verbose:        print solver progress

    Returns:
        x_sol:   (n,) int8 recovered key, or None if solver failed
        t_mosek: float, wall-clock time spent in MOSEK
    """
    global _active_mosek_model

    try:
        from mosek.fusion import (Model, Domain, Expr, Matrix,
                                  ObjectiveSense, AccSolutionStatus)
    except ImportError:
        print("  [MOSEK] ERROR: mosek.fusion not available. "
              "Install with: pip install Mosek")
        return None, 0.0

    n = params["n"]
    eta = params["eta"]
    beta_eff = compute_beta_eff(params, leakage_index)
    R = C.shape[0]

    lb = z_tilde - beta_eff
    ub = z_tilde + beta_eff
    x_warm_f = x_warm.astype(np.float64)

    if verbose:
        print(f"  [MOSEK] Building ILP: {n} integer vars, {R} relations, "
              f"eta={eta}, beta_eff={beta_eff}")

    t0 = perf_counter()

    with Model("mldsa_key_recovery") as M:
        # Register model for signal-handler access
        with _active_mosek_lock:
            _active_mosek_model = M

        try:
            if verbose:
                M.setLogHandler(sys.stdout)

            # Decision variables: x_j in {-eta, ..., eta}
            x = M.variable("x", n,
                            Domain.integral(Domain.inRange(float(-eta),
                                                           float(eta))))

            # Auxiliary variables for L1 objective: d_j >= |x_j - x'_j|
            d = M.variable("d", n, Domain.greaterThan(0.0))
            M.constraint("abs_pos", Expr.sub(d, x),
                          Domain.greaterThan((-x_warm_f).tolist()))
            M.constraint("abs_neg", Expr.add(d, x),
                          Domain.greaterThan(x_warm_f.tolist()))

            # Verification relation constraints: lb <= C @ x <= ub
            C_mosek = Matrix.dense(C.astype(np.float64))
            M.constraint("verify", Expr.mul(C_mosek, x),
                          Domain.inRange(lb.tolist(), ub.tolist()))

            # Objective: minimise L1 distance to warm start
            M.objective("l1_dist", ObjectiveSense.Minimize, Expr.sum(d))

            # Warm start from hill climbing
            x.setLevel(x_warm_f.tolist())
            M.setSolverParam("mioConstructSol", "on")

            # Solver time limit
            M.setSolverParam("mioMaxTime", mosek_timeout)

            if verbose:
                print(f"  [MOSEK] Solving ILP (timeout={mosek_timeout}s)...")

            M.solve()

        finally:
            # Unregister model so signal handler won't touch a stale object
            with _active_mosek_lock:
                _active_mosek_model = None

        t_mosek = perf_counter() - t0

        # Report interrupt
        if _interrupt_event.is_set():
            print(f"  [MOSEK] Interrupted ({t_mosek:.2f}s)")

        # Extract solution
        try:
            M.acceptedSolutionStatus(AccSolutionStatus.Feasible)
            x_level = x.level()
            x_sol = np.round(x_level).astype(np.int8)
            x_sol = np.clip(x_sol, -eta, eta)

            # Post-solve verification
            ip_check = C.astype(np.int32) @ x_sol.astype(np.int32)
            violated = int(np.sum((ip_check < lb) | (ip_check > ub)))

            if verbose:
                obj_val = int(np.sum(np.abs(
                    x_sol.astype(np.int32) - x_warm.astype(np.int32))))
                print(f"  [MOSEK] Solution found: L1 dist to warm start = "
                      f"{obj_val}, constraint violations = {violated} "
                      f"({t_mosek:.2f}s)")

            if violated > 0:
                print(f"  [MOSEK] WARNING: solution violates {violated} "
                      "constraints (numerical issue?)")

            return x_sol, t_mosek

        except Exception:
            if verbose:
                print(f"  [MOSEK] No feasible solution found ({t_mosek:.2f}s)")
            return None, t_mosek


# ===================================================================
# Output helpers
# ===================================================================
def _print_summary(results, total_keys, seed=None, interrupted=False):
    """Print experiment summary from collected results."""
    n_done = len(results)
    n_success = sum(1 for r in results if r["success"])
    n_mosek_attempted = sum(1 for r in results
                           if r.get("mosek_attempted", False))
    n_mosek_success = sum(1 for r in results
                         if r.get("mosek_success", False))
    status = " (INTERRUPTED)" if interrupted else ""

    print(f"\n=== Summary: {n_success}/{n_done} keys recovered "
          f"(of {total_keys} planned){status} ===")
    if seed is not None:
        print(f"  Seed: {seed}")

    if n_mosek_attempted > 0:
        n_climb_only = n_success - n_mosek_success
        print(f"  Hill-climbing alone: {n_climb_only}, "
              f"MOSEK ILP fallback: {n_mosek_success}/{n_mosek_attempted}")

    if n_success > 0:
        succ = [r for r in results if r["success"]]
        print(f"  Avg initial accuracy: "
              f"{np.mean([r['init_accuracy'] for r in results]):.1%}")
        print(f"  Avg iterations to success: "
              f"{np.mean([r['iterations'] for r in succ]):.0f}")
        print(f"  Avg perturbations to success: "
              f"{np.mean([r['num_perturbations'] for r in succ]):.1f}")
        print(f"  Avg hill-climb time: "
              f"{np.mean([r['t_hillclimb'] for r in succ]):.2f}s")
        if n_mosek_success > 0:
            mosek_succ = [r for r in results if r.get("mosek_success", False)]
            print(f"  Avg MOSEK solve time (successful): "
                  f"{np.mean([r['t_mosek'] for r in mosek_succ]):.2f}s")

    fail_results = [r for r in results if not r["success"]]
    if fail_results:
        print(f"  Failed keys: "
              f"avg D_final={np.mean([r['D_final'] for r in fail_results]):.1f}, "
              f"avg F_final={np.mean([r['F_final'] for r in fail_results]):.1f}, "
              f"avg perturbations="
              f"{np.mean([r['num_perturbations'] for r in fail_results]):.1f}")


def _write_csv(results, csv_path):
    """Write results to CSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to {csv_path}")


def _print_experiment_banner(args, params, beta_eff, fitness_mode,
                             fitness_lambda, opt_flags, verbose):
    """Print the experiment configuration banner."""
    beta = params["eta"] * params["tau"]

    print(f"=== Hill-Climbing Experiment v6: {params['name']} ===")
    print(f"  Leakage index: {args.leakage}")
    if beta_eff < beta:
        print(f"  Effective error bound: beta_eff={beta_eff} "
              f"(< beta={beta}, j-independence transform skipped)")
    print(f"  Informative relations: {args.inf_rels}")
    print(f"  Block size w: {args.block_size}")
    print(f"  Max iterations T: {args.max_iter}")
    print(f"  Number of keys: {args.num_keys}")
    print(f"  Seed: {args.seed}")
    print(f"  Verbose: {verbose}")

    if args.mosek:
        print(f"  MOSEK ILP fallback: enabled (timeout={args.mosek_timeout}s)")

    mode_desc = {"count": "count (violation count, L0)",
                 "excess": "excess (sum of constraint excesses, L1)",
                 "combined": (f"combined (lambda={fitness_lambda:.1f}, "
                              f"beta_eff={beta_eff})")}
    print(f"  Fitness: {mode_desc[fitness_mode]}")

    if args.workers > 1:
        print(f"  Workers: {args.workers}")

    # Active optimizations
    active_opts = []
    if opt_flags["score_guided"]:
        active_opts.append(
            f"score-guided (temp={args.score_temperature})")
    if opt_flags["adaptive_w"]:
        active_opts.append(
            f"adaptive-w (max={args.adaptive_w_max}, "
            f"patience={args.adaptive_w_patience})")
    if opt_flags["lateral"]:
        active_opts.append(
            f"lateral-moves (tabu={args.lateral_tabu_size})")
    if opt_flags["diversify"]:
        active_opts.append(
            f"diversify (strength={args.diversify_strength}, "
            f"sweep={args.sweep_interval})")
    if opt_flags["perturb"]:
        target = "score-guided" if args.perturb_score_guided else "uniform"
        active_opts.append(
            f"perturb-restart (p={args.perturb_strength}, "
            f"patience={args.perturb_patience}, "
            f"max={args.perturb_max}, target={target})")
    print(f"  Optimizations: {', '.join(active_opts) or 'none (baseline)'}")
    print()


# ===================================================================
# Main experiment loop
# ===================================================================
def run_experiment(args):
    """Run the full key-recovery experiment."""
    # Install signal handler
    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint_handler)
    _interrupt_event.clear()

    # If no seed provided, generate one from system entropy so experiment is reproducible
    if args.seed is None:
        args.seed = np.random.SeedSequence().entropy

    rng = np.random.default_rng(args.seed)
    params = MLDSA_PARAMS[args.params]
    n = params["n"]
    eta = params["eta"]
    verbose = not args.non_verbose
    print_keys = args.print_keys

    # Resolve fitness mode
    fitness_mode = args.fitness
    fitness_lambda = args.fitness_lambda if args.fitness_lambda is not None else float(eta * params["tau"])

    # Resolve --all-optimizations into individual flags
    opt_flags = dict(
        score_guided=args.score_guided or args.all_optimizations,
        adaptive_w=args.adaptive_w or args.all_optimizations,
        lateral=args.lateral_moves or args.all_optimizations,
        diversify=args.diversify or args.all_optimizations,
        perturb=args.perturb_restart or args.all_optimizations,
        sequential_w=args.sequential_w or args.all_optimizations,
    )

    beta_eff = compute_beta_eff(params, args.leakage)

    _print_experiment_banner(args, params, beta_eff, fitness_mode,
                             fitness_lambda, opt_flags, verbose)

    csv_path = Path(args.output) if args.output else None
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    try:
        for key_idx in range(args.num_keys):
            if _interrupt_event.is_set():
                break

            print(f"--- Key {key_idx + 1}/{args.num_keys} ---")

            # --- Phase 1: Generate key and data ---
            x_true = rng.integers(-eta, eta + 1, size=n, dtype=np.int8)
            if verbose and print_keys:
                print(f"  True key: {format_key(x_true)}")

            t0 = perf_counter()
            z_tilde, C, total_sigs = generate_informative_relations(
                rng, x_true, args.inf_rels, params, args.leakage)
            t_datagen = perf_counter() - t0
            R = len(z_tilde)
            print(f"  Phase 1: {R} inf. rels from {total_sigs} signatures "
                  f"({t_datagen:.1f}s)")

            if _interrupt_event.is_set():
                break

            # --- Regression warm start ---
            t0 = perf_counter()
            x_hat_float, x_init = regression_warm_start(C, z_tilde, n, eta)
            t_regression = perf_counter() - t0

            init_correct = int(np.sum(x_init == x_true))
            init_accuracy = init_correct / n
            D_init = n - init_correct
            print(f"  Regression: {init_correct}/{n} correct "
                  f"({init_accuracy:.1%}), D_0={D_init} ({t_regression:.2f}s)")

            if verbose and print_keys:
                print(f"  Regression estimate: {format_key(x_init)}")

            # --- Score weights (Tier 1) ---
            score_weights = None
            if opt_flags["score_guided"]:
                score_weights = compute_score_weights(
                    x_hat_float, eta, temperature=args.score_temperature)
                if verbose:
                    top10 = np.argsort(score_weights)[-10:][::-1]
                    top10_w = score_weights[top10]
                    top10_str = list(zip(
                        top10.tolist(),
                        [f"{wt:.4f}" for wt in top10_w]
                    ))
                    print(f"  Score-guided: top-10 positions by weight: "
                          f"{top10_str}")

            if _interrupt_event.is_set():
                break

            # --- Phase 2: Hill-climbing ---
            t0 = perf_counter()
            x_final, F_final, iters_used, history, num_perturbs = hillclimb(
                C, z_tilde, x_init, params, rng,
                w=args.block_size, T=args.max_iter,
                leakage_index=args.leakage,
                true_key=x_true, verbose=verbose, print_keys=print_keys,
                num_workers=args.workers,
                fitness_mode=fitness_mode, fitness_lambda=fitness_lambda,
                score_weights=score_weights,
                use_adaptive_w=opt_flags["adaptive_w"],
                adaptive_w_max=args.adaptive_w_max,
                adaptive_w_patience=args.adaptive_w_patience,
                use_lateral_moves=opt_flags["lateral"],
                lateral_tabu_size=args.lateral_tabu_size,
                use_diversify=opt_flags["diversify"],
                diversify_strength=args.diversify_strength,
                sweep_interval=args.sweep_interval,
                use_perturb_restart=opt_flags["perturb"],
                perturb_strength=args.perturb_strength,
                perturb_patience=args.perturb_patience,
                perturb_max=args.perturb_max,
                perturb_score_guided=args.perturb_score_guided,
                use_sequential_w=opt_flags["sequential_w"])
            t_hillclimb = perf_counter() - t0

            # --- Result evaluation (+ optional MOSEK fallback) ---
            success = bool(np.array_equal(x_final, x_true))
            D_final = int(np.sum(x_final != x_true))
            t_mosek = 0.0
            mosek_attempted = False
            mosek_success = False

            if F_final == 0 and not success:
                print(f"  *** FALSE POSITIVE: F=0 but D={D_final} ***")

            elif success:
                perturb_info = (f", {num_perturbs} perturbation(s)"
                                if num_perturbs > 0 else "")
                print(f"  SUCCESS: key recovered in {iters_used} iterations"
                      f"{perturb_info} ({t_hillclimb:.2f}s)")

            else:
                perturb_info = (f", {num_perturbs} perturbation(s)"
                                if num_perturbs > 0 else "")
                print(f"  FAILURE: T={args.max_iter} reached, F={F_final}, "
                      f"D={D_final}{perturb_info} ({t_hillclimb:.2f}s)")

                # MOSEK ILP fallback
                if args.mosek and not _interrupt_event.is_set():
                    mosek_attempted = True
                    print(f"  Attempting MOSEK ILP recovery from "
                          f"hill-climbing warm start (D={D_final})...")
                    x_mosek, t_mosek = mosek_ilp_recovery(
                        C, z_tilde, x_final, params, args.leakage,
                        mosek_timeout=args.mosek_timeout, verbose=verbose)

                    if x_mosek is not None:
                        D_mosek = int(np.sum(x_mosek != x_true))
                        mosek_success = bool(
                            np.array_equal(x_mosek, x_true))
                        if mosek_success:
                            x_final = x_mosek
                            D_final = 0
                            F_final = 0
                            success = True
                            print(f"  SUCCESS (MOSEK): key recovered "
                                  f"({t_hillclimb:.2f}s climb + "
                                  f"{t_mosek:.2f}s ILP)")
                        else:
                            print(f"  MOSEK returned a solution but "
                                  f"D={D_mosek} (not the true key)")
                            if D_mosek < D_final:
                                print(f"  MOSEK improved D: "
                                      f"{D_final} -> {D_mosek}")
                                x_final = x_mosek
                                D_final = D_mosek
                    else:
                        print(f"  MOSEK ILP did not find a solution "
                              f"({t_mosek:.2f}s)")

            if verbose and print_keys:
                print(f"  True key: {format_key(x_true)}")

            # --- Collect result ---
            result = dict(
                key_idx=key_idx,
                variant=params["name"],
                leakage_index=args.leakage,
                r_informative=R,
                total_signatures=total_sigs,
                block_size=args.block_size,
                max_iter=args.max_iter,
                fitness_mode=fitness_mode,
                fitness_lambda=(fitness_lambda
                                if fitness_mode == "combined" else None),
                init_correct=init_correct,
                init_accuracy=init_accuracy,
                D_init=D_init,
                F_final=F_final,
                D_final=D_final,
                iterations=iters_used,
                num_perturbations=num_perturbs,
                success=success,
                t_datagen=t_datagen,
                t_regression=t_regression,
                t_hillclimb=t_hillclimb,
                t_mosek=t_mosek,
                mosek_attempted=mosek_attempted,
                mosek_success=mosek_success,
                **{f"opt_{k}": v for k, v in opt_flags.items()},
            )
            results.append(result)
            print()

    finally:
        interrupted = _interrupt_event.is_set()
        if results:
            _print_summary(results, args.num_keys, seed=args.seed, interrupted=interrupted)
            if csv_path:
                _write_csv(results, csv_path)
        elif interrupted:
            print("\n=== No keys completed before interrupt ===")

        signal.signal(signal.SIGINT, original_handler)

    return results


# ===================================================================
# CLI
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Hill-climbing ML-DSA key recovery experiment (v6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Fitness modes (--fitness):
  excess    Sum of excesses S(x) = sum of overshoots (L1, default)
  count     Violation count F(x) = |{violated eqs}| (L0)
  combined  M(x) = lambda*F(x) + S(x) (penalty + excess)

Optimization strategies:
  --score-guided      Tier 1: Bias position sampling toward uncertain positions
  --adaptive-w        Tier 2: VNS -- expand block size when stuck
  --lateral-moves     Tier 3a: Accept ties to drift along plateaus
  --diversify         Tier 3b: Penalise frequently selected positions
  --perturb-restart   Tier 4: ILS -- perturb & restart to escape plateaus
  --sequential-w      Tier 5: Sequentially sample all w_base-size subsets before repeating for higher w
  --all-optimizations Enable all of the above

MOSEK ILP fallback:
  --mosek             On failure, attempt exact recovery via MOSEK ILP
  --mosek-timeout N   Solver time limit in seconds (default: 300)

Examples:
  python hillclimb_mldsa_v6.py --params 44 --inf-rels 25000 --block-size 5
  python hillclimb_mldsa_v6.py --params 44 --inf-rels 25000 --block-size 5 \\
      --all-optimizations --mosek --mosek-timeout 120
        """,
    )

    # Core parameters
    core = parser.add_argument_group("Core parameters")
    core.add_argument("--params", type=int, choices=[44, 65, 87],
                      required=True, help="ML-DSA variant (44, 65, or 87)")
    core.add_argument("--leakage", type=int, default=8,
                      help="Leakage bit index (default: 8)")
    core.add_argument("--num-keys", type=int, default=10,
                      help="Number of random keys to test")
    core.add_argument("--inf-rels", type=int, required=True,
                      help="Number of informative relations to collect (r)")
    core.add_argument("--block-size", type=int, default=5,
                      help="Base block size w (positions per step)")
    core.add_argument("--max-iter", type=int, default=100000,
                      help="Maximum hill-climbing iterations T")
    core.add_argument("--output", type=str, default=None,
                      help="CSV output path")
    core.add_argument("--seed", type=int, default=None,
                      help="Random seed for reproducibility")
    core.add_argument("--non-verbose", action="store_true",
                      help="Suppress per-step output")
    core.add_argument("--print-keys", action="store_true",
                      help="Print intermediate key vectors (very verbose)")
    core.add_argument("--workers", type=int, default=1,
                      help="Threads for parallel candidate evaluation")

    # Fitness mode
    fitness_grp = parser.add_argument_group("Fitness function")
    fitness_grp.add_argument(
        "--fitness", type=str, default="excess",
        choices=["count", "excess", "combined"],
        help="Fitness mode (default: excess)")
    fitness_grp.add_argument(
        "--fitness-lambda", type=float, default=None,
        help="Penalty per violation for 'combined' mode (default: beta_eff)")

    # Optimization flags
    opt = parser.add_argument_group("Optimization strategies")
    opt.add_argument("--all-optimizations", action="store_true",
                     help="Enable all optimization strategies")
    opt.add_argument("--score-guided", action="store_true",
                     help="Tier 1: Score-guided position sampling")
    opt.add_argument("--score-temperature", type=float, default=2.0,
                     help="Temperature for score-guided softmax (default: 2.0)")
    opt.add_argument("--adaptive-w", action="store_true",
                     help="Tier 2: VNS-style adaptive block size")
    opt.add_argument("--adaptive-w-max", type=int, default=7,
                     help="Maximum block size during expansion (default: 7)")
    opt.add_argument("--adaptive-w-patience", type=int, default=50,
                     help="Iters without improvement before expanding w")
    opt.add_argument("--lateral-moves", action="store_true",
                     help="Tier 3a: Accept ties with tabu")
    opt.add_argument("--lateral-tabu-size", type=int, default=20,
                     help="Tabu list size (default: 20)")
    opt.add_argument("--diversify", action="store_true",
                     help="Tier 3b: Frequency-based diversification")
    opt.add_argument("--diversify-strength", type=float, default=1.0,
                     help="Penalty strength for frequent positions")
    opt.add_argument("--sweep-interval", type=int, default=0,
                     help="Forced sweep every K iterations (0 = off)")
    opt.add_argument("--perturb-restart", action="store_true",
                     help="Tier 4: ILS perturbation restarts")
    opt.add_argument("--perturb-strength", type=int, default=30,
                     help="Positions to reassign per perturbation")
    opt.add_argument("--perturb-patience", type=int, default=50,
                     help="Iters stuck before perturbing")
    opt.add_argument("--perturb-max", type=int, default=50,
                     help="Maximum perturbations per key")
    opt.add_argument("--perturb-score-guided", action="store_true",
                     help="Bias perturbation targets toward uncertain "
                          "positions")
    opt.add_argument("--sequential-w", action="store_true",
                     help="Sequential random position selection: pick w random "
                          "positions from available pool each iteration. "
                          "Increases w when all positions exhausted.")

    # MOSEK ILP fallback
    mosek_grp = parser.add_argument_group("MOSEK ILP fallback")
    mosek_grp.add_argument(
        "--mosek", action="store_true",
        help="On failure, attempt exact recovery via MOSEK ILP")
    mosek_grp.add_argument(
        "--mosek-timeout", type=float, default=300.0,
        help="MOSEK solver time limit in seconds (default: 300)")

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
