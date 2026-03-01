#!/usr/bin/env python3
"""
Hill-Climbing Key Recovery for ML-DSA via Verification Fitness Oracle
(Noisy Leakage Model)

Variant of the deterministic v8 attack adapted for the noisy leakage setting
(Damm et al., Section 6, Definition 7).  The leaked bit y_j is independently
flipped with error rate p (the --noise-level parameter), modelling realistic
side-channel noise.

Key adaptations for the noisy model:

  1. Noise injection:  Each leaked bit y_j is flipped with probability p.

  2. Outlier rejection:  After relation extraction, samples with
     |z_bar| > 2^{j-1} + beta are discarded.  These arise from
     noise-corrupted leak bits (Theorem 9) and fall outside the
     valid range for both correct and flipped extractions.

  3. Count (L0) fitness only:  F(x) = number of violated verification
     equations.  The excess (L1) metric is unsuitable because noise-
     corrupted relations produce errors of ~2*beta that dominate the
     sum and mislead the optimizer.

  4. OLS rescaling:  The regression warm start is rescaled by
     r = 1/(1-2p) to correct the bias E[x_hat] = (1-2p)*x
     (Theorem 12, Algorithm 3 of Damm et al.).

  5. Tuned defaults:  Hyperparameters adjusted for the harder noisy
     regime (higher patience values).

Optional MOSEK ILP fallback (--mosek) (currently disabled):
  When hill climbing converges (best-ever stagnation exceeds --patience), the best
  intermediate solution x' is handed to MOSEK's mixed-integer optimizer as a
  warm start for an Integer Linear Program.  The ILP encodes the verification
  relations as hard constraints and minimises the L1 distance to x'.

  Requires the MOSEK Python package (``pip install Mosek``) and a valid license.

Optional Gurobi ILP fallback (--gurobi) (currently disabled):
  Alternative ILP fallback using Gurobi as a pure feasibility solver (no
  objective function).  The hill-climbing estimate x' is injected via Gurobi's
  VarHintVal mechanism, which guides both heuristics and branching decisions
  throughout the MIP search.

  Requires the gurobipy package (``pip install gurobipy``) and a valid license.

Termination model:
  In the noisy regime, the true key violates some constraints (due to residual
  noise-corrupted relations), so F=0 is not a meaningful target.  The algorithm
  terminates when the global best-ever solution has not improved for --patience
  iterations (default: 10000), then returns that best-ever solution.  Iterated
  Local Search (ILS) perturbation is always active to prevent the optimizer from
  getting stuck in local optima induced by corrupted constraints.  The
  perturbation parameters --perturb-strength, --perturb-patience, and
  --perturb-max control ILS behaviour.  Note: only improvements to the global
  best-ever fitness reset the patience counter; ILS perturbations and local
  improvements that do not beat the best-ever do not reset it.

Optional optimization strategies (independently toggleable):

  Tier 1 -- Score-guided sampling (--score-guided):
    Uses regression residuals to bias position selection toward uncertain
    positions, dramatically increasing the hit rate on wrong positions.

  Tier 2 -- Adaptive block size / VNS (--adaptive-w):
    Variable Neighborhood Search: increases w when stuck, resets on progress.

  Tier 3a -- Accept lateral moves (--lateral-moves):
    Accepts F' == F to drift along plateaus.

  Tier 3b -- Frequency-based diversification (--diversify):
    Biases selection toward under-explored positions; periodic forced sweeps
    guarantee full coverage.

  Tier 4 -- w=1 sweep preamble (--sequential-w):
    Before each adaptive search phase, performs a batch parallel sweep over
    all n positions at block size w=1.  Positions are tested independently
    against a frozen snapshot of the current solution, then all improving
    changes are applied simultaneously.  Each full sweep counts as one
    iteration.  The sweep repeats until no further single-position
    improvements exist, then transitions to adaptive block search.

  --all-optimizations: Enables all of the above.
  --default-optimizations: Enables all except score-guided (recommended).

Graceful interruption:
  Ctrl+C during execution will finish the current iteration, then print
  summary statistics for all keys processed so far and write partial CSV
  output if --output was specified.  Ctrl+C also interrupts an active MOSEK
  solve gracefully via Model.breakSolver() or an active Gurobi solve via
  Model.terminate().

Usage:
  python hillclimb_mldsa_noisy.py --params 44 --noise-level 0.2 \\
      --inf-rels 25000 --block-size 2
  python hillclimb_mldsa_noisy.py --params 44 --noise-level 0.2 \\
      --inf-rels 25000 --block-size 2 --all-optimizations
  python hillclimb_mldsa_noisy.py --params 44 --noise-level 0.3 \\
      --inf-rels 50000 --block-size 2 --default-optimizations \\
      --gurobi --gurobi-timeout 120
"""

import argparse
import csv
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
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
_active_gurobi_model = None         # Set while Gurobi is solving
_active_gurobi_lock = threading.Lock()


def _sigint_handler(signum, frame):
    """Handle Ctrl+C: set interrupt flag and break active solver."""
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

    # If Gurobi is currently solving, terminate it immediately
    with _active_gurobi_lock:
        if _active_gurobi_model is not None:
            print("\n  [INTERRUPT] Ctrl+C -- terminating Gurobi solver...",
                  flush=True)
            _active_gurobi_model.terminate()
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
# Phase 1: Data generation (noisy leakage model)
# ===================================================================
def generate_informative_relations(rng, x_true, r_target, params,
                                   leakage_index, noise_level):
    """
    Simulate signature generation with noisy bit leakage and extract Integer
    LWE relations via the j-independence transformation (Damm et al., Eq. 3).

    In the noisy model (Definition 7), each leaked bit y_j is independently
    flipped with probability p = noise_level.  After relation extraction,
    two filters are applied:
      - Informativity filter: discard zero-knowledge samples |z_bar| <= Q - beta
      - Outlier rejection:    discard noise-corrupted samples |z_bar| > Q + beta
    The second filter removes identifiably corrupted relations that arise when
    the leaked bit was flipped (Theorem 9: z^(f) = z^(c) - sign(z^(c)) * 2^j,
    pushing |z^(f)| beyond Q + beta).

    Parameters:
        rng:            numpy random generator
        x_true:         (n,) true partial key (int8)
        r_target:       number of informative relations to collect
        params:         ML-DSA parameter dict
        leakage_index:  bit position j of the leaked randomness bit
        noise_level:    error rate p in [0, 0.5), probability of flipping y_j

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

    # Precompute constants for the relation extraction
    M = np.int64(2 ** (ell + 1))      # centered-mod modulus
    H = np.int64(2 ** ell)            # half-modulus
    Q = np.int64(2 ** (ell - 1))      # quarter-modulus
    threshold_lo = Q - beta            # informativity: |z_bar| > threshold_lo
    threshold_hi = Q + beta            # outlier rejection: |z_bar| <= threshold_hi
    reject_bound = np.int64(gamma - beta)

    n_collected = 0
    z_collected = []
    C_collected = []
    total_signatures = 0
    batch_size = max(r_target, 1000)

    while n_collected < r_target:
        remaining = r_target - n_collected
        n_batch = max(remaining * 5, batch_size)

        # --- Batch sample generation ---
        y_batch = rng.integers(-(gamma - 1), gamma, size=n_batch,
                               dtype=np.int64)
        y_j_batch = (y_batch >> ell) & 1               # true leaked bit

        # --- Noisy leakage: flip each bit with probability noise_level ---
        if noise_level > 0.0:
            flip_mask = rng.random(n_batch) < noise_level
            y_j_batch = np.where(flip_mask, y_j_batch ^ 1, y_j_batch)

        # Challenge vectors: tau distinct random positions per sample.
        rand = rng.random((n_batch, n))
        c_idx_batch = np.argpartition(rand, tau, axis=1)[:, :tau]
        c_signs_batch = (2 * rng.integers(0, 2, size=(n_batch, tau),
                                          dtype=np.int8) - 1)

        # Inner product <c, x_true> and signature coefficient z = y + <c, x>
        cx_batch = np.sum(x_true[c_idx_batch] * c_signs_batch, axis=1,
                          dtype=np.int64)
        z_batch = y_batch + cx_batch

        # --- Rejection sampling ---
        mask_accept = (z_batch > -reject_bound) & (z_batch < reject_bound)

        idx_acc = np.nonzero(mask_accept)[0]
        n_accepted_batch = len(idx_acc)
        if n_accepted_batch == 0:
            continue

        z_acc = z_batch[idx_acc]
        y_j_acc = y_j_batch[idx_acc]
        c_idx_acc = c_idx_batch[idx_acc]
        c_signs_acc = c_signs_batch[idx_acc]

        # --- Vectorized LWE relation extraction (Eq. 6, 7, 10) ---
        # Step 1: centered mod and bit test (Eq. 6 branching)
        z_raw = z_acc % M
        z_mod = np.where(z_raw > H, z_raw - M, z_raw)
        bit_j = (z_mod >> ell) & 1

        # Step 2: rotate and derive b_j (Eq. 6 & 7)
        z_up = np.where(bit_j == 1, z_acc + Q, z_acc - Q)
        b_j = np.where(bit_j == 1, y_j_acc, y_j_acc ^ 1)

        # Step 3: final z_bar via three-way branch (Eq. 10)
        z_up_raw = z_up % M
        z_up_mod = np.where(z_up_raw > H, z_up_raw - M, z_up_raw)
        bit_j_up = (z_up >> ell) & 1

        z_bar = np.where(
            b_j == 1,
            z_up_mod,
            np.where(bit_j_up == 1, z_up_mod + H, z_up_mod - H))

        # --- Informativity filter (remove zero-knowledge samples) ---
        abs_z_bar = np.abs(z_bar)
        mask_inf = abs_z_bar > threshold_lo

        # --- Outlier rejection (remove identifiably noise-corrupted) ---
        mask_bounds = abs_z_bar <= threshold_hi

        # Combined filter
        mask_keep = mask_inf & mask_bounds
        idx_keep = np.nonzero(mask_keep)[0]
        if len(idx_keep) == 0:
            # No informative relations in this batch, but all accepted
            # signatures were observed by the attacker.
            total_signatures += n_accepted_batch
            continue

        z_bar_kept = z_bar[idx_keep]
        c_idx_kept = c_idx_acc[idx_keep]
        c_signs_kept = c_signs_acc[idx_keep]

        # --- j-independence transformation (Eq. 3) ---
        if Q > beta:
            shift = np.int64(Q - beta)
            z_bar_kept = np.where(z_bar_kept >= shift,
                                  z_bar_kept - shift,
                                  z_bar_kept + shift)

        # --- Trim to what we still need and build dense challenge rows ---
        n_new = len(z_bar_kept)
        need = r_target - n_collected
        if n_new > need:
            # Only count accepted signatures up to the last needed relation.
            # idx_keep indexes into the accepted arrays; idx_keep[need-1] is
            # the position within the accepted batch of the last consumed
            # informative relation.  All accepted signatures from 0 to that
            # index (inclusive) were observed by the attacker.
            total_signatures += int(idx_keep[need - 1]) + 1

            z_bar_kept = z_bar_kept[:need]
            c_idx_kept = c_idx_kept[:need]
            c_signs_kept = c_signs_kept[:need]
            n_new = need
        else:
            # No trimming: all accepted signatures in this batch were consumed.
            total_signatures += n_accepted_batch

        C_new = np.zeros((n_new, n), dtype=np.int8)
        rows = np.repeat(np.arange(n_new), tau)
        C_new[rows, c_idx_kept.ravel()] = c_signs_kept.ravel()

        z_collected.append(z_bar_kept)
        C_collected.append(C_new)
        n_collected += n_new

    z_tilde = np.concatenate(z_collected)[:r_target].astype(np.float64)
    C = np.vstack(C_collected)[:r_target]
    return z_tilde, C, total_signatures


# ===================================================================
# Phase 1b: Regression warm start with noise rescaling
# ===================================================================
def regression_warm_start(C, z_tilde, n, eta, noise_level):
    """
    Run LSQR regression on the Integer LWE system C @ x ~ z_tilde,
    with bias correction for noisy leakage.

    By Theorem 12 of Damm et al., the OLS estimator has
    E[x_hat] = (1 - 2p) * x, so the estimate is biased toward zero.
    We correct this by rescaling: x_scaled = x_hat / (1 - 2p).

    Parameters:
        C:           (R, n) challenge matrix (int8)
        z_tilde:     (R,) transformed relation values (float64)
        n:           key dimension
        eta:         coefficient bound
        noise_level: error rate p

    Returns:
        x_hat_float:   (n,) rescaled OLS estimate (float64)
        x_hat_rounded: (n,) rounded and clipped to [-eta, eta] (int8)
    """
    C_csr = csr_matrix(C)
    x_hat_raw = lsqr(C_csr, z_tilde)[0]

    # Rescale to correct bias: E[x_hat] = (1-2p)*x  =>  x = x_hat / (1-2p)
    scale = 1.0 - 2.0 * noise_level
    if scale > 1e-6:
        x_hat_float = x_hat_raw / scale
    else:
        # Near p = 0.5, rescaling is numerically unstable; fall back to raw
        x_hat_float = x_hat_raw

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
# Fitness computation (count / L0 only for noisy model)
# ===================================================================
def _compute_fitness_scalar(ip, lb, ub, z_tilde_int=None, beta_eff=None,
                            modulus=None):
    """
    Compute violation count for a single candidate.

    Parameters:
        ip: (R,) inner product vector C @ x
        lb, ub: (R,) constraint bounds (used when modulus is None)
        z_tilde_int: (R,) int64 relation values (used when modulus is set)
        beta_eff: scalar effective error bound (used when modulus is set)
        modulus: int64, 2^{ell+1} when modular checking is needed, else None

    Returns:
        (F_count_float, F_count_int)
    """
    if modulus is not None:
        residual = (ip.astype(np.int64) - z_tilde_int) % modulus
        r_centered = np.where(residual > modulus // 2,
                              residual - modulus, residual)
        violated = np.abs(r_centered) > beta_eff
    else:
        violated = (ip < lb) | (ip > ub)
    F = int(np.count_nonzero(violated))
    return float(F), F


def _compute_fitness_batch(ip_batch, lb, ub, z_tilde_int=None, beta_eff=None,
                           modulus=None):
    """
    Compute violation counts for a batch of candidates.

    Parameters:
        ip_batch: (R, num_candidates) inner products
        lb, ub:   (R,) bounds (used when modulus is None)
        z_tilde_int: (R,) int64 relation values (used when modulus is set)
        beta_eff: scalar effective error bound (used when modulus is set)
        modulus: int64, 2^{ell+1} when modular checking is needed, else None

    Returns:
        fitness:  (num_candidates,) float64
        F_counts: (num_candidates,) int
    """
    if modulus is not None:
        residual = (ip_batch.astype(np.int64)
                    - z_tilde_int[:, np.newaxis]) % modulus
        r_centered = np.where(residual > modulus // 2,
                              residual - modulus, residual)
        violated = np.abs(r_centered) > beta_eff
    else:
        violated = ((ip_batch < lb[:, np.newaxis])
                    | (ip_batch > ub[:, np.newaxis]))
    F_counts = np.count_nonzero(violated, axis=0)
    return F_counts.astype(np.float64), F_counts


# ===================================================================
# Phase 2: Hill-climbing with optional optimizations
# ===================================================================
def _evaluate_candidate_chunk(C_block, ip_base, lb, ub, candidate_chunk,
                              chunk_offset, z_tilde_int=None, beta_eff=None,
                              modulus=None):
    """Evaluate fitness for a chunk of candidate assignments (thread worker)."""
    new_contrib = C_block @ candidate_chunk.astype(np.int32).T
    ip_new = ip_base[:, np.newaxis] + new_contrib
    fitness, F_counts = _compute_fitness_batch(ip_new, lb, ub,
                                               z_tilde_int=z_tilde_int,
                                               beta_eff=beta_eff,
                                               modulus=modulus)
    best_local = int(np.argmin(fitness))
    return (chunk_offset + best_local,
            float(fitness[best_local]),
            int(F_counts[best_local]))


def _precompute_candidates(values, w):
    """Precompute all (2*eta+1)^w candidate tuples for a given block size."""
    nv = len(values)
    indices = np.indices((nv,) * w).reshape(w, -1).T    # (nv^w, w)
    return values[indices]                                # (nv^w, w), same dtype as values



def _w1_sweep_worker(C_i32, ip_frozen, x_curr, lb, ub, values,
                     position_indices, z_tilde_int=None, beta_eff=None,
                     modulus=None):
    """Evaluate all (2*eta+1) candidates for each position in a batch.

    Each position is tested independently against the frozen inner-product
    vector ip_frozen.  This is the thread worker for the parallel w=1 sweep.

    Parameters:
        C_i32:            (R, n) int32 challenge matrix
        ip_frozen:        (R,) inner products at sweep start (frozen)
        x_curr:           (n,) current solution (int8, read-only)
        lb, ub:           (R,) constraint bounds (used when modulus is None)
        values:           (2*eta+1,) candidate values array
        position_indices: array of position indices to test
        z_tilde_int:      (R,) int64 relation values (when modulus is set)
        beta_eff:         scalar effective error bound (when modulus is set)
        modulus:          int64, 2^{ell+1} for modular checking, else None

    Returns:
        List of (position_index, best_value, best_fitness) for each position.
    """
    vals_i32 = values.astype(np.int32)
    results = []
    for j in position_indices:
        c_j = C_i32[:, j]                         # (R,)
        ip_base_j = ip_frozen - c_j * int(x_curr[j])  # (R,)

        # ip_base_j[:, None] + c_j[:, None] * vals_i32[None, :]  -> (R, nvals)
        ip_candidates = ip_base_j[:, np.newaxis] + np.outer(c_j, vals_i32)

        fitness_vals, _ = _compute_fitness_batch(ip_candidates, lb, ub,
                                                  z_tilde_int=z_tilde_int,
                                                  beta_eff=beta_eff,
                                                  modulus=modulus)

        best_idx = int(np.argmin(fitness_vals))
        results.append((int(j), int(values[best_idx]),
                         float(fitness_vals[best_idx])))
    return results


def hillclimb(C_i32, z_tilde, x_init, params, rng, w, patience,
              leakage_index,
              true_key=None, verbose=True, print_keys=False, num_workers=1,
              # Tier 1: Score-guided sampling
              score_weights=None,
              # Tier 2: Adaptive block size / VNS
              use_adaptive_w=False, adaptive_w_max=4, adaptive_w_patience=100,
              # Tier 3a: Lateral moves
              use_lateral_moves=False,
              # Tier 3b: Frequency diversification
              use_diversify=False, diversify_strength=1.0, sweep_interval=0,
              # ILS perturbation (always active)
              perturb_strength=30,
              perturb_patience=150, perturb_max=500,
              perturb_score_guided=False,
              # Tier 4: w=1 sweep preamble
              use_w1_sweep=False,
              w1_batch_size=16):
    """
    Hill-climbing key recovery using count (L0) fitness function,
    with Iterated Local Search (ILS) perturbation and optional
    optimization strategies.

    In the noisy model, the true key has F > 0 due to noise-corrupted
    relations.  The algorithm terminates when the global best-ever solution
    has not improved for ``patience`` consecutive iterations, then returns
    that best-ever solution.  Only improvements to the global best-ever
    fitness reset the patience counter; ILS perturbations and local
    improvements that do not surpass the best-ever are not counted.
    ILS perturbation is always active to escape local optima induced by
    corrupted constraints.

    w=1 sweep preamble (--sequential-w):
      When enabled, the algorithm begins (and restarts after each improvement
      or perturbation) with a batch parallel sweep over all n positions at
      w=1.  All positions are tested independently against a frozen snapshot
      of the current inner products, then all individually-improving changes
      are applied simultaneously.  Each sweep counts as one iteration.
      The sweep repeats until no positions improve, then transitions to
      adaptive block size search starting at w = max(block_size, 2).

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

    values = np.arange(-eta, eta + 1, dtype=np.int8)

    # When w=1 sweep is active, adaptive search starts at max(w, 2) after
    # the sweep, since w=1 was already exhaustively covered.
    w_adaptive_start = max(w, 2) if use_w1_sweep else w
    w_curr = w_adaptive_start

    # Candidate tuple cache (keyed by block size).
    # Stores (int8_tuples, int32_transposed) to avoid per-iteration casts.
    def _make_cached(block_size):
        tuples = _precompute_candidates(values, block_size)
        return tuples, tuples.astype(np.int32).T

    candidate_cache = {w_curr: _make_cached(w_curr)}

    def get_candidates(block_size):
        if block_size not in candidate_cache:
            candidate_cache[block_size] = _make_cached(block_size)
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
    # w=1 sweep state
    # ---------------------------------------------------------------
    in_w1_sweep = use_w1_sweep

    def _reset_w1_sweep():
        """Re-enter the w=1 sweep phase."""
        nonlocal in_w1_sweep, w_curr
        in_w1_sweep = True
        w_curr = 1  # will be set to w_adaptive_start when sweep finishes

    # ---------------------------------------------------------------
    # Mutable exploration state (reset on perturbation)
    # ---------------------------------------------------------------
    freq_counts = np.zeros(n, dtype=np.int64)
    sweep_permutation = None
    sweep_offset = 0
    iters_since_improvement = 0

    def _reset_soft_state():
        """Reset exploration state after a perturbation."""
        nonlocal freq_counts
        nonlocal sweep_permutation, sweep_offset, w_curr
        nonlocal iters_since_improvement
        freq_counts[:] = 0
        sweep_permutation = None
        sweep_offset = 0
        w_curr = w_adaptive_start
        iters_since_improvement = 0
        if use_w1_sweep:
            _reset_w1_sweep()

    # ---------------------------------------------------------------
    # Precompute constraint bounds (constant throughout)
    # ---------------------------------------------------------------
    lb = z_tilde - beta_eff
    ub = z_tilde + beta_eff

    # Modular constraint checking: when beta_eff < beta (low-leakage regime,
    # j-independence transformation not applied), the LWE extraction is modular
    # M=mod 2^{ell+1}, so constraint residuals must be reduced mod M before
    # comparison.  When beta_eff == beta (high-leakage), linear checks suffice.
    beta = params["eta"] * params["tau"]
    if beta_eff < beta:
        _modulus = np.int64(2 ** (leakage_index + 1))
        _z_tilde_int = np.round(z_tilde).astype(np.int64)
        _beta_eff_scalar = np.int64(beta_eff)
    else:
        _modulus = None
        _z_tilde_int = None
        _beta_eff_scalar = None

    # Current solution state
    x_curr = x_init.copy()
    ip = C_i32 @ x_curr.astype(np.int32)
    fitness_curr, F_curr = _compute_fitness_scalar(
        ip, lb, ub, z_tilde_int=_z_tilde_int, beta_eff=_beta_eff_scalar,
        modulus=_modulus)

    # History and logging
    history = []
    D_init = int(np.sum(x_curr != true_key)) if true_key is not None else -1
    history.append((0, F_curr, D_init))

    if verbose:
        print(f"  Iter 0: F={F_curr}, D={D_init}")
        if print_keys:
            print(f"    x* = {format_key(x_curr)}")

    # Best-ever tracking (survives perturbations)
    fitness_best_ever = fitness_curr
    F_best_ever = F_curr
    x_best_ever = x_curr.copy()
    ip_best_ever = ip.copy()

    num_perturbations = 0
    iters_since_best_ever = 0          # global patience: terminates when >= patience

    # Thread pool for parallel candidate evaluation
    use_parallel = num_workers > 1
    executor = (ThreadPoolExecutor(max_workers=num_workers)
                if use_parallel else None)

    iters_used = 0
    try:
        t = 0
        while iters_since_best_ever < patience:
            t += 1
            if _interrupt_event.is_set():
                break
            iters_used = t
            iters_since_best_ever += 1

            # ===== ILS perturbation check (always active) =====
            if (not in_w1_sweep
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
                        iters_since_best_ever = 0
                    else:
                        # Restore best-ever before perturbing
                        x_curr = x_best_ever.copy()
                        ip = ip_best_ever.copy()
                        fitness_curr = fitness_best_ever
                        F_curr = F_best_ever

                    # Apply perturbation
                    x_curr[perturb_pos] = rng.integers(
                        -eta, eta + 1, size=p, dtype=np.int8)
                    ip = C_i32 @ x_curr.astype(np.int32)
                    fitness_curr, F_curr = _compute_fitness_scalar(
                        ip, lb, ub, z_tilde_int=_z_tilde_int,
                        beta_eff=_beta_eff_scalar, modulus=_modulus)
                    _reset_soft_state()

                    D_now = (int(np.sum(x_curr != true_key))
                             if true_key is not None else -1)
                    if verbose:
                        print(f"    [ILS] Perturbation #{num_perturbations}: "
                              f"perturbed {p} positions -> F={F_curr}, "
                              f"D={D_now}  (best-ever F={F_best_ever})")
                        if print_keys:
                            print(f"    x* = {format_key(x_curr)}")

            # ===== Position selection =====
            in_sweep = False

            if in_w1_sweep:
                # --- Batch parallel w=1 sweep ---
                all_positions = np.arange(n, dtype=int)
                ip_frozen = ip.copy()

                # Split positions into chunks and evaluate in parallel
                n_chunks = max(1, int(np.ceil(n / w1_batch_size)))
                chunks = np.array_split(all_positions, n_chunks)

                if use_parallel:
                    futures = [
                        executor.submit(
                            _w1_sweep_worker, C_i32, ip_frozen, x_curr,
                            lb, ub, values, chunk,
                            z_tilde_int=_z_tilde_int,
                            beta_eff=_beta_eff_scalar, modulus=_modulus)
                        for chunk in chunks
                    ]
                    all_results = []
                    for f in futures:
                        all_results.extend(f.result())
                else:
                    all_results = _w1_sweep_worker(
                        C_i32, ip_frozen, x_curr, lb, ub, values,
                        all_positions, z_tilde_int=_z_tilde_int,
                        beta_eff=_beta_eff_scalar, modulus=_modulus)

                # Apply only strictly improving changes
                num_changed = 0
                for j, best_val, best_fit in all_results:
                    if best_fit < fitness_curr and best_val != int(x_curr[j]):
                        x_curr[j] = np.int8(best_val)
                        num_changed += 1

                # Recompute ip and fitness only if something changed
                if num_changed > 0:
                    ip = C_i32 @ x_curr.astype(np.int32)
                    fitness_new, F_new = _compute_fitness_scalar(
                        ip, lb, ub, z_tilde_int=_z_tilde_int,
                        beta_eff=_beta_eff_scalar, modulus=_modulus)
                    sweep_improved = fitness_new < fitness_curr
                    fitness_curr = fitness_new
                    F_curr = F_new
                else:
                    sweep_improved = False

                # Logging
                D_now = (int(np.sum(x_curr != true_key))
                         if true_key is not None else -1)
                history.append((t, F_curr, D_now))

                if verbose:
                    tag = " *" if sweep_improved else ""
                    print(f"  Iter {t}: F={F_curr}, D={D_now}{tag}"
                          f"  [w1-sweep: {num_changed} positions changed]")
                    if sweep_improved and print_keys:
                        print(f"    x* = {format_key(x_curr)}")

                if sweep_improved:
                    iters_since_improvement = 0
                    if fitness_curr < fitness_best_ever:
                        fitness_best_ever = fitness_curr
                        F_best_ever = F_curr
                        x_best_ever = x_curr.copy()
                        ip_best_ever = ip.copy()
                        iters_since_best_ever = 0
                    _reset_w1_sweep()
                else:
                    # No improvement -- transition to adaptive phase
                    in_w1_sweep = False
                    w_curr = w_adaptive_start
                    iters_since_improvement = 0
                    if verbose:
                        print(f"    [w1-sweep] no improvement, transitioning "
                              f"to adaptive w={w_curr}  "
                              f"({(2*eta+1)**w_curr} candidates/step)")
                continue  # sweep consumed this iteration

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
            candidate_tuples, candidates_i32T = get_candidates(actual_w)
            num_candidates = candidate_tuples.shape[0]

            C_block = C_i32[:, positions]
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
                        z_tilde_int=_z_tilde_int,
                        beta_eff=_beta_eff_scalar, modulus=_modulus)
                    for chunk, offset in zip(chunks, chunk_offsets)
                ]
                best_idx, fitness_best, F_best = min(
                    (f.result() for f in futures), key=lambda r: r[1])
            else:
                ip_new = ip_base[:, np.newaxis] + C_block @ candidates_i32T
                fitness_vals, F_counts = _compute_fitness_batch(
                    ip_new, lb, ub, z_tilde_int=_z_tilde_int,
                    beta_eff=_beta_eff_scalar, modulus=_modulus)
                best_idx = int(np.argmin(fitness_vals))
                fitness_best = float(fitness_vals[best_idx])
                F_best = int(F_counts[best_idx])

            # ===== Acceptance logic =====
            strict_improvement = fitness_best < fitness_curr
            lateral_move = (not strict_improvement
                            and use_lateral_moves
                            and fitness_best == fitness_curr)

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

            # ===== Stagnation tracking + adaptive block size =====
            if strict_improvement:
                iters_since_improvement = 0
                if fitness_curr < fitness_best_ever:
                    fitness_best_ever = fitness_curr
                    F_best_ever = F_curr
                    x_best_ever = x_curr.copy()
                    ip_best_ever = ip.copy()
                    iters_since_best_ever = 0
                # On improvement: restart w=1 sweep (if enabled) or reset
                # adaptive-w to base
                if use_w1_sweep:
                    _reset_w1_sweep()
                elif use_adaptive_w and w_curr != w_adaptive_start:
                    w_curr = w_adaptive_start
                    if verbose:
                        print(f"    [VNS] w reset to {w_curr}")
            else:
                iters_since_improvement += 1

            # Adaptive-w expansion (patience-based, only outside w=1 sweep)
            if (use_adaptive_w
                    and not in_w1_sweep
                    and iters_since_improvement >= adaptive_w_patience):
                new_w = min(w_curr + 1, adaptive_w_max, n)
                if new_w != w_curr:
                    w_curr = new_w
                    iters_since_improvement = 0
                    if verbose:
                        print(f"    [VNS] w expanded to {w_curr}  "
                              f"({(2*eta+1)**w_curr} candidates/step)")
                    continue  # start fresh with new w next iteration

            # ===== Logging =====
            D_now = (int(np.sum(x_curr != true_key))
                     if true_key is not None else -1)
            history.append((t, F_curr, D_now))

            if verbose:
                tag = " *" if strict_improvement else (
                    " ~" if (lateral_move and accepted) else "")
                extra_parts = []
                if in_sweep:
                    extra_parts.append("[sweep]")
                if use_adaptive_w and w_curr != w_adaptive_start:
                    extra_parts.append(f"[w={w_curr}]")
                extra = ("  " + ", ".join(extra_parts)) if extra_parts else ""
                print(f"  Iter {t}: F={F_curr}, D={D_now}{tag}"
                      f"  pos={sorted(positions.tolist())}{extra}")
                if accepted and print_keys:
                    print(f"    x* = {format_key(x_curr)}")

    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    # Revert to best-ever if current is worse
    if fitness_curr > fitness_best_ever:
        x_curr = x_best_ever
        ip = ip_best_ever
        F_curr = F_best_ever
        fitness_curr = fitness_best_ever

    D_final = int(np.sum(x_curr != true_key)) if true_key is not None else -1

    if verbose:
        perturb_info = (f", perturbations={num_perturbations}"
                        if num_perturbations > 0 else "")
        print(f"  --- Final: F={F_curr}, D={D_final}, "
              f"iters={iters_used}{perturb_info} ---")
        if print_keys:
            print(f"    x* = {format_key(x_curr)}")
            if true_key is not None:
                print(f"    x  = {format_key(true_key)}")

    return x_curr, F_curr, iters_used, history, num_perturbations


# ===================================================================
# MOSEK ILP fallback: exact key recovery via integer feasibility
# ===================================================================
def mosek_ilp_recovery(C_i32, z_tilde, x_warm, params, leakage_index,
                       mosek_timeout=300.0, verbose=True):
    """
    Attempt exact key recovery by solving an Integer Linear Program (ILP)
    encoding the verification relations as hard constraints.

    NOTE: In the noisy model, some constraints are violated by the true key
    (due to noise-corrupted relations that survived outlier rejection).
    The ILP may therefore be infeasible or find a non-true-key solution.
    The ILP remains useful when noise is low enough that the true key
    satisfies the vast majority of constraints.

    ILP formulation:
        minimise   sum_j d_j                         (L1 distance to warm start)
        subject to lb_i <= sum_j C[i,j] * x_j <= ub_i   for all relations i
                   d_j >= x_j - x'_j                    for all j
                   d_j >= -(x_j - x'_j)                 for all j
                   d_j >= 0                              for all j
                   x_j in {-eta, ..., eta}               for all j

    Parameters:
        C_i32:          (R, n) int32 challenge matrix
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
    R = C_i32.shape[0]

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
            C_mosek = Matrix.dense(C_i32.astype(np.float64))
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
            with _active_mosek_lock:
                _active_mosek_model = None

        t_mosek = perf_counter() - t0

        if _interrupt_event.is_set():
            print(f"  [MOSEK] Interrupted ({t_mosek:.2f}s)")

        # Extract solution
        try:
            M.acceptedSolutionStatus(AccSolutionStatus.Feasible)
            x_level = x.level()
            x_sol = np.round(x_level).astype(np.int8)
            x_sol = np.clip(x_sol, -eta, eta)

            # Post-solve verification
            ip_check = C_i32 @ x_sol.astype(np.int32)
            violated = int(np.sum((ip_check < lb) | (ip_check > ub)))

            if verbose:
                obj_val = int(np.sum(np.abs(
                    x_sol.astype(np.int32) - x_warm.astype(np.int32))))
                print(f"  [MOSEK] Solution found: L1 dist to warm start = "
                      f"{obj_val}, constraint violations = {violated} "
                      f"({t_mosek:.2f}s)")

            if violated > 0:
                print(f"  [MOSEK] WARNING: solution violates {violated} "
                      "constraints (expected in noisy model)")

            return x_sol, t_mosek

        except Exception:
            if verbose:
                print(f"  [MOSEK] No feasible solution found ({t_mosek:.2f}s)")
            return None, t_mosek


# ===================================================================
# Gurobi ILP fallback: exact key recovery via integer feasibility
# ===================================================================
def gurobi_ilp_recovery(C_i32, z_tilde, x_warm, params, leakage_index,
                        gurobi_timeout=300.0, norel_time=0.0,
                        solution_limit=1, verbose=True):
    """
    Attempt key recovery by solving a pure Integer Feasibility Problem
    with Gurobi, using variable hints from the hill-climbing estimate.

    NOTE: In the noisy model, the ILP may be infeasible because noise-
    corrupted relations impose contradictory constraints.

    Parameters:
        C_i32:          (R, n) int32 challenge matrix
        z_tilde:        (R,) transformed relation values
        x_warm:         (n,) best key estimate from hill climbing (int8)
        params:         ML-DSA parameter dict
        leakage_index:  bit position j of the leaked bit
        gurobi_timeout: solver time limit in seconds (default: 300)
        norel_time:     time budget for no-relaxation heuristic before B&B
        solution_limit: number of feasible solutions to find (default: 1)
        verbose:        print solver progress

    Returns:
        solutions: list of (n,) int8 arrays, or empty list
        t_gurobi:  float, wall-clock time spent in Gurobi
    """
    global _active_gurobi_model

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        print("  [GUROBI] ERROR: gurobipy not available. "
              "Install with: pip install gurobipy")
        return [], 0.0

    n = params["n"]
    eta = params["eta"]
    beta_eff = compute_beta_eff(params, leakage_index)
    R = C_i32.shape[0]

    lb = z_tilde - beta_eff
    ub = z_tilde + beta_eff

    if verbose:
        print(f"  [GUROBI] Building feasibility ILP: {n} integer vars, "
              f"{R} relations, eta={eta}, beta_eff={beta_eff}")
        if solution_limit > 1:
            print(f"  [GUROBI] Solution pool: up to {solution_limit} "
                  f"solutions requested")

    t0 = perf_counter()

    try:
        env = gp.Env(empty=True)
        if not verbose:
            env.setParam("OutputFlag", 0)
        env.start()

        with gp.Model("mldsa_key_recovery", env=env) as M:
            with _active_gurobi_lock:
                _active_gurobi_model = M

            try:
                # Decision variables: x_j in {-eta, ..., eta}
                x = M.addMVar(n, vtype=GRB.INTEGER,
                              lb=float(-eta), ub=float(eta), name="x")

                # Verification relation constraints: lb <= C @ x <= ub
                C_f64 = C_i32.astype(np.float64)
                M.addMConstr(C_f64, x, GRB.GREATER_EQUAL, lb,
                             name="verify_lb")
                M.addMConstr(C_f64, x, GRB.LESS_EQUAL, ub,
                             name="verify_ub")

                # No objective -- pure feasibility problem
                M.setObjective(0)

                # Variable hints from hill-climbing estimate
                x_warm_f = x_warm.astype(np.float64)
                for j in range(n):
                    x[j].VarHintVal = x_warm_f[j]

                # Solver parameters: feasibility-focused
                M.setParam("MIPFocus", 1)
                M.setParam("SolutionLimit", solution_limit)
                M.setParam("TimeLimit", gurobi_timeout)

                if norel_time > 0:
                    M.setParam("NoRelHeurTime", norel_time)

                # Solution pool for multi-solution enumeration
                if solution_limit > 1:
                    M.setParam("PoolSolutions", solution_limit)
                    M.setParam("PoolSearchMode", 2)

                if verbose:
                    print(f"  [GUROBI] Solving feasibility ILP "
                          f"(timeout={gurobi_timeout}s, "
                          f"norel={norel_time}s)...")

                M.optimize()

            finally:
                with _active_gurobi_lock:
                    _active_gurobi_model = None

            t_gurobi = perf_counter() - t0

            if _interrupt_event.is_set():
                print(f"  [GUROBI] Interrupted ({t_gurobi:.2f}s)")

            # Extract solutions
            solutions = []
            status = M.Status

            if status in (GRB.OPTIMAL, GRB.SOLUTION_LIMIT,
                          GRB.TIME_LIMIT, GRB.INTERRUPTED):
                n_found = M.SolCount
                if n_found == 0:
                    if verbose:
                        print(f"  [GUROBI] No feasible solution found "
                              f"({t_gurobi:.2f}s)")
                    return [], t_gurobi

                for s in range(n_found):
                    M.Params.SolutionNumber = s
                    x_level = x.Xn
                    x_sol = np.round(x_level).astype(np.int8)
                    x_sol = np.clip(x_sol, -eta, eta)

                    # Post-solve constraint verification
                    ip_check = C_i32 @ x_sol.astype(np.int32)
                    violated = int(np.sum(
                        (ip_check < lb) | (ip_check > ub)))

                    if violated > 0 and verbose:
                        print(f"  [GUROBI] Pool solution {s}: violates "
                              f"{violated} constraints")

                    solutions.append(x_sol)

                if verbose:
                    print(f"  [GUROBI] Found {n_found} feasible solution(s) "
                          f"({t_gurobi:.2f}s)")

                return solutions, t_gurobi

            elif status == GRB.INFEASIBLE:
                if verbose:
                    print(f"  [GUROBI] Problem proven infeasible "
                          f"({t_gurobi:.2f}s) -- "
                          f"expected in noisy model with high p")
                return [], t_gurobi
            else:
                if verbose:
                    print(f"  [GUROBI] Solver ended with status {status} "
                          f"({t_gurobi:.2f}s)")
                return [], t_gurobi

    except Exception as e:
        t_gurobi = perf_counter() - t0
        if verbose:
            print(f"  [GUROBI] Error: {e} ({t_gurobi:.2f}s)")
        return [], t_gurobi


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
    n_gurobi_attempted = sum(1 for r in results
                             if r.get("gurobi_attempted", False))
    n_gurobi_success = sum(1 for r in results
                           if r.get("gurobi_success", False))
    status = " (INTERRUPTED)" if interrupted else ""

    print(f"\n=== Summary: {n_success}/{n_done} keys recovered "
          f"(of {total_keys} planned){status} ===")

    if n_mosek_attempted > 0 or n_gurobi_attempted > 0:
        n_solver_success = n_mosek_success + n_gurobi_success
        n_climb_only = n_success - n_solver_success
        print(f"  Hill-climbing alone: {n_climb_only}")
        if n_mosek_attempted > 0:
            print(f"  MOSEK ILP fallback: "
                  f"{n_mosek_success}/{n_mosek_attempted}")
        if n_gurobi_attempted > 0:
            print(f"  Gurobi ILP fallback: "
                  f"{n_gurobi_success}/{n_gurobi_attempted}")

    print(f"  Avg initial accuracy: "
          f"{np.mean([r['init_accuracy'] for r in results]):.1%}")
    print(f"  Avg iterations: "
          f"{np.mean([r['iterations'] for r in results]):.0f}")
    print(f"  Avg perturbations: "
          f"{np.mean([r['num_perturbations'] for r in results]):.1f}")
    print(f"  Avg hill-climb time: "
          f"{np.mean([r['t_hillclimb'] for r in results]):.2f}s")

    if n_success > 0:
        succ = [r for r in results if r["success"]]
        print(f"  Avg D_final (successful): "
              f"{np.mean([r['D_final'] for r in succ]):.1f}")
        print(f"  Avg F_final (successful): "
              f"{np.mean([r['F_final'] for r in succ]):.1f}")
        if n_mosek_success > 0:
            mosek_succ = [r for r in results
                          if r.get("mosek_success", False)]
            print(f"  Avg MOSEK solve time (successful): "
                  f"{np.mean([r['t_mosek'] for r in mosek_succ]):.2f}s")
        if n_gurobi_success > 0:
            grb_succ = [r for r in results
                        if r.get("gurobi_success", False)]
            print(f"  Avg Gurobi solve time (successful): "
                  f"{np.mean([r['t_gurobi'] for r in grb_succ]):.2f}s")
            pool_counts = [r.get("gurobi_solutions_found", 0)
                           for r in grb_succ]
            if any(c > 1 for c in pool_counts):
                print(f"  Avg Gurobi pool solutions (successful): "
                      f"{np.mean(pool_counts):.1f}")

    fail_results = [r for r in results if not r["success"]]
    if fail_results:
        print(f"  Failed keys: {len(fail_results)}, "
              f"avg D_final="
              f"{np.mean([r['D_final'] for r in fail_results]):.1f}, "
              f"avg F_final="
              f"{np.mean([r['F_final'] for r in fail_results]):.1f}, "
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


def _print_experiment_banner(args, params, beta_eff, opt_flags, verbose):
    """Print the experiment configuration banner."""
    beta = params["eta"] * params["tau"]

    print(f"=== Hill-Climbing Experiment (Noisy Model): "
          f"{params['name']} ===")
    print(f"  Noise level: {args.noise_level:.2%}")
    print(f"  Leakage index: {args.leakage}")
    if beta_eff < beta:
        print(f"  Effective error bound: beta_eff={beta_eff} "
              f"(< beta={beta}, j-independence transform skipped)")
        print(f"  Modular constraint checking: active "
              f"(mod {2 ** (args.leakage + 1)})")
    print(f"  Informative relations: {args.inf_rels}")
    print(f"  Block size w: {args.block_size}")
    print(f"  Patience (best-ever stagnation): {args.patience}")
    print(f"  Number of keys: {args.num_keys}")
    print(f"  Seed: {args.seed}")
    print(f"  Verbose: {verbose}")
    print(f"  Fitness: count (violation count, L0)")

    #TODO
    args.mosek = False
    args.gurobi = False
    if args.mosek:
        print(f"  MOSEK ILP fallback: enabled "
              f"(timeout={args.mosek_timeout}s)")

    if args.gurobi:
        norel_str = (f", norel={args.gurobi_norel_time}s"
                     if args.gurobi_norel_time > 0 else "")
        pool_str = (f", pool={args.gurobi_solution_limit}"
                    if args.gurobi_solution_limit > 1 else "")
        print(f"  Gurobi ILP fallback: enabled "
              f"(timeout={args.gurobi_timeout}s{norel_str}{pool_str})")

    # ILS perturbation (always active)
    target = "score-guided" if args.perturb_score_guided else "uniform"
    print(f"  ILS perturbation: always active (strength={args.perturb_strength}, "
          f"patience={args.perturb_patience}, "
          f"max={args.perturb_max}, target={target})")

    if args.workers > 1:
        print(f"  Workers: {args.workers}")

    # Optional optimizations
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
            f"lateral-moves")
    if opt_flags["diversify"]:
        active_opts.append(
            f"diversify (strength={args.diversify_strength}, "
            f"sweep={args.sweep_interval})")
    if opt_flags["sequential_w"]:
        active_opts.append(
            f"w1-sweep (batch={args.w1_batch_size}, "
            f"adaptive from w={max(args.block_size, 2)})")
    print(f"  Optimizations: {', '.join(active_opts) or 'none (baseline)'}")
    print()


# ===================================================================
# Per-key result evaluation and solver fallbacks
# ===================================================================
def _evaluate_key_result(x_final, F_final, iters_used, num_perturbs,
                         x_true, C_i32, z_tilde, beta_eff, params, args,
                         t_hillclimb, t_datagen, t_regression,
                         key_idx, R, total_sigs,
                         init_correct, init_accuracy, D_init,
                         opt_flags, verbose, print_keys):
    """Evaluate hill-climbing result, run solver fallbacks, return result dict.

    In the noisy model, the algorithm terminates when the global best-ever
    solution has not improved for ``patience`` consecutive iterations, and returns
    that best-ever solution.  Success is determined solely by whether the
    output matches the true key (known only in simulation).  The attacker's
    output is always a single point estimate.
    """
    D_final = int(np.sum(x_final != x_true))
    success = bool(np.array_equal(x_final, x_true))
    t_mosek = 0.0
    mosek_attempted = False
    mosek_success = False
    t_gurobi = 0.0
    gurobi_attempted = False
    gurobi_success = False
    gurobi_solutions_found = 0

    perturb_info = (f", {num_perturbs} perturbation(s)"
                    if num_perturbs > 0 else "")

    if success:
        print(f"  SUCCESS: key recovered, F={F_final}, "
              f"{iters_used} iterations{perturb_info} "
              f"({t_hillclimb:.2f}s)")
    else:
        print(f"  Hill-climbing finished: F={F_final}, D={D_final}"
              f"{perturb_info} ({t_hillclimb:.2f}s)")

        # MOSEK ILP fallback
        if args.mosek and not _interrupt_event.is_set():
            mosek_attempted = True
            print(f"  Attempting MOSEK ILP recovery from "
                  f"hill-climbing warm start (D={D_final})...")
            x_mosek, t_mosek = mosek_ilp_recovery(
                C_i32, z_tilde, x_final, params, args.leakage,
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

        # Gurobi ILP fallback
        if (args.gurobi and not success
                and not _interrupt_event.is_set()):
            gurobi_attempted = True
            sol_str = (f", pool={args.gurobi_solution_limit}"
                       if args.gurobi_solution_limit > 1 else "")
            print(f"  Attempting Gurobi feasibility ILP from "
                  f"hill-climbing warm start "
                  f"(D={D_final}{sol_str})...")
            grb_solutions, t_gurobi = gurobi_ilp_recovery(
                C_i32, z_tilde, x_final, params, args.leakage,
                gurobi_timeout=args.gurobi_timeout,
                norel_time=args.gurobi_norel_time,
                solution_limit=args.gurobi_solution_limit,
                verbose=verbose)

            gurobi_solutions_found = len(grb_solutions)

            if grb_solutions:
                for s_idx, x_grb in enumerate(grb_solutions):
                    D_grb = int(np.sum(x_grb != x_true))
                    if np.array_equal(x_grb, x_true):
                        gurobi_success = True
                        x_final = x_grb
                        D_final = 0
                        F_final = 0
                        success = True
                        if gurobi_solutions_found > 1:
                            print(
                                f"  SUCCESS (Gurobi): key is pool "
                                f"solution {s_idx} of "
                                f"{gurobi_solutions_found} "
                                f"({t_hillclimb:.2f}s climb + "
                                f"{t_gurobi:.2f}s ILP)")
                        else:
                            print(
                                f"  SUCCESS (Gurobi): key recovered "
                                f"({t_hillclimb:.2f}s climb + "
                                f"{t_gurobi:.2f}s ILP)")
                        break
                    elif verbose:
                        print(
                            f"  Gurobi pool solution {s_idx}: "
                            f"D={D_grb}")

                if not gurobi_success:
                    distances = [int(np.sum(x_g != x_true))
                                 for x_g in grb_solutions]
                    best_idx = int(np.argmin(distances))
                    D_grb_best = distances[best_idx]
                    print(
                        f"  Gurobi found {gurobi_solutions_found} "
                        f"solution(s), none is the true key "
                        f"(best D={D_grb_best})")
                    if D_grb_best < D_final:
                        print(f"  Gurobi improved D: "
                              f"{D_final} -> {D_grb_best}")
                        x_final = grb_solutions[best_idx]
                        D_final = D_grb_best
            else:
                print(f"  Gurobi ILP did not find a solution "
                      f"({t_gurobi:.2f}s)")

    if verbose and print_keys:
        print(f"  True key: {format_key(x_true)}")

    return dict(
        key_idx=key_idx,
        variant=params["name"],
        noise_level=args.noise_level,
        leakage_index=args.leakage,
        r_informative=R,
        total_signatures=total_sigs,
        block_size=args.block_size,
        patience=args.patience,
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
        t_gurobi=t_gurobi,
        gurobi_attempted=gurobi_attempted,
        gurobi_success=gurobi_success,
        gurobi_solutions_found=gurobi_solutions_found,
        **{f"opt_{k}": v for k, v in opt_flags.items()},
    )


# ===================================================================
# Main experiment loop
# ===================================================================
def run_experiment(args):
    """Run the full key-recovery experiment."""
    # Install signal handler
    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint_handler)
    _interrupt_event.clear()

    rng = np.random.default_rng(args.seed)
    params = MLDSA_PARAMS[args.params]
    n = params["n"]
    eta = params["eta"]
    verbose = not args.non_verbose
    print_keys = args.print_keys

    beta_eff = compute_beta_eff(params, args.leakage)

    # Resolve --all-optimizations / --default-optimizations
    all_opt = args.all_optimizations
    dflt_opt = args.default_optimizations
    opt_flags = dict(
        score_guided=args.score_guided or all_opt,
        adaptive_w=args.adaptive_w or all_opt or dflt_opt,
        lateral=args.lateral_moves or all_opt or dflt_opt,
        diversify=args.diversify or all_opt or dflt_opt,
        sequential_w=args.sequential_w or all_opt or dflt_opt,
    )

    _print_experiment_banner(args, params, beta_eff, opt_flags, verbose)

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
                rng, x_true, args.inf_rels, params, args.leakage,
                args.noise_level)
            t_datagen = perf_counter() - t0
            R = len(z_tilde)
            C_i32 = C.astype(np.int32)         # single cast for all consumers
            print(f"  Phase 1: {R} inf. rels from {total_sigs} signatures "
                  f"({t_datagen:.1f}s)")

            if _interrupt_event.is_set():
                break

            # --- Regression warm start with noise rescaling ---
            t0 = perf_counter()
            x_hat_float, x_init = regression_warm_start(
                C, z_tilde, n, eta, args.noise_level)
            t_regression = perf_counter() - t0

            init_correct = int(np.sum(x_init == x_true))
            init_accuracy = init_correct / n
            D_init = n - init_correct
            print(f"  Regression: {init_correct}/{n} correct "
                  f"({init_accuracy:.1%}), D_0={D_init} "
                  f"(rescaled by r={1/(1-2*args.noise_level):.2f}) "
                  f"({t_regression:.2f}s)")

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
                C_i32, z_tilde, x_init, params, rng,
                w=args.block_size, patience=args.patience,
                leakage_index=args.leakage,
                true_key=x_true, verbose=verbose, print_keys=print_keys,
                num_workers=args.workers,
                score_weights=score_weights,
                use_adaptive_w=opt_flags["adaptive_w"],
                adaptive_w_max=args.adaptive_w_max,
                adaptive_w_patience=args.adaptive_w_patience,
                use_lateral_moves=opt_flags["lateral"],
                use_diversify=opt_flags["diversify"],
                diversify_strength=args.diversify_strength,
                sweep_interval=args.sweep_interval,
                perturb_strength=args.perturb_strength,
                perturb_patience=args.perturb_patience,
                perturb_max=args.perturb_max,
                perturb_score_guided=args.perturb_score_guided,
                use_w1_sweep=opt_flags["sequential_w"],
                w1_batch_size=args.w1_batch_size)
            t_hillclimb = perf_counter() - t0

            # --- Result evaluation, solver fallbacks, result collection ---
            result = _evaluate_key_result(
                x_final, F_final, iters_used, num_perturbs,
                x_true, C_i32, z_tilde, beta_eff, params, args,
                t_hillclimb, t_datagen, t_regression,
                key_idx, R, total_sigs,
                init_correct, init_accuracy, D_init,
                opt_flags, verbose, print_keys)
            results.append(result)
            print()

    finally:
        interrupted = _interrupt_event.is_set()
        if results:
            _print_summary(results, args.num_keys, seed=args.seed,
                           interrupted=interrupted)
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
        description="Hill-climbing ML-DSA key recovery -- noisy leakage model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Noise model:
  The leaked bit y_j is flipped with probability p = --noise-level.
  Fitness uses count (L0) only.  Outlier rejection discards relations
  with |z_bar| > 2^{j-1} + beta.  OLS warm start is rescaled by
  r = 1/(1-2p) to correct the bias E[x_hat] = (1-2p)*x.

Optimization strategies:
  --score-guided      Tier 1: Bias position sampling toward uncertain positions
  --adaptive-w        Tier 2: VNS -- expand block size when stuck
  --lateral-moves     Tier 3a: Accept ties to drift along plateaus
  --diversify         Tier 3b: Penalise frequently selected positions
  --sequential-w      Tier 4: w=1 sweep preamble before adaptive search
  --all-optimizations Enable all of the above
  --default-optimizations  Enable all except score-guided (recommended)

ILS perturbation (always active):
  Perturbation restarts are always enabled in the noisy model to prevent
  the optimizer from getting stuck.  Use --perturb-strength, --perturb-patience,
  and --perturb-max to tune behaviour.

MOSEK ILP fallback (currently disabled):
  --mosek             On failure, attempt exact recovery via MOSEK ILP
  --mosek-timeout N   Solver time limit in seconds (default: 300)

Gurobi ILP fallback (currently disabled):
  --gurobi                   On failure, attempt recovery via Gurobi ILP
  --gurobi-timeout N         Solver time limit in seconds (default: 300)
  --gurobi-norel-time N      No-relaxation heuristic budget (default: 0)
  --gurobi-solution-limit K  Find up to K solutions (default: 1)

Examples:
  python hillclimb_mldsa_noisy.py --params 44 --noise-level 0.2 \\
      --inf-rels 25000 --block-size 2
  python hillclimb_mldsa_noisy.py --params 44 --noise-level 0.2 \\
      --inf-rels 25000 --block-size 2 --all-optimizations
  python hillclimb_mldsa_noisy.py --params 44 --noise-level 0.3 \\
      --inf-rels 50000 --default-optimizations \\
      --gurobi --gurobi-timeout 120
        """,
    )

    # Core parameters
    core = parser.add_argument_group("Core parameters")
    core.add_argument("--params", type=int, choices=[44, 65, 87],
                      required=True, help="ML-DSA variant (44, 65, or 87)")
    core.add_argument("--noise-level", type=float, required=True,
                      help="Error rate p in [0, 0.5): probability of "
                           "flipping each leaked bit (REQUIRED)")
    core.add_argument("--leakage", type=int, default=8,
                      help="Leakage bit index (default: 8)")
    core.add_argument("--num-keys", type=int, default=10,
                      help="Number of random keys to test")
    core.add_argument("--inf-rels", type=int, required=True,
                      help="Number of informative relations to collect (r)")
    core.add_argument("--block-size", type=int, default=2,
                      help="Base block size w for adaptive search "
                           "(default: 2)")
    core.add_argument("--patience", type=int, default=10000,
                      help="Global best-ever stagnation patience: terminate "
                           "when the best-ever key estimate has not improved "
                           "for this many iterations (default: 10000)")
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

    # Optimization flags
    opt = parser.add_argument_group("Optimization strategies")
    opt.add_argument("--all-optimizations", action="store_true",
                     help="Enable all optional optimization strategies "
                          "(score-guided, adaptive-w, lateral, diversify, "
                          "sequential-w). ILS perturbation is always active.")
    opt.add_argument("--default-optimizations", action="store_true",
                     help="Enable all optional optimizations except "
                          "score-guided sampling (recommended starting point). "
                          "ILS perturbation is always active.")
    opt.add_argument("--score-guided", action="store_true",
                     help="Tier 1: Score-guided position sampling")
    opt.add_argument("--score-temperature", type=float, default=2.0,
                     help="Temperature for score-guided softmax (default: 2.0)")
    opt.add_argument("--adaptive-w", action="store_true",
                     help="Tier 2: VNS-style adaptive block size")
    opt.add_argument("--adaptive-w-max", type=int, default=4,
                     help="Maximum block size during expansion (default: 4)")
    opt.add_argument("--adaptive-w-patience", type=int, default=100,
                     help="Iters without improvement before expanding w "
                          "(default: 100)")
    opt.add_argument("--lateral-moves", action="store_true",
                     help="Tier 3a: Accept ties to drift along plateaus")
    opt.add_argument("--diversify", action="store_true",
                     help="Tier 3b: Frequency-based diversification")
    opt.add_argument("--diversify-strength", type=float, default=1.0,
                     help="Penalty strength for frequent positions")
    opt.add_argument("--sweep-interval", type=int, default=0,
                     help="Forced sweep every K iterations (0 = off)")
    opt.add_argument("--sequential-w", action="store_true",
                     help="Tier 4: w=1 sweep preamble before adaptive search. "
                          "Exhaustively tests every position individually, "
                          "then transitions to adaptive block size.")
    opt.add_argument("--w1-batch-size", type=int, default=16,
                     help="Positions per thread during w=1 sweep (default: 16)")

    # ILS perturbation (always active)
    ils = parser.add_argument_group("ILS perturbation (always active)")
    ils.add_argument("--perturb-strength", type=int, default=30,
                     help="Positions to reassign per perturbation (default: 30)")
    ils.add_argument("--perturb-patience", type=int, default=150,
                     help="Iters stuck before perturbing (default: 150)")
    ils.add_argument("--perturb-max", type=int, default=500,
                     help="Maximum perturbations per key (default: 500)")
    ils.add_argument("--perturb-score-guided", action="store_true",
                     help="Bias perturbation targets toward uncertain "
                          "positions (requires --score-guided)")

    # MOSEK ILP fallback
    #mosek_grp = parser.add_argument_group("MOSEK ILP fallback")
    #mosek_grp.add_argument(
    #    "--mosek", action="store_true",
    #    help="On failure, attempt exact recovery via MOSEK ILP")
    #mosek_grp.add_argument(
    #    "--mosek-timeout", type=float, default=300.0,
    #    help="MOSEK solver time limit in seconds (default: 300)")

    ## Gurobi ILP fallback
    #gurobi_grp = parser.add_argument_group("Gurobi ILP fallback")
    #gurobi_grp.add_argument(
    #    "--gurobi", action="store_true",
    #    help="On failure, attempt recovery via Gurobi feasibility ILP")
    #gurobi_grp.add_argument(
    #    "--gurobi-timeout", type=float, default=300.0,
    #    help="Gurobi solver time limit in seconds (default: 300)")
    #gurobi_grp.add_argument(
    #    "--gurobi-norel-time", type=float, default=0.0,
    #    help="No-relaxation heuristic time budget in seconds (default: 0 = "
    #         "Gurobi default). Runs a feasibility heuristic before B&B; "
    #         "terminates immediately on finding a feasible point.")
    #gurobi_grp.add_argument(
    #    "--gurobi-solution-limit", type=int, default=1,
    #    help="Number of feasible solutions to find (default: 1). "
    #         "When >1, activates Gurobi's solution pool to enumerate "
    #         "multiple distinct keys.")

    args = parser.parse_args()

    # Validate noise level
    if not (0.0 <= args.noise_level < 0.5):
        parser.error("--noise-level must be in [0, 0.5)")

    run_experiment(args)


if __name__ == "__main__":
    main()
