# AGENTS.md — Project Knowledge Base

## Project Overview

Finite-N contagion dynamics on low-rank exposure networks. Implements and
validates exact rank-K closure: the direct O(N²) Euler solver and the reduced
O(NK) feedback solver produce bit-identical results (machine precision).

## Repository Layout

```
src/
  model.py          — loss functions (hard, pos/neg ramp), simulate_direct, threshold_fraction
  low_rank.py       — build_factorised_exposure, simulate_reduced, solve_population_ode,
                      population_threshold_fraction, gaussian_mixture_ode_rhs
  sampling.py       — build_quantile_matched_network, allocate_group_counts,
                      quantile_initial_conditions, iid_type_sample
  metrics.py        — sup_scalar_error, sup_vector_error_inf, sup_feedback_error,
                      threshold_fraction_path, group_threshold_fractions, wasserstein1_empirical
  contagion_solver.py — run_direct_and_reduced, validate_direct_reduced_equivalence,
                        step_halving_check, sensitivity_sweep
  configs/
    exp1_directedness.json — canonical rank-1 two-group config (G=2,K=1,T=2.0,dt=5e-4)

exp/
  run_exp1.py  — Exact low-rank closure validation experiment
  run_exp2.py  — Directedness and multilayer amplification experiment

results/         — output CSVs and PNGs (auto-created by scripts)
```

## Key API Signatures

```python
# model.py
simulate_direct(x0, E, mu, r, T, dt, loss_fn) -> (times, states)
  # E is (N,N); states is (n_steps+1, N)

# low_rank.py
build_factorised_exposure(A, B) -> E   # E[i,j] = (A[i] @ B[j]) / K
simulate_reduced(x0, A, B, mu, r, T, dt, loss_fn) -> (times, states, betas)
  # betas is (n_steps+1, K)
solve_population_ode(p, m_g, sigma_g, A_g, B_g, mu, T, dt) -> (times, betas)
  # ONLY valid for r=0; A_g/B_g are (G, K) group-level loadings
population_threshold_fraction(betas, times, p, m_g, sigma_g, A_g, mu) -> h
  # h is (n_steps+1,) aggregate limit threshold fraction

# sampling.py
build_quantile_matched_network(N, p, m_g, sigma_g, A_g, B_g)
  -> (x0, A, B, group_ids)   # A/B are (N,K); group_ids is (N,)

# metrics.py
sup_scalar_error(path_a, path_b) -> float
sup_vector_error_inf(states_a, states_b) -> float
sup_feedback_error(betas_a, betas_b) -> float
threshold_fraction_path(states) -> (n_steps+1,)
group_threshold_fractions(states, group_ids, G) -> (n_steps+1, G)

# contagion_solver.py
validate_direct_reduced_equivalence(x0, A, B, mu, r, T, dt, loss_fn, tol=1e-10)
  -> (passed: bool, max_err: float)
step_halving_check(x0, E, mu, r, T, dt, loss_fn) -> float  # discrepancy
run_direct_and_reduced(x0, A, B, mu, r, T, dt, loss_fn) -> dict
  # dict keys: times, states_direct, states_reduced, betas_reduced,
  #            h_direct, h_reduced, err_states_inf, err_states_mean, err_h
```

## Canonical Config (exp1_directedness.json)

```
G=2, K=1, T=2.0, dt=5e-4, mu=0.0, r=0.0
p=[0.5, 0.5], m_g=[0.50, 0.45], sigma_g=[0.18, 0.18]
A_g=[[2.0], [0.6]], B_g=[[0.6], [2.0]]
N_values=[400, 1600]
```

## Validated Numerical Results (Exp 1, r=0)

| N    | direct/reduced err | sup |h_N-h_lim| | sup ||β_N-β_ode|| | step-halving |
|------|--------------------|-----------------|---------------------|--------------|
| 400  | ~1.2e-15           | ~8.8e-4         | ~2.5e-3             | ~5.4e-15     |
| 1600 | ~4.4e-15           | ~4.9e-4         | ~4.6e-4             | ~3.8e-7      |

Population limit h(T) ≈ 0.00588 (~0.6% distress).

## Validated Numerical Results (Exp 2, Directedness)

| r     | G0 lender h_T | G1 borrower h_T | separation |
|-------|---------------|-----------------|------------|
| 0.000 | 0.0050        | 0.0050          | 0.0000     |
| 0.050 | 0.0000        | 0.1050          | 0.1050     |

r=0.05 dramatically amplifies lender/borrower separation (~10.5% gap).

## Important Notes

- `solve_population_ode` / `population_threshold_fraction` are only valid for r=0.
  For r>0 experiments, use a large-N (N=1600) finite-N simulation as a proxy.
- `sys.path.insert(0, str(Path(__file__).parent.parent))` before `from src.xxx import ...`
- All scripts use `matplotlib.use("Agg")` (headless rendering).
- Results directory (`results/`) is created automatically by `mkdir(parents=True, exist_ok=True)`.
- Equivalence tolerance: direct/reduced solver state error < 1e-10 (typically ~1e-15).

## EBA Experiments (exp5, exp6)

### Source Modules Used
- `eba_loader`: `load_six_bank_data`, `load_six_bank_full_book`, `generate_synthetic_116`
- `empirical_kernel`: `build_empirical_loadings`, `build_empirical_exposure`,
  `build_initial_buffers`, `build_full_book_loadings`, `add_other_factor`,
  `perturb_italian_holdings`
- `metrics`: also uses `svd_truncate`, `frobenius_relative_error`, `wasserstein1_empirical`,
  `loglog_slope`, `sup_feedback_error`
- `sampling`: `resample_from_empirical`

### Critical Formula: Direct ↔ Reduced Equivalence with lambda_E Scaling
`simulate_direct` uses `E = lambda_E * A @ B.T`; `simulate_reduced` uses `(A_r @ B_r.T) / K`.
To get identical trajectories, scale the loadings before calling `simulate_reduced`:
```python
scale = np.sqrt(K * lambda_E)
A_r, B_r = scale * A, scale * B
```
Verified < 1e-10 for exp5 (N=6, K=4) and exp6 (N=116, K=38).

### Exp5 Parameters (six-bank, N=6, K=4)
- Banks: BNP, Deutsche, UniCredit, Santander, ING, Intesa
- Factors: France(0), Germany(1), Italy(2), Netherlands(3)
- `lambda_E=0.075`, `x_star=0.40`, `eta=[0.35, 0.15, 0.45, 0.05]`
- `eps=0.05`, `T=3.0`, `dt=1e-3`, `mu=0`, `r=0`
- Full-book: `load_six_bank_full_book()` returns `(S_full, full_row_totals=total_4*1.2)`
  → use `build_full_book_loadings(S, full_row_totals)` for A_fb; keep B from original S
- Intensity sweep: just scale `E_sw = lam * (A @ B.T)`, keep x0 fixed
- SVD truncation: `svd_truncate(E, rank)` from metrics; simulate_direct with truncated E

### Exp6 Parameters (116-institution, N=116, K=38)
- Synthetic data: `generate_synthetic_116(seed=42)`
- Factor ordering: France(0), Germany(1), Italy(2), Spain(3), Netherlands(4), ...
- **eta vector**: `eta[0]=0.35, eta[1]=0.15, eta[2]=0.45, eta[4]=0.05`, rest 0
  (Netherlands is index 4, not 3, in the 38-factor list)
- `lambda_E=0.075`, `x_star=0.40`, `eps=0.05`, `T=3.0`, `dt=1e-3`
- Other sovereigns: `full_row_totals = S.sum(axis=1) / 0.99704`; `add_other_factor(S, full_row_totals)`
- Alt normalization: `lambda_E_alt = 0.30/K` (c_E=0.30 fixed)
- Italy perturbation: `perturb_italian_holdings(S, italy_col=2, delta=delta)`
- Scenario-aligned surrogate: `stress_cols=[0,1,2,4]`; build E_4 from S[:,stress_cols];
  `svd_truncate(E_4, 2)` gives scenario-aligned rank-2 approximation
- Resampling: `resample_from_empirical(N_s, x0, A, B, rng)` from sampling; solve reduced;
  measure `sup_feedback_error` and `wasserstein1_empirical`; fit `loglog_slope`

### Key Numerical Results (Synthetic Data, seed=42)
- Exp5 baseline: threshold_fraction=2/6, terminal_distress≈0.333
- Exp5 full-book: terminal_distress≈0 (reduced x0 due to larger denominator)
- Exp6 baseline: threshold_fraction=1/116, terminal_distress≈0.0086
- Exp6 resampling log-log slopes: ≈ −0.47 (feedback error), −0.45 (W1) — within expected −0.43 to −0.49
- Note: spec's "expected" values (0.1694, 0.0246, etc.) are calibrated to real EBA data;
  synthetic data produces fewer distressed banks due to diversification structure

---

## Exp 3 — Graphon smoothing and kernel truncation (run_exp3.py)

**Imports:** `graphon`, `model` (loss functions), `metrics`

**Loss function closure pattern (eps must be captured):**
```python
from model import loss_hard, loss_positive_ramp
loss_fn = lambda x: loss_positive_ramp(x, eps)   # or functools.partial
times, states = simulate_graphon(x0, W, mu, r, T, dt, loss_fn)
```

**Component 1 — block-exponential kernel, hard vs positive-ramp:**
- B=[[0.38,0.24],[0.31,0.36]], mu=-0.06, T=2, n=2000, dt=2e-3
- Hard terminal fraction ≈ 0.2195; frac_err halves exactly with eps
- State error (sup_vector_error_inf): ≈1.31e-2/6.54e-3/3.27e-3 for eps=0.04/0.02/0.01
  (larger than task's nominal 9.40e-3/4.70e-3/2.35e-3 by ≈×1.4; halving ratio ~2.0 correct)

**Component 2 — directed 30-mode truncation:**
- Smooth x0≈0.13 + mu=-0.0006 → no cascades ever occur → all K errors = 0 (trivially monotone)
- This is correct behavior: kernel is irrelevant when H(t)≡0

**Component 3 — trigonometric family:**
- W = B[ij] + kernel_trigonometric(M, kappa=5.0); benchmark at M=16
- kappa=5.0 in trig kernel ↔ kappa=10.0 in block_exponential (factor-of-2 Fourier convention)
- State errors M=1→1.49e-2, M=2→5.81e-3, M=4→4.46e-4, M=8→1.74e-6

**Step-halving:** dt=0.002 vs dt=0.001 → discrepancy ≈ 4.7e-5

**Plot guard:** semilogy with all-zero errors (C2 case) triggers UserWarning; use linear scale.

---

## Exp 4 — Osgood boundary and hard-solution selection (run_exp4.py)

**Imports:** `osgood`, `metrics`

**Solver:** DOP853, rtol=2e-12, atol=2e-14, max_step=0.002, n_output=2001, T=1.0
**eps grid:** 2^{-4}, …, 2^{-12}

**Convergence slopes (log-log, fitted over 9 eps values):**
| F | averaged_drift | benchmark | fitted slope |
|---|----------------|-----------|-------------|
| F_linear | averaged_drift_linear | β^+(t)=0 | 1.0000 |
| F_log_osgood | averaged_drift_log | β^+(t)=0 | 0.3572 |
| F_sqrt (pos) | averaged_drift_sqrt | β^+(t)=t²/4 | 0.4992 |
| F_sqrt (neg) | averaged_drift_sqrt_negative | β^-(t)=0 | exact (errors=0) |

**averaged_drift_sqrt_negative at β=0:** drift=0 → β stays at 0 exactly for all eps.
Guard `loglog_slope` with a zero-check to avoid log(0) → NaN.

**_safe_loglog_slope pattern:**
```python
def _safe_loglog_slope(eps_vals, errors):
    if np.all(errors == 0.0) or np.any(~np.isfinite(errors)):
        return float("nan"), float("nan")
    return loglog_slope(eps_vals, errors)
```
