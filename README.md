# Exact Low-Rank Synthetic Contagion Experiments

Implements and validates the exact rank-K closure for finite-network contagion
dynamics described in the paper on low-rank graphon contagion models. Experiments
cover synthetic networks, graphon kernels, Osgood boundary analysis, and empirical
EBA sovereign-holdings applications.

## Repository Structure

```
src/            Core implementation modules
exp/            Experiment runner scripts
tests/          pytest test suite (121 tests)
results/        Output artifacts (CSVs, PNGs, RESULTS.md)
data/           Data directory (EBA input files go here)
```

## Core Model

The finite-network contagion dynamics (equation 6) are:

```
X_t^{i,N} = x_i^N + μt + (rt/N)Σ_j(e_ij - e_ji) - (1/N)Σ_j e_ij ∫_0^t ℓ(X_s^j) ds
```

Under the factorised exposure structure `e_ij = (1/K) Σ_k a_{i,k} b_{j,k}`, the
full N-dimensional dynamics reduce exactly to K feedback coordinates.

## Source Modules

| Module | Description |
|--------|-------------|
| `src/model.py` | Loss functions, direct Euler simulator |
| `src/low_rank.py` | Factorised exposure, reduced feedback, population ODE |
| `src/sampling.py` | Quantile-matched and iid type sampling |
| `src/metrics.py` | Path errors, threshold fractions, Wasserstein-1 |
| `src/contagion_solver.py` | High-level direct/reduced runner and validators |
| `src/graphon.py` | Graphon kernel simulator and truncation utilities |
| `src/osgood.py` | Osgood regime ODE systems and ramp regularisations |
| `src/eba_loader.py` | EBA six-bank and 116-institution data loaders |
| `src/empirical_kernel.py` | Empirical factor loading construction |

## Experiments

| Script | Title |
|--------|-------|
| `exp/run_exp1.py` | Exact low-rank closure validation (direct vs reduced) |
| `exp/run_exp2.py` | Directedness and lender/borrower separation |
| `exp/run_exp3.py` | Graphon smoothing and kernel truncation |
| `exp/run_exp4.py` | Osgood boundary and hard-solution selection |
| `exp/run_exp5.py` | EBA six-bank sovereign-holdings (rank-4) |
| `exp/run_exp6.py` | EBA full-sample 116-institution (rank-38) |

## Installation

```bash
pip install numpy pandas scipy matplotlib seaborn statsmodels
```

All required packages are pre-installed in the experiment environment.

## Running Experiments

```bash
python exp/run_exp1.py   # ~5 s
python exp/run_exp2.py   # ~2 s
python exp/run_exp3.py   # ~60 s
python exp/run_exp4.py   # ~10 s
python exp/run_exp5.py   # ~5 s
python exp/run_exp6.py   # ~60 s
```

Results (CSVs and PNGs) are written to `results/`.

## Running Tests

```bash
pytest tests/ -v
```

121 tests covering all core modules.

## Key Validated Results

- **Direct/Reduced equivalence:** Machine-precision agreement (~10⁻¹⁵) under
  exact factorised exposures.
- **Finite-N convergence:** Threshold-fraction and feedback-path errors decrease
  as N grows from 400 to 1600.
- **Directedness:** r=0.05 creates a 10.5% lender/borrower gap vs 0% under r=0.
- **Graphon ramp convergence:** Terminal threshold fraction matches paper benchmark
  exactly; state errors halve when ε halves (first-order convergence).
- **Osgood boundary:** Fitted convergence slopes: linear=1.000, log-Osgood=0.357,
  sqrt(pos)=0.499, sqrt(neg)=0 (minimal solution selected exactly).
- **Resampling:** Feedback-path and Wasserstein-1 errors decay with slope −0.47/−0.45,
  consistent with N^{−1/2} theoretical bound.

## Results Summary

See [`results/RESULTS.md`](results/RESULTS.md) for full experiment output tables,
comparison with paper benchmarks, and notes on data limitations.
