# SSC-TV ADMM variants

Total-variation Sparse Subspace Clustering, solved by ADMM. The solvers share
the same splitting and the same `X` / `C` updates; they differ only in the
proximal steps on the residual `E` and the TV auxiliaries `P`, `Q`.

`osc.py` is the Ordered Subspace Clustering (Tierney, Gao & Guo, CVPR 2014)
baseline used in `benchmark_sbm.py`.

## Shared formulation

```
min_{X, C, E, P, Q}   (λ_z/2) ||Y − YX − E||_F²  +  penalty(E)  +  γ ( pen(P) + pen(Q) )
s.t.  X = C_off,   DC = P,   C Dᵀ = Q,   diag(C) = 0
```

`D ∈ ℝ^{(N−1)×N}` is the first-order finite-difference operator, `C_off = C −
diag(C)`, `P = DC` are consecutive **row** differences of `C`, and `Q = CDᵀ`
are consecutive **column** differences. The TV terms push `C` toward a
piecewise-constant (block) structure.

| Variable | Role |
|---|---|
| `X` | reconstruction / self-expression (`Y ≈ YX + E`) |
| `C` | coefficient matrix (diag forced to 0) |
| `E` | residual / outliers |
| `P` | row-difference auxiliary (`DC = P`) |
| `Q` | column-difference auxiliary (`CDᵀ = Q`) |
| `F` | copy of `E` (hybrid `E` penalty only; `E = F`) |

Duals: `Λ` for `X = C_off` (penalty `μ`), `Π_P` / `Π_Q` for the TV constraints
(penalty `σ`), and `Π_F` for `E = F` (penalty `ρ`) when the hybrid residual is
used.

### ADMM updates (common skeleton)

1. **X** (normal equations):
   `(λ_z YᵀY + μ I) X = λ_z Yᵀ(Y − E) + μ C_off − Λ`
2. **C** (Sylvester, then `diag(C) = 0`). Let `K = DᵀD = V Λ_eig Vᵀ`. Then
   `μ C + σ (KC + CK) = RHS_C` with
   `RHS_C = μ (X + Λ/μ) + σ (Dᵀ P̃ + Q̃ D)`, `P̃ = P − Π_P/σ`, `Q̃ = Q − Π_Q/σ`,
   solved entrywise as `C'_{ij} = (Vᵀ RHS_C V)_{ij} / [μ + σ(λ_i + λ_j)]`.
3. **P / Q**: proximal map of `pen(P)` / `pen(Q)` (see table below).
4. **E** (and **F** if hybrid): proximal map of `penalty(E)` (see table).
5. Dual ascent on the equality constraints.

`S_τ` is entrywise soft-threshold; `GroupSoft_τ` is column-wise block
soft-threshold (prox of `τ ||·||_{2,1}`).

## Solver files

| File | Benchmark name | `E` | `P = DC` | `Q = CDᵀ` |
|---|---|---|---|---|
| `ssc_tv.py` | SSC-TV | `‖E‖₁` | `‖P‖₁` | `‖Q‖₁` |
| `ssc_tv_with_column_l21_on_columns_l1_on_rows.py` | SSC-TV-L21-P | `‖E‖₁` | `‖P‖_{2,1}` | `‖Q‖₁` |
| `ssc_tv_with_l21_on_rows_and_columns.py` | SSC-TV-L21-PQ | `‖E‖₁` | `‖P‖_{2,1}` | `‖Q‖_{2,1}` |
| `ssc_tv_e21_e1.py` | SSC-TV-E1E21 | `‖E‖₁ + ‖F‖_{2,1}` | `‖P‖₁` | `‖Q‖₁` |
| `ssc_tv_e21_e1_and_l21_on_columns_l1_on_rows.py` | SSC-TV-E1E21-L21-P | `‖E‖₁ + ‖F‖_{2,1}` | `‖P‖_{2,1}` | `‖Q‖₁` |
| `ssc_tv_e21_e1_and_l21_on_rows_and_columns.py` | SSC-TV-E1E21-L21-PQ | `‖E‖₁ + ‖F‖_{2,1}` | `‖P‖_{2,1}` | `‖Q‖_{2,1}` |

The three non-hybrid files also expose `ssc_admm_nuc_tv_e21`, which swaps
`‖E‖₁` for `‖E‖_{2,1}` only (no `F` split). That function is not in the
benchmark.

Entrywise `‖·‖₁` uses `S_{γ/σ}`; column-group `‖·‖_{2,1}` uses
`GroupSoft_{γ/σ}`.

## Residual penalties

**`‖E‖₁`** (SSC-style, Elhamifar & Vidal): scattered entry-level corruption.
`E = S_{λ_e/λ_z}(Y − YX)`.

**`‖E‖_{2,1}`** (LRR-style, Liu et al.): whole-column / sample-level outliers.
`E = GroupSoft_{λ_e/λ_z}(Y − YX)`.

**Hybrid `λ_e1 ‖E‖₁ + λ_e21 ‖F‖_{2,1}`** with `E = F`: sparse-group lasso.
Elementwise shrinkage soaks up scattered noise; column-group shrinkage flags
outlier nodes. Closed form:

```
R = Y − YX,     F̃ = F − Π_F/ρ
E = S_{λ_e1/(λ_z+ρ)} ( (λ_z R + ρ F̃) / (λ_z + ρ) )
F = GroupSoft_{λ_e21/ρ} ( E + Π_F/ρ )
Π_F += ρ (E − F)
```

At convergence `E ≈ F`, and `‖F_{:j}‖₂` is a per-column outlier score
(`outlier_scores` / `flag_outliers` in the hybrid modules).

OSC does not regularize `E` this way. Its `‖ZR‖_{1,2}` term is the analogue of
`‖P‖_{2,1}` / `‖Q‖_{2,1}` here: group-smoothness of consecutive coefficient
columns, not outlier detection.

## Running

```bash
python ssc_tv.py                          # synthetic sanity check
python benchmark_sbm.py                   # OSC vs the six SSC-TV variants
python benchmark_sbm.py --smoke
python tune_hyperparams.py
```
