# Mathematical Summary of the Rust Deconvolution Approach

## Problem Setup

**Goal:** Decompose bulk RNA-seq data into cell type proportions using single-cell reference signatures.

**Given:**
- `S` = Signature matrix (genes × cell_types), where each column sums to 1 (normalized)
- `k` = Observed bulk counts (genes × samples)
- `C` = Gene length matrix (genes × samples), scaled by insert size
- `w` = Gene weights (genes), values in [0, 1]

**Find:**
- `E` = Exposure matrix (cell_types × samples) = estimated cell counts per type per sample

---

## Forward Model

### 1. **Molecular to Fragment Conversion**

```
Predicted counts: ŷ = (S' × E) ⊙ C
```

Where:
- `S'` = Augmented signatures `[S | intercept_column]` where intercept = uniform 1/n_genes
- `E` = Exposures including intercept `[e₁, e₂, ..., eₖ, intercept]ᵀ`
- `⊙` = Element-wise multiplication
- `C[g,s]` = gene_length[g] / insert_size (conversion from molecules to sequencing fragments)

**Step-by-step:**
1. Compute molecules per gene: `q = S' × E` (matrix multiplication)
2. Convert to fragments: `ŷ = q ⊙ C`

### 2. **Parameterization**

To ensure non-negativity, we use log-space:
```
E = exp(log_E)
```

Where `log_E` are the learned parameters (can be any real number).

**Initialization:**
```
log_E[i,j] ~ Normal(init_log_exposure, 0.1)
```
Random initialization breaks symmetry so different cell types can learn different values.

---

## Loss Function

### 1. **Poisson Negative Log-Likelihood**

For each gene `g` and sample `s`:

```
NLL(g,s) = log(k[g,s]!) - k[g,s]·log(ŷ[g,s]) + ŷ[g,s]
```

Using Stirling's approximation for `log(k!)`:
```
log(k!) ≈ k·log(k) - k  (for k > 0)
log(0!) = 0
```

**Weighted NLL:**
```
NLL_weighted = Σ w[g] · NLL(g,s)
```

Sum over all genes and samples for total loss.

### 2. **Regularization**

**L1 Penalty (promotes sparsity):**
```
Penalty_L1 = λ₁ · Σᵢⱼ scale[j] · E[i,j]
```

Where `scale[j]` = per-sample null likelihood (from null model), normalized so λ₁=1 means penalty ≈ null deviance when exposures sum to total molecules with uniform distribution.

**L2 Penalty (prevents large values, excludes intercept):**
```
Penalty_L2 = λ₂ · Σᵢⱼ scale[j] · E[i,j]²
```

Only applied to non-intercept exposures.

**Total Loss:**
```
L = NLL_weighted + Penalty_L1 + Penalty_L2
```

---

## Two-Phase Optimization

### Phase 1: Null Model

**Purpose:** Establish baseline and compute per-sample null likelihood for regularization scaling.

**Procedure:**
1. Freeze all cell type exposures at exp(-100) ≈ 0
2. Optimize only the intercept term using Adam optimizer
3. Compute null NLL per sample for regularization scaling

**Result:** 
- Fitted intercept values
- `null_NLL[s]` = baseline negative log-likelihood per sample

### Phase 2: Full Model

**Purpose:** Learn cell type exposures.

**Procedure:**
1. Re-initialize cell type log-exposures: `log_E ~ Normal(init_log_exposure, 0.1)`
2. Keep intercept from null model
3. Optimize all parameters (exposures + intercept) using Adam

**Optimizer:** Adam with learning rate `lr` (default 0.01)

**Update rule:**
```
θₜ₊₁ = θₜ - lr · ADAM_gradient(∇L)
```

---

## Convergence Criteria

Stop when **both** conditions met:
1. **Likelihood:** `ΔL / L < tolerance_ll` (default: 0.01%)
2. **Sparsity:** `Δ(nonzero) / nonzero < tolerance_sparsity` (default: 1%)

**Additional constraints:**
- Don't converge before iteration 5000
- Maximum iterations: 100,000 (default)

---

## Output Transformations

### 1. **Raw Exposures**
```
E[celltype, sample] = exp(log_E[celltype, sample])
```
Units: Cell-equivalents (estimated number of cells)

### 2. **Proportions (excluding intercept)**
```
P[celltype, sample] = E[celltype, sample] / Σ_celltypes E[celltype, sample]
```
Units: Dimensionless, sums to 1 per sample

### 3. **Proportions (including intercept)**
```
P_full[component, sample] = E[component, sample] / Σ_all E[component, sample]
```
Includes intercept to show what fraction of signal is unexplained.

### 4. **Intercept Fraction** (quality metric)
```
intercept_frac[sample] = E[intercept, sample] / Σ_all E[component, sample]
```
High values (>0.3) suggest poor fit.

---

## Key Mathematical Properties

### 1. **Non-negativity**
Guaranteed by exponential transform: `E = exp(log_E) ≥ 0`

### 2. **Identifiability**
- Signatures must be normalized (Σ S[g,c] = 1 for each cell type c)
- This sets the scale: exposures represent cell-equivalents
- Without normalization, exposures and signatures would be confounded

### 3. **Numerical Stability**
- Add `ε = 1e-10` to avoid `log(0)` in Poisson NLL
- Random initialization breaks symmetry
- Log-space parameterization prevents negative values

### 4. **Regularization Scaling**
Penalties are scaled by null likelihood so:
- `λ = 1` means penalty ≈ null deviance improvement
- Relative to baseline model (intercept only)
- Makes regularization strength interpretable across datasets

---

## Comparison to Python Implementation

**Similarities:**
- Same Poisson likelihood
- Same two-phase optimization (null → full)
- Same regularization approach
- Same gene length correction

**Differences:**
- **Initialization:** Rust uses `Normal(init_log_exp, 0.1)` for symmetry breaking
- **Convergence:** Rust requires minimum 5000 iterations, looser tolerance
- **Backend flexibility:** Rust can switch between CPU (NdArray) and GPU (WGPU) at runtime

---

## Computational Complexity

**Per iteration:**
- Forward pass: `O(G × C × S)` where G=genes, C=cell types, S=samples
- Backward pass: Same (automatic differentiation)
- Memory: `O(G × S + G × C + C × S)`

**Typical runtime:**
- 18,879 genes × 21 cell types × 81 samples
- ~5,000-20,000 iterations to convergence
- CPU: ~1-5 minutes
- GPU (WGPU): ~10-60 seconds