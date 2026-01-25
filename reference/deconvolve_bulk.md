# Deconvolve bulk RNA-seq data using single-cell signatures

Deconvolve bulk RNA-seq data using single-cell signatures

## Usage

``` r
deconvolve_bulk(
  signatures,
  bulk_counts,
  gene_lengths,
  gene_weights = NULL,
  backend = "wgpu",
  insert_size = 500,
  init_log_exp = -10,
  learning_rate = 0.01,
  l1_lambda = 0,
  l2_lambda = 0,
  max_iter = 10000,
  poll_interval = 100,
  ll_tolerance = 1e-06,
  sparsity_tolerance = 1e-04,
  verbose = TRUE,
  method = c("gd", "em")
)
```

## Arguments

- signatures:

  Matrix of cellular signatures (genes × cell_types). Each column should
  be normalized to sum to 1.

- bulk_counts:

  Matrix of bulk RNA-seq counts (genes × samples). Gene names must match
  signatures.

- gene_lengths:

  Numeric vector of gene lengths, same length as nrow(signatures).

- gene_weights:

  Optional numeric vector of gene weights (0-1), same length as
  nrow(signatures). Default is all 1s.

- backend:

  Which compute backend to use: "ndarray" (CPU, default), "wgpu" (GPU if
  compiled with wgpu feature), "cuda" (CUDA if available).

- insert_size:

  Insert size for RNA-seq fragments. Default 500.

- init_log_exp:

  Initial value for log exposures. Default -10.

- learning_rate:

  Learning rate for optimization. Default 0.01.

- l1_lambda:

  L1 regularization parameter. Default 0.0 (no regularization).

- l2_lambda:

  L2 regularization parameter. Default 0.0 (no regularization).

- max_iter:

  Maximum number of iterations. Default 10000.

- poll_interval:

  Check convergence every N iterations. Default 100.

- ll_tolerance:

  Log-likelihood convergence tolerance. Default 1e-6.

- sparsity_tolerance:

  Sparsity convergence tolerance. Default 1e-4.

- verbose:

  emit learning progress stats

- method:

  either default - gd (Poisson regression with gradient descent) or em
  (Expectation-Maximization algorithm)

## Value

A list with:

- exposures:

  Matrix of cell type proportions (cell_types+intercept × samples)

- pred_counts:

  Matrix of predicted bulk counts (genes × samples)

- proportions:

  Normalized cell type proportions excluding intercept

## Examples

``` r
if (FALSE) { # \dontrun{
# CPU backend (default)
result <- deconvolve_bulk(
  signatures = signatures,
  bulk_counts = bulk_counts,
  gene_lengths = gene_lengths
)

# GPU backend (if available)
result <- deconvolve_bulk(
  signatures = signatures,
  bulk_counts = bulk_counts,
  gene_lengths = gene_lengths,
  backend = "wgpu"
)
} # }
```
