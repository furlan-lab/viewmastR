# Sparse matrix training entry-point for MLR and ANN/ANN-2L.

Instead of receiving lists of cells from R, receives sparse matrices
directly. Size factors should be pre-computed in R (or set to 1.0 if
data is already normalized).

## Usage

``` r
process_learning_obj_sparse(
  model_type,
  train_x,
  train_i,
  train_p,
  train_dims,
  train_labels,
  train_size_factors,
  test_x,
  test_i,
  test_p,
  test_dims,
  test_labels,
  test_size_factors,
  query_x,
  query_i,
  query_p,
  query_dims,
  query_size_factors,
  labels,
  feature_names,
  hidden_size,
  learning_rate,
  num_epochs,
  directory,
  verbose,
  backend
)
```
