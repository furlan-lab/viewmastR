# Call the sparse training function in Rust

Internal wrapper to pass sparse matrices to Rust for MLR/ANN training.

## Usage

``` r
call_process_learning_obj_sparse(
  model_type,
  train_mat,
  train_labels,
  train_sf,
  test_mat,
  test_labels,
  test_sf,
  query_mat = NULL,
  query_sf = NULL,
  labels,
  feature_names,
  hidden_size = NULL,
  learning_rate = 0.001,
  num_epochs = 10,
  directory = "/tmp/sc_local",
  verbose = TRUE,
  backend = "nd"
)
```

## Arguments

- model_type:

  Character: "mlr", "ann", or "ann2"

- train_mat:

  dgCMatrix of training data (genes x cells)

- train_labels:

  Integer vector of 0-indexed labels

- train_sf:

  Numeric vector of size factors

- test_mat:

  dgCMatrix of test data

- test_labels:

  Integer vector of test labels

- test_sf:

  Numeric vector of test size factors

- query_mat:

  dgCMatrix of query data (or NULL)

- query_sf:

  Numeric vector of query size factors (or NULL)

- labels:

  Character vector of class names

- feature_names:

  Character vector of feature/gene names

- hidden_size:

  Integer vector of hidden layer sizes

- learning_rate:

  Numeric learning rate

- num_epochs:

  Integer number of epochs

- directory:

  Character path to model directory

- verbose:

  Logical for verbose output

- backend:

  Character: "nd", "wgpu", or "candle"

## Value

List with params, probs, history, duration
