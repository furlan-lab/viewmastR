# A *single* entry-point that covers MLR and ANN/ANN-2L.

- `model_type` – `"mlr"`, `"ann"`, or `"ann2"` (you can choose any
  tokens you like)

- `hidden_size` – `NULL` for MLR; numeric (len 1 or 2) for ANN.

## Usage

``` r
process_learning_obj(
  model_type,
  train,
  test,
  query,
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
