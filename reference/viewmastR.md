# Integrate and Train Models on Reference Dataset and (Optionally) Infer on Query Datasets

The `viewmastR` function preprocesses one or two single-cell datasets (a
reference and an optional query), splits the reference data into
training and test sets, and optionally includes the ability to run
inference on a query dataset for downstream analysis. It then applies
specified modeling functions (e.g., MLR, NN, NB) to train and optionally
predict on the query data.

## Usage

``` r
viewmastR(
  query_cds,
  ref_cds,
  ref_celldata_col,
  query_celldata_col = NULL,
  FUNC = c("mlr", "nn", "nb"),
  norm_method = c("log", "binary", "size_only", "none"),
  selected_features = NULL,
  train_frac = 0.8,
  tf_idf = FALSE,
  scale = FALSE,
  hidden_layers = c(as.integer(500), as.integer(100)),
  learning_rate = 0.001,
  max_epochs = 5,
  LSImethod = 1,
  verbose = TRUE,
  backend = c("auto", "wgpu", "nd", "candle"),
  threshold = NULL,
  keras_model = NULL,
  model_dir = "/tmp/sc_local",
  return_probs = FALSE,
  return_type = c("object", "list"),
  debug = FALSE,
  train_only = FALSE
)
```

## Arguments

- query_cds:

  A `Seurat` or `cell_data_set` object representing the query dataset.
  If `NULL`, the function will operate in "reference-only" mode, using
  the reference dataset for training and testing only.

- ref_cds:

  A `Seurat` or `cell_data_set` object representing the reference
  dataset. This is required.

- ref_celldata_col:

  A character string specifying the metadata column in `ref_cds` that
  contains the cell labels.

- query_celldata_col:

  A character string specifying a metadata column name in `query_cds`
  (or reference in reference-only mode) where predicted labels should be
  stored. If `NULL`, defaults to `"viewmastR_pred"`.

- FUNC:

  A character string specifying the modeling function to apply. One of
  `"mlr"`, `"nn"`, or `"nb"`.

- norm_method:

  Character string indicating the normalization method. One of `"log"`,
  `"binary"`, `"size_only"`, or `"none"`.

- selected_features:

  A character vector specifying genes to subset. If `NULL`, uses the set
  of common features (if query is provided) or selected genes directly
  (if reference-only).

- train_frac:

  A numeric value between 0 and 1 specifying the fraction of reference
  cells to use for training. The remainder are used for testing.

- tf_idf:

  Logical, whether to apply TF-IDF transformation after normalization.

- scale:

  Logical, whether to scale the data. If both `tf_idf` and `scale` are
  `TRUE`, TF-IDF takes precedence.

- hidden_layers:

  A numeric vector indicating the size of hidden layers (for the NN
  model). Only 1 or 2 layers are allowed.

- learning_rate:

  Numeric, learning rate for model training.

- max_epochs:

  Integer, the maximum number of epochs for model training.

- LSImethod:

  Integer, specifying the TF-IDF method variant if using TF-IDF.

- verbose:

  Logical, whether to print progress messages.

- backend:

  A character string specifying the backend to use. One of `"wgpu"`,
  `"nd"`, `"candle"`.

- threshold:

  Currently unused. Can be `NULL`.

- keras_model:

  Currently unused. Can be `NULL`.

- model_dir:

  A character string specifying the directory to store model artifacts.

- return_probs:

  Logical, whether to return predicted probabilities in the object's
  metadata.

- return_type:

  A character string specifying the return type. One of `"object"` or
  `"list"`. If `"object"`, returns the updated `query_cds`. If `"list"`,
  returns a list containing `object` and `training_output`.

- debug:

  Logical, whether to print debugging messages and dimension checks.

- train_only:

  Logical, if `TRUE`, only the reference data is processed and no query
  data is included.

## Value

Depending on `return_type`, returns either:

- `return_type = "object"`: the input `query_cds` (or `ref_cds` if
  `query_cds = NULL`) with predicted labels (and optionally
  probabilities) appended.

- `return_type = "list"`: a list containing:

  - object:

    The updated `query_cds` (or `ref_cds`).

  - training_output:

    The output from the model training process, including probabilities
    if applicable.

## Details

The function first calls
[`setup_training`](https://furlan-lab.github.io/viewmastR/reference/setup_training.md)
to preprocess and split the data into training, testing, and optionally
query subsets. Then, based on the selected `FUNC`, it calls one of the
model training and prediction functions (`process_learning_obj_mlr`,
`process_learning_obj_ann`, `process_learning_obj_nb`). If
`train_only = TRUE`, the query portion is skipped and no query
predictions are made.

For `"mlr"` and `"nn"` functions, predicted log odds are converted to
probabilities using the logistic function. Predicted cell labels are
assigned to the `query_cds` (or `ref_cds` if query is not provided).

## Examples

``` r
if (FALSE) { # \dontrun{
# Training and predicting with reference and query data:
res <- viewmastR(
  query_cds = query_seurat_obj,
  ref_cds = ref_seurat_obj,
  ref_celldata_col = "cell_type",
  FUNC = "mlr",
  norm_method = "log",
  train_frac = 0.8,
  backend = "wgpu",
  verbose = TRUE,
  return_type = "object"
)

# Reference-only scenario:
res_ref <- viewmastR(
  query_cds = NULL,
  ref_cds = ref_cds_obj,
  ref_celldata_col = "cell_type",
  FUNC = "nn",
  norm_method = "none",
  train_frac = 0.7,
  scale = TRUE,
  train_only = TRUE,
  return_type = "list"
)
} # }
```
