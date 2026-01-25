# Set up training, testing, and optional query datasets for model training

This function prepares normalized and optionally scaled data matrices
from reference and (optionally) query datasets. If a `query_cds` is
provided, it identifies common features between the reference and query
datasets, normalizes them, optionally performs TF-IDF or scaling, and
returns both training/test splits from the reference and a query
dataset. If no `query_cds` is provided, the function behaves like a
reference-only setup, returning just training and testing splits.

## Usage

``` r
setup_training(
  query_cds = NULL,
  ref_cds,
  ref_celldata_col,
  norm_method = c("log", "binary", "size_only", "none"),
  selected_features = NULL,
  train_frac = 0.8,
  tf_idf = FALSE,
  scale = FALSE,
  LSImethod = 1,
  verbose = TRUE,
  addbias = FALSE,
  return_type = c("sparse_mat", "list", "matrix", "S4obj"),
  debug = FALSE
)
```

## Arguments

- query_cds:

  A
  [`Seurat`](https://satijalab.org/seurat/reference/Seurat-package.html)
  or `cell_data_set` object representing the query dataset. If `NULL`,
  no query data is processed.

- ref_cds:

  A
  [`Seurat`](https://satijalab.org/seurat/reference/Seurat-package.html)
  or `cell_data_set` object representing the reference dataset.

- ref_celldata_col:

  A character string indicating the metadata column in `ref_cds` to use
  as labels.

- norm_method:

  A character string specifying the normalization method to use. One of
  `"log"`, `"binary"`, `"size_only"`, or `"none"`.

- selected_features:

  A character vector of gene names to subset. If `NULL`, uses all common
  features (if query is provided) or all selected features (if only
  reference is provided).

- train_frac:

  A numeric value between 0 and 1 indicating the fraction of reference
  cells to use for training. The rest are used for testing.

- tf_idf:

  A logical indicating whether to perform TF-IDF normalization on the
  count matrices.

- scale:

  A logical indicating whether to scale the data. If `TRUE` and
  `tf_idf = TRUE`, TF-IDF takes precedence and scaling is ignored.

- LSImethod:

  An integer specifying the TF-IDF method variant to use (passed to
  [`tf_idf_transform`](https://furlan-lab.github.io/viewmastR/reference/tf_idf_transform.md)).

- verbose:

  A logical indicating whether to print progress messages.

- addbias:

  A logical indicating whether to add a bias row (ones) to the data
  matrices.

- return_type:

  A character string specifying the return format. One of
  `"sparse_mat"`, `"list"`, `"matrix"`, or `"S4obj"`.

- debug:

  A logical indicating whether to print debugging messages.

## Value

A list containing either matrices, lists of data items, or S4 objects,
depending on `return_type`.

## Details

This function handles two scenarios:

- If `query_cds` is provided, it extracts common features between
  `ref_cds` and `query_cds`, normalizes both datasets, performs optional
  TF-IDF or scaling, and returns training/testing splits from the
  reference along with a query dataset.

- If `query_cds` is `NULL`, it behaves like a reference-only setup,
  returning just training and testing splits from `ref_cds`.

The returned object depends on `return_type`:

- `"sparse_mat"`: A list containing sparse matrices (`dgCMatrix`) for
  train/test/query, labels (0-indexed integers), size factors, and
  metadata.

- `"matrix"`: A list of matrices containing `Xtrain_data`, `Xtest_data`,
  `Ytrain_label`, `Ytest_label`, optionally `query`, as well as
  `label_text` and `features`.

- `"list"`: A list of lists, where each cell is represented as a `list`
  with `data` and `target` elements.

- `"S4obj"`: A list of S4 objects `training_set`, `test_set`, and
  optionally `query_set`, each containing training items and metadata.

## Examples

``` r
if (FALSE) { # \dontrun{
# Example with both reference and query data:
result <- setup_training(
  query_cds = query_seurat_obj,
  ref_cds = ref_seurat_obj,
  ref_celldata_col = "cell_type",
  norm_method = "log",
  train_frac = 0.8,
  tf_idf = TRUE,
  scale = FALSE,
  return_type = "sparse_mat"
)

# Example with reference only:
result_ref <- setup_training(
  query_cds = NULL,
  ref_cds = ref_cds_obj,
  ref_celldata_col = "cell_type",
  norm_method = "none",
  train_frac = 0.7,
  scale = TRUE,
  return_type = "matrix"
)
} # }
```
