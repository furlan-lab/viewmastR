# ViewmastR inference

This function performs cell type inference using a trained model,
passing sparse matrices directly to Rust for maximum performance.

## Usage

``` r
viewmastR_infer(
  query_cds,
  model_dir,
  selected_features,
  query_celldata_col = "viewmastR_inferred",
  labels = NULL,
  verbose = TRUE,
  return_probs = FALSE,
  return_type = c("object", "list"),
  batch_size = NULL,
  threads = 1
)
```

## Arguments

- query_cds:

  Seurat or cell_data_set object

- model_dir:

  Path to the trained model directory

- selected_features:

  Character vector of feature names

- query_celldata_col:

  Name of column to store results (default: "viewmastR_inferred")

- labels:

  Optional character vector of class labels

- verbose:

  Print progress messages

- return_probs:

  If TRUE, add probability columns to metadata

- return_type:

  "object" returns the modified object, "list" returns object and probs

- batch_size:

  Cells per inference batch (default: auto)

- threads:

  Number of parallel threads (default: 1)

## Value

Modified query_cds with inferred labels
