# Export a Seurat Object to 10X-Style Files

Write the counts matrix, features, barcodes, metadata, variable features
and—optionally—reduction embeddings from a Seurat object in the 10X
“3-file” layout.

## Usage

``` r
make3file(seu, assay = "RNA", dir, get_reductions = TRUE)
```

## Arguments

- seu:

  A Seurat object.

- assay:

  Which assay to export (default `"RNA"`).

- dir:

  Output directory (must already exist).

- get_reductions:

  Logical; also export reduction embeddings (default `TRUE`).

## Value

Invisibly returns `NULL`; called for its side effects.

## Details

The function creates a sub-directory called `3file` inside `dir` and
writes:

- `matrix.mtx.gz`Compressed Matrix Market file containing the counts
  matrix.

- `features.tsv.gz`Gene (feature) table.

- `barcodes.tsv.gz`Cell barcodes.

- `meta.csv`Cell-level metadata.

- `\<reduction\>_reduction.tsv.gz`Embeddings for each reduction (UMAP,
  PCA, …); written only when `get_reductions = TRUE`.

- `variablefeatures.tsv.gz`Variable-gene list.

## Examples

``` r
if (FALSE) { # \dontrun{
make3file(seu, assay = "RNA", dir = "out", get_reductions = FALSE)
} # }
```
