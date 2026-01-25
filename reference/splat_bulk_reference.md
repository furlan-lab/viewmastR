# Bulk reference

This function creates a seurat object (typically single cell genomics)
of multiple single cell profiles from each sample from a bulk object
(SummarizedExperiment or Seurat object currently supported). In doing
so, the function creates single cell profiles with a size distribution
that approximates the provided single cell object (query)

## Usage

``` r
splat_bulk_reference(
  query = NULL,
  ref,
  N = 2,
  assay = "RNA",
  bulk_feature_row = "gene_short_name",
  bulk_assay_name = "RNA",
  dist = c("sc-direct", "sc-mimic", "bulk"),
  seed = 42
)
```

## Arguments

- query:

  a single cell object (Seurat) with a size distribution of counts to be
  mimicked in the assay argument

- ref:

  the reference object (SummarizedExperiment or Seurat)

- N:

  the number of simulated cells to create per bulk sample

- assay:

  the assay slot of the query (Seurat)

- bulk_feature_row:

  the column name of gene symbols in ref (only used if ref is
  SummarizedExperiment)

- bulk_assay_name:

  the name of the assay object in the ref

- dist:

  distribution method: "sc-direct", "sc-mimic", or "bulk"

- seed:

  random seed for reproducibility

## Value

a classified seurat object labeled according to the bulk reference
