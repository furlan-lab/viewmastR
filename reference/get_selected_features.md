# Get Selected Features for Ordering

This function extracts the features marked as `use_for_ordering` within
the object's internal dispersion metadata. It supports both Monocle3
(`cell_data_set`) and Seurat objects, provided the dispersion data is
stored in the expected slots.

## Usage

``` r
get_selected_features(cds, gene_column = "id")
```

## Arguments

- cds:

  An object of class `cell_data_set` (Monocle3) or `Seurat`.

- gene_column:

  Character. The name of the column in the dispersion table containing
  the gene identifiers you wish to retrieve. Default is "id".

## Value

A character vector containing the identifiers of the selected features.

## Details

Retrieves the list of features (genes) currently selected for trajectory
ordering or downstream analysis from a single-cell object.

## Examples

``` r
if (FALSE) { # \dontrun{
  # Retrieve IDs of ordering genes
  ordering_genes <- get_selected_features(cds)

  # Retrieve common names if stored in a "gene_short_name" column
  ordering_genes_names <- get_selected_features(cds, gene_column = "gene_short_name")
} # }
```
