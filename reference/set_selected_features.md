# Set Selected Features for Ordering

This function updates the internal metadata of a Monocle3
(`cell_data_set`) or Seurat object to mark specific genes as
`use_for_ordering`. This is useful for defining a custom set of features
for trajectory inference.

## Usage

``` r
set_selected_features(cds, genes, gene_column = "id", unique_column = "id")
```

## Arguments

- cds:

  An object of class `cell_data_set` (Monocle3) or `Seurat`.

- genes:

  Character vector. A list of gene identifiers to mark as selected.
  These must match the identifiers found in `gene_column`.

- gene_column:

  Character. The name of the column in the object's dispersion metadata
  to match against the provided `genes` vector. Default is "id".

- unique_column:

  Character. Used only for Monocle3 objects with a dispersion function
  present. Specifies the unique identifier column in the dispersion
  table to map back to if `gene_column` is being used for lookup (e.g.,
  mapping gene symbols back to Ensembl IDs). Default is "id".

## Value

The modified `cds` object with the updated `use_for_ordering` slot.

## Details

Manually defines the set of features (genes) to be used for trajectory
ordering or downstream analysis.

## Examples

``` r
if (FALSE) { # \dontrun{
  # Define a list of interesting genes
  my_genes <- c("GeneA", "GeneB", "GeneC")

  # Update the object to use these genes for ordering
  cds <- set_selected_features(cds, genes = my_genes)
} # }
```
