# Detects genes above minimum threshold.

For each gene in a cell_data_set object, detect_genes counts how many
cells are expressed above a minimum threshold. In addition, for each
cell, detect_genes counts the number of genes above this threshold that
are detectable. Results are added as columns num_cells_expressed and
num_genes_expressed in the rowData and colData tables respectively.

## Usage

``` r
detect_genes(cds, min_expr = 0, exprs_bin = TRUE, exprs_cuts = 25)
```

## Arguments

- cds:

  Input cell_data_set object.

- min_expr:

  Numeric indicating expression threshold

- exprs_bin:

  Boolean whether to bin genes by mean expression

- exprs_cuts:

  Numeic indicating number of bins if using exprs_bin

## Value

Updated cell_data_set object
