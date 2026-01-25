# Augment data

This function takes a seurat object and finds cells that are not
sufficiently abundant when grouped by the column parameter, then
simulates data to augment cell number to a level of the parameter -
norm_number

## Usage

``` r
augment_data(obj, column, norm_number = 2000, assay = "RNA", prune = F)
```

## Arguments

- column:

  column from the metadata that designates cell group (i.e. celltype)

- norm_number:

  cell number to augment data to for cells that are not sufficiently
  abundant in the object

- prune:

  downsample cells present at number higher than norm_number to the
  level of norm_number (default = F)

## Value

a seurat object augmented with simulated cells such that all cell groups
are present at a level of norm_number of cells
