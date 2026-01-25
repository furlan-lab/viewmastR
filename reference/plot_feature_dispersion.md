# Plot Feature Dispersion

This function generates a scatter plot comparing feature expression
levels to their variability (dispersion). It supports objects from both
Monocle3 (`cell_data_set`) and Seurat. If feature selection has been
performed (i.e., `use_for_ordering` is present in the dispersion
metadata), selected features are highlighted in red ("firebrick1"),
while others are shown in gray.

## Usage

``` r
plot_feature_dispersion(cds, size = 1, alpha = 0.4)
```

## Arguments

- cds:

  An object of class `cell_data_set` (Monocle3) or `Seurat`. The object
  must have dispersion data pre-calculated and stored in:

  - `cds@int_metadata$dispersion` for `cell_data_set` objects.

  - `cds@misc$dispersion` for `Seurat` objects.

- size:

  Numeric. The size of the points in the scatter plot. Default is 1.

- alpha:

  Numeric. The transparency level of the points, ranging from 0
  (invisible) to 1 (solid). Default is 0.4.

## Value

A [`ggplot`](https://ggplot2.tidyverse.org/reference/ggplot.html) object
representing the dispersion plot. This allows further modification using
standard ggplot2 functions (e.g., adding titles or changing themes).

## Details

Visualizes the relationship between the log mean expression and log
dispersion of features (genes) in a single-cell dataset.

## Examples

``` r
if (FALSE) { # \dontrun{
  # For a Monocle3 object
  p <- plot_feature_dispersion(cds, size = 1.5, alpha = 0.5)
  p + ggplot2::ggtitle("Dispersion Plot")

  # For a Seurat object
  # Ensure dispersion is calculated and stored in @misc$dispersion first
  p_seurat <- plot_feature_dispersion(seurat_obj)
} # }
```
