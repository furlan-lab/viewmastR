# Select features in a cell_data_set for dimensionality reduction

Monocle3 aims to learn how cells transition through a biological program
of gene expression changes in an experiment. Each cell can be viewed as
a point in a high-dimensional space, where each dimension describes the
expression of a different gene. Identifying the program of gene
expression changes is equivalent to learning a *trajectory* that the
cells follow through this space. However, the more dimensions there are
in the analysis, the harder the trajectory is to learn. Fortunately,
many genes typically co-vary with one another, and so the dimensionality
of the data can be reduced with a wide variety of different algorithms.
Monocle3 provides two different algorithms for dimensionality reduction
via `reduce_dimensions` (UMAP and tSNE). The function `select_features`
is an optional step in the trajectory building process before
`preprocess_cds`. After calculating dispersion for a cell_data_set using
the `calculate_feature_dispersion` function, the `select_features`
function allows the user to identify a set of genes that will be used in
downstream dimensionality reduction methods.

## Usage

``` r
select_features(
  cds,
  fit_min = 1,
  fit_max = Inf,
  logmean_ul = Inf,
  logmean_ll = -Inf,
  top_n = NULL
)
```

## Arguments

- cds:

  the cell_data_set upon which to perform this operation.

- fit_min:

  the minimum multiple of the dispersion fit calculation; default = 1

- fit_max:

  the maximum multiple of the dispersion fit calculation; default = Inf

- logmean_ul:

  the maximum multiple of the dispersion fit calculation; default = Inf

- logmean_ll:

  the maximum multiple of the dispersion fit calculation; default = Inf

- top:

  top_n if specified, will override the fit_min and fit_max to select
  the top n most variant features. logmena_ul and logmean_ll can still
  be used.

## Value

an updated cell_data_set object with selected features
