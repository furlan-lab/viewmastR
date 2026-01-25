# Finds common features in a list of single cell objects

Machine learning algorithms often require features to be the same across
datasets. This function finds common features between a list of cell
data set objects (monocle3) and returns a list of cds's that have the
same features. Note that this function uses rownames of the 'fData'
DataFrame (monocle3) and the rownames of the seurat_object to find the
intersect of features common to all objects

## Usage

``` r
common_features(cds_list)
```

## Arguments

- cds_list:

  Input object.
