# Confusion matrix

This function will generate a confusion matrix between two factors; pred
(short for prediction) and gt (short for ground truth). One may
optionally supply a named vector of colors to annotate the row and
column legends.

## Usage

``` r
confusion_matrix(pred, gt, cols = NULL)
```

## Arguments

- pred:

  factor of predictions

- gt:

  factor of ground truth

- cols:

  named vector of colors

## Value

a confusion matrix plot
