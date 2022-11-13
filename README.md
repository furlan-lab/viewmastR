# ViewmastR

<p align="center"><img src="blob/viewmastR2.png" alt="" width="100"></a></p>
<hr>


ViewmastR is a method for performing unsupervised classification of single cells across datasets written for use in the R environment.

The inputs are cell_data_set objects (see http://github/trapnell-lab/monocle3)

## Installation 

1) Download ArrayFire: https://arrayfire.com/binaries/

2) Install in R
```

if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")

if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")


devtools::install_github("daqana/rcpparrayfire")
devtools::install_github("furlan-lab/viewmastR")

```


## Example
```
query_cds <- viewmastR(query_cds, 
                      ref_cds, 
                      ref_celldata_col="celltype", 
                      query_celldata_col=NULL, 
                      FUNC=c("naive_bayes", "neural_network", "bagging","softmax_regression", "logistic_regression", "deep_belief_nn", "perceptron"),
                      selected_genes=NULL,
                      train_frac = 0.8,
                      verbose = T)

```

## Acknowledgements

Written by Scott Furlan.
