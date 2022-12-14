% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/reduce_dimensions.R
\name{cluster_LSI}
\alias{cluster_LSI}
\title{Cluster LSI}
\usage{
cluster_LSI(
  cds,
  k = 20,
  weight = F,
  num_iter = 1,
  resolution_parameter = NULL,
  random_seed = 2020,
  verbose = T,
  partition_q_value = 0.05
)
}
\arguments{
\item{cds}{the cell_data_set upon which to perform this operation.}

\item{k}{Nnteger number of nearest neighbors to use when creating the k nearest neighbor graph for Leiden clustering. k is 
related to the resolution of the clustering result, a bigger k will result in lower resolution and vice versa. Default is 20.}

\item{weight}{A logical argument to determine whether or not to use Jaccard coefficients for two nearest neighbors (based on the 
overlapping of their kNN) as the weight used for Louvain clustering. Default is FALSE}

\item{num_iter}{Integer number of iterations used for Leiden clustering. The clustering result giving the largest modularity 
score will be used as the final clustering result. Default is 1. Note that if num_iter is greater than 1, the random_seed argument will be ignored for the louvain method.}

\item{random_seed}{The seed used by the random number generator in louvain-igraph package. This argument will be ignored if num_iter is larger than 1.}

\item{verbose}{A logic flag to determine whether or not we should print the run details.}

\item{binarize}{boolean whether to binarize data prior to TFIDF transformation}

\item{resolution}{Parameter that controls the resolution of clustering. If NULL (Default), the parameter is determined automatically.}

\item{partition_qval}{Numeric, the q-value cutoff to determine when to partition. Default is 0.05.}
}
\description{
This function extracts clustering from the last iteration of LSI (see \code{iterativeLSI})
cell type differences in a single cell experiment.  This function uses the leiden clustering as implemented in monocle3, then finds
less granular clusters in the data using partitions (monocle3) using the reduced dimension LSI input from the last iteration of LSI used.
}
\references{
Granja, J. M.et al. (2019). Single-cell multiomic analysis identifies regulatory programs in mixed-phenotype 
acute leukemia. Nature Biotechnology, 37(12), 1458???1465.

Cusanovich, D. A., Reddington, J. P., Garfield, D. A., Daza, R. M., Aghamirzaie, D., Marco-Ferreres, R., et al. (2018). The 
  cis-regulatory dynamics of embryonic development at single-cell resolution. Nature, 555(7697), 538???542.

Vincent D. Blondel, Jean-Loup Guillaume, Renaud Lambiotte, Etienne Lefebvre: Fast unfolding of communities in large 
networks. J. Stat. Mech. (2008) P10008

V. A. Traag and L. Waltman and N. J. van Eck: From Louvain to Leiden: guaranteeing well-connected communities. 
Scientific Reports, 9(1) (2019). doi: 10.1038/s41598-019-41695-z.

Jacob H. Levine and et. al. Data-Driven Phenotypic Dissection of AML Reveals Progenitor-like Cells that 
Correlate with Prognosis. Cell, 2015.
}
