% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/inference.R
\name{viewmastR_infer}
\alias{viewmastR_infer}
\title{Function to infer cell labels using a trained model}
\usage{
viewmastR_infer(
  query_cds,
  model_path,
  vg,
  query_celldata_col = "viewmastR_inferred",
  labels = NULL,
  verbose = T,
  return_probs = F
)
}
\arguments{
\item{query_cds}{Seurat or cell_data_set object - The dataset for which cell labels are to be inferred.}

\item{model_path}{character path to the trained model file.}

\item{vg}{character vector - Features used for inference (must be the same used during model creation).}

\item{query_celldata_col}{character vector - names of the column to store inferred cell labels in the query dataset. Default is "viewmastR_inferred".}

\item{labels}{character vector - optional labels corresponding to the class indices. Default is NULL.}

\item{verbose}{bool - show messaging}

\item{return_probs}{logical If TRUE, returns the class probabilities. Default is FALSE.}
}
\value{
If return_probs is TRUE, returns a data frame containing class probabilities.
If return_probs is FALSE, updates the query_cds object with inferred cell labels and returns it.
}
\description{
This function infers cell labels using a trained model and updates the input dataset with the inferred labels.
}
