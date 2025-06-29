% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/extendr-wrappers.R
\name{process_learning_obj}
\alias{process_learning_obj}
\title{A \emph{single} entry-point that covers MLR and ANN/ANN-2L.}
\usage{
process_learning_obj(
  model_type,
  train,
  test,
  query,
  labels,
  feature_names,
  hidden_size,
  learning_rate,
  num_epochs,
  directory,
  verbose,
  backend
)
}
\description{
\itemize{
\item \code{model_type} – \code{"mlr"}, \code{"ann"}, or \code{"ann2"} (you can choose any tokens you like)
\item \code{hidden_size} – \code{NULL} for MLR; numeric (len 1 or 2) for ANN.
}
}
\keyword{internal}
