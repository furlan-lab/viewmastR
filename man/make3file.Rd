% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/extras.R
\name{make3file}
\alias{make3file}
\title{Export a Seurat Object to 10X-Style Files}
\usage{
make3file(seu, assay = "RNA", dir, get_reductions = TRUE)
}
\arguments{
\item{seu}{A Seurat object.}

\item{assay}{Which assay to export (default \code{"RNA"}).}

\item{dir}{Output directory (must already exist).}

\item{get_reductions}{Logical; also export reduction embeddings
(default \code{TRUE}).}
}
\value{
Invisibly returns \code{NULL}; called for its side effects.
}
\description{
Write the counts matrix, features, barcodes, metadata, variable
features and—optionally—reduction embeddings from a Seurat object in
the 10X “3-file” layout.
}
\details{
The function creates a sub-directory called \code{3file} inside
\code{dir} and writes:

\itemize{
\item{\file{matrix.mtx.gz}}{Compressed Matrix Market file containing the
counts matrix.}
\item{\file{features.tsv.gz}}{Gene (feature) table.}
\item{\file{barcodes.tsv.gz}}{Cell barcodes.}
\item{\file{meta.csv}}{Cell-level metadata.}
\item{\file{\<reduction\>_reduction.tsv.gz}}{Embeddings for each
reduction (UMAP, PCA, …); written only when
\code{get_reductions = TRUE}.}
\item{\file{variablefeatures.tsv.gz}}{Variable-gene list.}
}
}
\examples{
\dontrun{
make3file(seu, assay = "RNA", dir = "out", get_reductions = FALSE)
}
}
