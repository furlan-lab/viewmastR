#' @title Create a Seurat Object from Bulk Data by Simulating Single-Cell Profiles
#' @description This function creates a Seurat object by simulating single-cell profiles from a bulk RNA-seq dataset (a `SummarizedExperiment` object).
#' It generates multiple single-cell profiles for each sample in the bulk dataset, matching the size distribution of counts from a provided single-cell Seurat object.
#' @param query A single-cell Seurat object with a size distribution of counts to be mimicked.
#' @param ref The reference bulk dataset, a `SummarizedExperiment` object.
#' @param N The number of simulated single cells per bulk sample. Default is 2.
#' @param assay The assay slot of the query Seurat object. Default is "RNA".
#' @param bulk_feature_row The column name in `rowData(ref)` that contains gene symbols. Default is "gene_short_name".
#' @param bulk_assay_name The name of the assay in the bulk dataset to use. Default is "counts".
#' @param dist The method to use for simulating counts. Options are "sc-direct", "sc-mimic", or "bulk". Default is "sc-direct".
#' @return A Seurat object with simulated single-cell profiles labeled according to the bulk reference.
#' @importFrom pbmcapply pbmclapply
#' @importFrom parallel detectCores
#' @importFrom Seurat CreateSeuratObject GetAssayData
#' @importFrom SummarizedExperiment assays rowData colData
#' @importFrom Matrix Matrix
#' @export
splat_bulk_reference <- function(
    query,
    ref,
    N = 2,
    assay = "RNA",
    bulk_feature_row = "gene_short_name",
    bulk_assay_name = "counts",
    dist = c("sc-direct", "sc-mimic", "bulk")
) {
  dist <- match.arg(dist)
  
  # Check input types and requirements
  if (!inherits(ref, "SummarizedExperiment")) {
    stop("The 'ref' parameter must be a SummarizedExperiment object.")
  }
  if (dist %in% c("sc-mimic", "sc-direct")) {
    if (is.null(query)) {
      stop("A 'query' Seurat object must be provided when 'dist' is 'sc-mimic' or 'sc-direct'.")
    }
    if (!inherits(query, "Seurat")) {
      stop("The 'query' parameter must be a Seurat object.")
    }
    message("Finding count distribution of query")
    sizes <- colSums(get_counts_seurat(query, assay_name = assay))
    den <- density(sizes)
    replace_counts <- FALSE
  } else {
    # For 'dist' == "bulk"
    message("Finding count distribution of bulk reference")
    sizes <- colSums(get_counts_se(ref, bulk_assay_name))
    den <- density(sizes)
    replace_counts <- TRUE
  }
  
  message("Finding common features between ref and query")
  universe <- intersect(
    rowData(ref)[[bulk_feature_row]],
    rownames(query)
  )
  if (length(universe) == 0) {
    stop("No common features found between 'ref' and 'query'.")
  }
  
  message(paste0("Simulating ", N, " single cells for every bulk dataset sample"))
  counts <- get_counts_se(ref, bulk_assay_name)[
    match(universe, rowData(ref)[[bulk_feature_row]]),
    , drop = FALSE
  ]
  rownames(counts) <- universe
  num_samples <- ncol(ref)
  
  # Simulate single-cell profiles
  newdata <- pbmcapply::pbmclapply(
    seq_len(num_samples),
    function(n) {
      newsizes <- sample(sizes, N, replace = TRUE) + rnorm(N, 0, den$bw)
      trimmed_newsizes <- round(newsizes[newsizes > min(sizes) & newsizes < max(sizes)])
      if (length(trimmed_newsizes) < N) {
        trimmed_newsizes <- rep(round(mean(sizes)), N)
      }
      final_newsizes <- sample(trimmed_newsizes, N, replace = TRUE)
      rsums <- counts[, n]
      names(rsums) <- universe
      splat <- rep(names(rsums), times = rsums)
      dl <- lapply(final_newsizes, function(i) {
        if (length(splat) == 0) {
          all_counts <- setNames(rep(0, length(universe)), universe)
        } else {
          sampled_genes <- sample(splat, i, replace = replace_counts)
          tab <- table(sampled_genes)
          all_counts <- setNames(rep(0, length(universe)), universe)
          all_counts[names(tab)] <- as.numeric(tab)
        }
        return(all_counts)
      })
      mat <- do.call(cbind, dl)
      Matrix::Matrix(mat, sparse = TRUE)
    },
    mc.cores = parallel::detectCores()
  )
  
  # Combine data into a single matrix
  newdata <- newdata[sapply(newdata, ncol) > 0]
  if (length(newdata) == 0) {
    stop("No data generated.")
  }
  tmpdata <- do.call(cbind, newdata)
  
  # Create metadata
  metai <- rep(seq_len(num_samples), each = N)
  meta <- as.data.frame(colData(ref))
  newmeta <- meta[rep(seq_len(nrow(meta)), each = N), , drop = FALSE]
  rownames(newmeta) <- paste0("Cell", seq_len(ncol(tmpdata)))
  colnames(tmpdata) <- rownames(newmeta)
  
  # Create Seurat object
  seurat_obj <- CreateSeuratObject(tmpdata, meta.data = newmeta)
  return(seurat_obj)
}

#' @title Get Counts from Seurat Object
#' @description Retrieves the count matrix from a Seurat object.
#' @param cds A Seurat object.
#' @param assay_name The name of the assay to extract counts from. Default is "RNA".
#' @return A matrix of counts.
get_counts_seurat <- function(cds, assay_name = "RNA") {
  GetAssayData(object = cds, assay = assay_name, slot = "counts")
}

#' @title Get Counts from SummarizedExperiment Object
#' @description Retrieves the count matrix from a SummarizedExperiment object.
#' @param obj A SummarizedExperiment object.
#' @param bulk_assay_name The name of the assay to extract counts from. Default is "counts".
#' @return A matrix of counts.
get_counts_se <- function(obj, bulk_assay_name = "counts") {
  counts <- assays(obj)[[bulk_assay_name]]
  if (is.null(counts)) {
    if (length(assays(obj)) > 1) {
      stop("More than one assay object found in 'ref'; please specify 'bulk_assay_name'.")
    }
    if (length(assays(obj)) == 0) {
      stop("No assay object found in 'ref'.")
    }
    counts <- assays(obj)[[1]]
  }
  counts
}
