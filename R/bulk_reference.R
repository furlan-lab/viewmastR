#' Bulk reference
#' @description This function creates a seurat object (typically single cell genomics) of multiple single cell profiles from each sample from a 
#' bulk object (SummarizedExperiment or Seurat object currently supported).  In doing so, the function creates single cell profiles with a size distribution
#' that approximates the provided single cell object (query)
#' @param query a single cell object (Seurat) with a size distribution of counts to be mimicked in the assay argument
#' @param ref the reference object (SummarizedExperiment or Seurat)
#' @param N the number of simulated cells to create per bulk sample
#' @param assay the assay slot of the query (Seurat)
#' @param bulk_feature_row the column name of gene symbols in ref (only used if ref is SummarizedExperiment)
#' @param bulk_assay_name the name of the assay object in the ref
#' @param dist distribution method: "sc-direct", "sc-mimic", or "bulk"
#' @param seed random seed for reproducibility
#' @return a classified seurat object labeled according to the bulk reference
#' @export
splat_bulk_reference <- function(query = NULL, 
                                 ref, 
                                 N = 2, 
                                 assay = "RNA", 
                                 bulk_feature_row = "gene_short_name", 
                                 bulk_assay_name = "RNA", 
                                 dist = c("sc-direct", "sc-mimic", "bulk"),
                                 seed = 42) {
  
  dist <- match.arg(dist)
  
  # Check ref type and prepare accordingly
  is_seurat_ref <- inherits(ref, "Seurat")
  is_se_ref <- inherits(ref, "SummarizedExperiment")
  
  if (!is_seurat_ref && !is_se_ref) {
    stop("ref must be either a Seurat or SummarizedExperiment object")
  }
  
  # Determine size distribution
  if (dist == "sc-mimic" | dist == "sc-direct") {
    message("Finding count distribution of query")
    sizes <- colSums(get_counts_seurat(query))
    den <- density(sizes)
    replace_counts <- FALSE
  } else {
    if (is_seurat_ref) {
      sizes <- colSums(get_counts_seurat(ref, assay = bulk_assay_name))
    } else {
      sizes <- colSums(assays(ref)[[bulk_assay_name]])
    }
    den <- density(sizes)
    replace_counts <- TRUE
  }
  
  # Find common features and get counts matrix
  message("Finding common features between ref and query")
  
  if (is_seurat_ref) {
    # Seurat reference
    universe <- intersect(rownames(ref), rownames(query))
    counts <- get_counts_seurat(ref, assay = bulk_assay_name)[universe, ]
    
    # Get metadata
    meta <- ref[[]]
  } else {
    # SummarizedExperiment reference
    universe <- intersect(rowData(ref)[[bulk_feature_row]], rownames(query))
    counts <- get_counts_se(ref, bulk_assay_name)[match(universe, rowData(ref)[[bulk_feature_row]]), ]
    rownames(counts) <- universe
    
    # Get metadata
    meta <- colData(ref)
  }
  
  message(paste0("Simulating ", N, " single cells for every bulk dataset case"))
  
  # Convert to dense matrix if needed (Rust expects matrix)
  if (inherits(counts, "sparseMatrix")) {
    counts <- as.matrix(counts)
  }
  
  # Call Rust function - now returns a single combined matrix
  rust_result <- splat_bulk_reference_rust(
    counts_matrix = counts,
    universe = universe,
    sizes = as.numeric(sizes),
    bandwidth = den$bw,
    n_cells_per_bulk = as.integer(N),
    replace_counts = replace_counts,
    seed = as.integer(seed),
    verbose = TRUE
  )
  
  # Convert single triplet matrix to dgCMatrix
  message("Converting to Seurat object...")
  tmpdata <- sparseMatrix(
    i = rust_result$i + 1L,  # R is 1-indexed
    j = rust_result$j + 1L,
    x = rust_result$x,
    dims = c(rust_result$nrow, rust_result$ncol),
    dimnames = list(rust_result$universe, NULL)
  )
  
  # Create metadata
  n_bulk <- rust_result$n_bulk
  n_cells <- rust_result$n_cells_per_bulk
  metai <- rep(seq_len(nrow(meta)), each = n_cells)
  newmeta <- meta[metai, , drop = FALSE]
  
  # Set dimnames
  rownames(newmeta) <- make.unique(paste0("bulk_", metai, "_cell_", rep(1:n_cells, n_bulk)))
  colnames(tmpdata) <- rownames(newmeta)
  
  # Create Seurat object
  CreateSeuratObject(tmpdata, meta.data = as.data.frame(newmeta))
}
# splat_bulk_reference <- function(query = NULL, 
#                                  ref, 
#                                  N = 2, 
#                                  assay = "RNA", 
#                                  bulk_feature_row = "gene_short_name", 
#                                  bulk_assay_name = "counts", 
#                                  dist = c("sc-direct", "sc-mimic", "bulk"),
#                                  seed = 42,
#                                  verbose = T) {
#   
#   dist <- match.arg(dist)
#   
#   # Determine size distribution
#   if (dist == "sc-mimic" | dist == "sc-direct") {
#     message("Finding count distribution of query")
#     sizes <- colSums(get_counts_seurat(query))
#     den <- density(sizes)
#     replace_counts <- FALSE
#   } else {
#     sizes <- colSums(assays(ref)[[bulk_assay_name]])
#     den <- density(sizes)
#     replace_counts <- TRUE
#   }
#   
#   # Find common features
#   message("Finding common features between ref and query")
#   universe <- intersect(rowData(ref)[[bulk_feature_row]], rownames(query))
#   
#   message(paste0("Simulating ", N, " single cells for every bulk dataset case"))
#   
#   # Get counts matrix with matched genes
#   counts <- get_counts_se(ref, bulk_assay_name)[match(universe, rowData(ref)[[bulk_feature_row]]), ]
#   rownames(counts) <- universe
#   
#   # Convert to dense matrix if needed (Rust expects matrix)
#   if (inherits(counts, "sparseMatrix")) {
#     counts <- as.matrix(counts)
#   }
#   
#   # Call Rust function
#   # message("Running Rust parallel computation...")
#   rust_results <- splat_bulk_reference_rust(
#     counts_matrix = counts,
#     universe = universe,
#     sizes = as.numeric(sizes),
#     bandwidth = den$bw,
#     n_cells_per_bulk = as.integer(N),
#     replace_counts = replace_counts,
#     seed = as.integer(seed),
#     verbose = verbose
#   )
#   
#   # Convert Rust triplet matrices to dgCMatrix
#   message("Converting to Seurat object...")
#   sparse_matrices <- pbmclapply(rust_results, function(triplet) {
#     sparseMatrix(
#       i = triplet$i + 1L,  # R is 1-indexed
#       j = triplet$j + 1L,
#       x = triplet$x,
#       dims = c(triplet$nrow, triplet$ncol),
#       dimnames = list(universe, NULL)
#     )
#   }, mc.cores = detectCores())
#   
#   # Combine matrices
#   tmpdata <- do.call(cbind, sparse_matrices)
#   
#   # Create metadata
#   meta <- colData(ref)
#   metai <- rep(seq_len(nrow(meta)), each = N)
#   newmeta <- meta[metai, ]
#   
#   # Set dimnames
#   rownames(newmeta) <- make.unique(paste0("bulk_", metai, "_cell_", rep(1:N, nrow(meta))))
#   colnames(tmpdata) <- rownames(newmeta)
#   
#   # Create Seurat object
#   CreateSeuratObject(tmpdata, meta.data = as.data.frame(newmeta))
# }


#' Bulk reference
#' @description This function creates a seurat object (typically single cell genomics) of multiple single cell profiles from each sample from a 
#' bulk object (SummarizedExperiment object currently supported).  In doing so, the function creates single cell profiles with a size distribution
#' that approximates the provided single cell object (query)
#' @param query a single cell object (Seurat) with a size distribution of counts to be mimicked in the assay argument
#' @param ref the reference object (Summarized Experiment)
#' @param assay the assay slot of the query (Seurat)
#' @param bulk_feature_row the column name of gene symbols in ref
#' @param bulk_assay_name the name of the assay object in the ref
#' @param dist IN DEVELOPMENT
#' @return a classified seurat object labeled according to the bulk reference
#' @importFrom pbmcapply pbmclapply
#' @importFrom parallel detectCores
#' @export
splat_bulk_reference2<-function(query=NULL, 
                               ref, N = 2, assay="RNA", 
                               bulk_feature_row = "gene_short_name", 
                               bulk_assay_name = "counts", 
                               dist=c("sc-direct", "sc-mimic", "bulk")){
  dist<-match.arg(dist)
  if(dist=="sc-mimic" | dist == "sc-direct"){
    message("Finding count distribution of query")
    sizes <- colSums(get_counts_seurat(query))
    den <- density(sizes)
    replace_counts<-F
  }else{
    sizes <- colSums(assays(ref)[[bulk_assay_name]])
    den <- density(sizes)
    replace_counts=T
  }
  message("Finding common features between ref and query")
  universe<-intersect(rowData(ref)[[bulk_feature_row]], rownames(query))
  message(paste0("Simulating ", N, " single cells for every bulk dataset case"))
  
  counts<-get_counts_se(ref, bulk_assay_name)[match(universe, rowData(ref)[[bulk_feature_row]]),]
  rownames(counts)<-universe
  newdata<-pbmcapply::pbmclapply(1:(dim(ref)[2]), function(n){
    newsizes <- sample(sizes, N, replace = TRUE) + rnorm(N * 10, 0, den$bw)
    trimmed_newdata <- round(newsizes[newsizes > min(sizes) &
                                        max(sizes)], 0)
    final_newsizes <- sample(trimmed_newdata, N)
    rsums <- counts[,n]
    total <- sum(rsums)
    if(is.null(names(rsums))){
      names(rsums)<-rownames(counts)
    }
    splat <- names(rsums)[rep(seq_along(rsums), rsums)]
    dl <- lapply(final_newsizes, function(i) {
      if(i>total){
        tab <- table(sample(splat, i, replace=T))
      } else {
        tab <- table(sample(splat, i, replace=replace_counts))
      }
      nf <- universe[!universe %in% names(tab)]
      all <- c(tab, setNames(rep(0, length(nf)), nf))
      all[match(universe, names(all))]
    })
    return(as.sparse(do.call(cbind, dl)))
  }, mc.cores = parallel::detectCores())

  output<-lapply(newdata, function(x) class(x)[1])
  good_ind<-lapply(output, function(x) x=="dgCMatrix")
  good_ind<-which(unlist(good_ind))

  metai<-lapply(good_ind, function(n) rep(n, N)) %>% do.call("c", .)
  meta<-colData(ref)
  newmeta<-meta[metai,]
  gooddata<-newdata[good_ind]
  tmpdata<-Reduce(cbind, gooddata[-1], gooddata[[1]])
  dim(tmpdata)
  dim(newmeta)
  rownames(newmeta)<-make.unique(as.character(metai))
  colnames(tmpdata)<-rownames(newmeta)
  CreateSeuratObject(tmpdata, meta.data = as.data.frame(newmeta))
}


get_counts_se<-function(obj, bulk_assay_name = "counts"){
  #step 1 check for counts
  counts<-assays(obj)[[bulk_assay_name]]
  #step 2 if not found grab any item in the assays slot if == 1
  if(is.null(counts)){
    if(length(assays(obj))>1){stop("More than one assay object found")}
    if(length(assays(obj))==0){stop("No assay object found")}
    counts<-assays(obj)[[1]]
  }
  counts
}