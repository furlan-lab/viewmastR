#' @export
#' @title Finds common features in a list of single cell objects
#'
#' @description Machine learning algorithms often require features to be the same across 
#' datasets.  This function finds common features between a list of cell data set objects (monocle3) and 
#' returns a list of cds's that have the same features.  Note that this function uses rownames 
#' of the 'fData' DataFrame (monocle3) and the rownames of the seurat_object to find the intersect of features common to all objects
#'
#' @param cds_list Input  object.
#' @export
common_features <- function(cds_list){
  len<-length(cds_list)
  common_features=vector()
  software<-NULL
  for(i in 1:len){
    if(i < 2){
      if(class(cds_list[[i]])=="Seurat"){software<-"seurat"}
      if(class(cds_list[[i]])=="cell_data_set"){software<-"monocle3"}
      if(is.null(software)){stop("software not found for input object 1")}
      if(software=="monocle3"){
        common_features<-rownames(fData(cds_list[[i]]))
      }
      if(software=="seurat"){
        common_features<-rownames(cds_list[[i]])
      }
    }else{
      if(software=="monocle3"){
        common_features<-unique(intersect(common_features, rownames(fData(cds_list[[i]]))))
      }
      if(software=="seurat"){
        common_features<-unique(intersect(common_features, rownames(cds_list[[i]])))
      }
    }
  }
  if(software=="monocle3"){
    for(i in 1:len){
      cds_list[[i]]<-cds_list[[i]][match(common_features, rownames(cds_list[[i]])),]
    }
    return(cds_list)
  }
  if(software=="seurat"){
    for(i in 1:len){
      mat<-cds_list[[i]]@assays[[cds_list[[i]]@active.assay]]@counts
      seutmp<- CreateSeuratObject(counts = mat[common_features, ]) # Create a new Seurat object with just the genes of interest
      cds_list[[i]] <- AddMetaData(object = seutmp, metadata = cds_list[[i]]@meta.data) # Add the idents to the meta.data slot
      rm(seutmp, mat)
    }
    return(cds_list)
  }
}



#' Performs TF-IDF transformation on a cell_data_set
#'
#' @description Just like it sounds.
#'
#' @param cds_list Input cell_data_set object or sparse matrix.
#' @importFrom Matrix rowSums
#' @importFrom Matrix colSums
#' @importFrom Matrix Diagonal
#' @importFrom Matrix t
#' @export
#' @keywords internal
tf_idf_transform <- function(input, method=1, verbose=T){
  if(class(input)=="cell_data_set"){
    mat<-exprs(input)
  }else{
    mat<-input
  }
  rn <- rownames(mat)
  row_sums<-rowSums(mat)
  nz<-which(row_sums>0)
  mat <- mat[nz,]
  rn <- rn[nz]
  row_sums <- row_sums[nz]
  col_sums <- colSums(mat)
  
  #column normalize
  mat <-Matrix::t(Matrix::t(mat)/col_sums)
  
  
  if (method == 1) {
    #Adapted from Casanovich et al.
    if(verbose) message("Computing Inverse Document Frequency")
    idf   <- as(log(1 + ncol(mat) / row_sums), "sparseVector")
    if(verbose) message("Computing TF-IDF Matrix")
    mat <- as(Diagonal(x = as.vector(idf)), "sparseMatrix") %*% 
      mat
  }
  else if (method == 2) {
    #Adapted from Stuart et al.
    if(verbose) message("Computing Inverse Document Frequency")
    idf   <- as( ncol(mat) / row_sums, "sparseVector")
    if(verbose) message("Computing TF-IDF Matrix")
    mat <- as(Diagonal(x = as.vector(idf)), "sparseMatrix") %*% 
      mat
    mat@x <- log(mat@x * scale_to + 1)
  }else if (method == 3) {
    mat@x <- log(mat@x + 1)
    if(verbose) message("Computing Inverse Document Frequency")
    idf <- as(log(1 + ncol(mat) /row_sums), "sparseVector")
    if(verbose) message("Computing TF-IDF Matrix")
    mat <- as(Diagonal(x = as.vector(idf)), "sparseMatrix") %*% 
      mat
  }else {
    stop("LSIMethod unrecognized please select valid method!")
  }
  rownames(mat) <- rn
  if(class(input)=="cell_data_set"){
    input@assays$data$counts<-mat
    return(input)
  }else{
    return(mat)
  }
}

#' Performs TF-IDF transformation on a cell_data_set v2
#'
#' @description Just like it sounds but different.
#'
#' @param cds_list Input cell_data_set object or sparse matrix.
#' @importFrom Matrix rowSums
#' @importFrom Matrix colSums
#' @importFrom Matrix Diagonal
#' @importFrom irlba irlba
#' @export
#' @keywords internal
tf_idf_transform_v2 <- function(input){
  if(class(input)=="cell_data_set"){
    mat<-exprs(input)
  }else{
    mat<-input
  }
  colSm <- colSums(mat)
  rowSm <- rowSums(mat)
  freqs <- t(t(mat)/colSm)
  idf   <- as(log(1 + ncol(mat) / rowSm), "sparseVector")
  tfidf <- as(Diagonal(x=as.vector(idf)), "sparseMatrix") %*% freqs
  tfidf@x[is.na(tfidf@x)] <- 0
  if(class(input)=="cell_data_set"){
    input@assays$data$counts<-tfidf
    return(input)
  }else{
    return(tfidf)
  }
}

#' @export
svd_lsi<-function(sp_mat, num_dim, mat_only=T){
  svd <- irlba(sp_mat, num_dim, num_dim)
  svdDiag <- matrix(0, nrow=num_dim, ncol=num_dim)
  diag(svdDiag) <- svd$d
  matSVD <- t(svdDiag %*% t(svd$v))
  rownames(matSVD) <- colnames(sp_mat)
  colnames(matSVD) <- seq_len(ncol(matSVD))
  if(mat_only){
    return(matSVD)
  }else{
    return(list(matSVD=matSVD, svd=svd))
  }
}


Noisify <- function(data, amount=0.0001) {
  if (is.vector(data)) {
    noise <- runif(length(data), -amount, amount)
    noisified <- data + noise
  } else {
    length <- dim(data)[1] * dim(data)[2]
    noise <- matrix(runif(length, -amount, amount), dim(data)[1])
    noisified <- data + noise
  }
  return(noisified)
}


#' Detects genes above minimum threshold.
#'
#' @description For each gene in a cell_data_set object, detect_genes counts
#' how many cells are expressed above a minimum threshold. In addition, for
#' each cell, detect_genes counts the number of genes above this threshold that
#' are detectable. Results are added as columns num_cells_expressed and
#' num_genes_expressed in the rowData and colData tables respectively.
#'
#' @param cds Input cell_data_set object.
#' @param min_expr Numeric indicating expression threshold
#' @param exprs_bin Boolean whether to bin genes by mean expression
#' @param exprs_cuts Numeic indicating number of bins if using exprs_bin
#' @return Updated cell_data_set object
#' @importFrom Hmisc cut2
#' @importFrom assertthat assert_that
#' @importFrom Matrix rowSums
#' @importFrom Matrix colSums
#' @importFrom Matrix rowMeans
#' @importFrom SingleCellExperiment counts
#' @export
#' @keywords internal
detect_genes <- function(cds, min_expr=0, exprs_bin=TRUE, exprs_cuts=25){
  assert_that(methods::is(cds, "cell_data_set"))
  assert_that(is.numeric(min_expr))
  
  rowData(cds)$num_cells_expressed <- rowSums((cds) > min_expr)
  colData(cds)$num_genes_expressed <- colSums(counts(cds) > min_expr)
  if(exprs_bin){
    fData(cds)$exprs_bin = cut2(log(rowMeans(normalized_counts(cds))), m=floor(nrow(fData(cds))/exprs_cuts))
  }
  cds
}



#' Write a compressed MatrixMarket file
#'
#' This function writes a sparse matrix in the MatrixMarket format to a compressed `.gz` file.
#' The function handles both real and integer matrix types.
#'
#' @param x A sparse matrix (typically a \code{dgCMatrix} or \code{ngCMatrix} object).
#' @param file A character string specifying the output file name, which will be compressed into `.gz` format.
#'
#' @details
#' This function writes the matrix in the MatrixMarket coordinate format. 
#' It first writes the header indicating the matrix type and size, and then appends the matrix data.
#' If the matrix is an `ngCMatrix`, it is treated as an integer matrix, otherwise as a real matrix.
#' The function compresses the output into a `.gz` file.
#'
#' @importFrom data.table fwrite
#' @importFrom Matrix summary
#'
#' @return This function does not return a value. It writes a file as a side effect.
#' @export
#' @examples
#' \dontrun{
#' library(Matrix)
#' m <- Matrix(c(0, 1, 0, 2), 2, 2, sparse = TRUE)
#' writeMMgz(m, "matrix.mtx.gz")
#' }
writeMMgz <- function(x, file) {
  mtype <- "real"
  if (is(x, "ngCMatrix")) {
    mtype <- "integer"
  }
  writeLines(
    c(
      sprintf("%%%%MatrixMarket matrix coordinate %s general", mtype),
      sprintf("%s %s %s", x@Dim[1], x@Dim[2], length(x@x))
    ),
    gzfile(file)
  )
  fwrite(
    x = summary(x),
    file = file,
    append = TRUE,
    sep = " ",
    row.names = FALSE,
    col.names = FALSE
  )
}


#' Export a Seurat Object to 10X-Style Files
#'
#' @description
#' Write the counts matrix, features, barcodes, metadata, variable
#' features and—optionally—reduction embeddings from a Seurat object in
#' the 10X “3-file” layout.
#'
#' @param seu            A Seurat object.
#' @param assay          Which assay to export (default `"RNA"`).
#' @param dir            Output directory (must already exist).
#' @param get_reductions Logical; also export reduction embeddings
#'                       (default `TRUE`).
#'
#' @details
#' The function creates a sub-directory called \code{3file} inside
#' \code{dir} and writes:
#'
#' \itemize{
#'   \item{\file{matrix.mtx.gz}}{Compressed Matrix Market file containing the
#'     counts matrix.}
#'   \item{\file{features.tsv.gz}}{Gene (feature) table.}
#'   \item{\file{barcodes.tsv.gz}}{Cell barcodes.}
#'   \item{\file{meta.csv}}{Cell-level metadata.}
#'   \item{\file{\<reduction\>_reduction.tsv.gz}}{Embeddings for each
#'     reduction (UMAP, PCA, …); written only when
#'     \code{get_reductions = TRUE}.}
#'   \item{\file{variablefeatures.tsv.gz}}{Variable-gene list.}
#' }
#'
#' @return Invisibly returns \code{NULL}; called for its side effects.
#'
#' @importFrom Seurat GetAssayData Cells VariableFeatures
#' @importFrom utils   write.csv
#' @importFrom R.utils gzip
#'
#' @examples
#' \dontrun{
#' make3file(seu, assay = "RNA", dir = "out", get_reductions = FALSE)
#' }
#' @export

make3file <- function(seu, assay = "RNA", dir, get_reductions = TRUE) {
  # Check if the directory exists
  if (!file.exists(dir)) {
    stop("Must provide a valid directory.")
  }
  
  # Create the 3file subdirectory if it doesn't exist
  out_dir <- file.path(dir, "3file")
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE)
  }
  
  # Write matrix.mtx.gz file
  mm <- file.path(out_dir, "matrix.mtx.gz")
  writeMMgz(GetAssayData(seu, assay = assay), mm)
  
  # Write features.tsv.gz file (gene information)
  genes <- data.frame(rownames(seu), rownames(seu), rep("Gene Expression", length(rownames(seu))))
  gz2 <- gzfile(file.path(out_dir, "features.tsv.gz"), "w")
  write.table(genes, gz2, sep = "\t", quote = FALSE, col.names = FALSE, row.names = FALSE)
  close(gz2)
  
  # Write barcodes.tsv.gz file (cell barcodes)
  gz3 <- gzfile(file.path(out_dir, "barcodes.tsv.gz"), "w")
  cb <- Cells(seu)
  writeLines(cb, gz3)
  close(gz3)
  
  # Write metadata to meta.csv
  meta <- file.path(out_dir, "meta.csv")
  write.csv(seu@meta.data, meta, quote = FALSE)
  
  # Write reductions if requested
  if (get_reductions) {
    reds <- names(seu@reductions)
    for (red in reds) {
      # Retrieve the reduction embeddings
      data <- seu@reductions[[red]]@cell.embeddings
      # Open gzipped reduction file
      gzr <- gzfile(file.path(out_dir, paste0(red, "_reduction.tsv.gz")), "w")
      # Write the reduction embeddings to the file
      write.table(data, gzr, sep = "\t", quote = FALSE, col.names = NA)
      close(gzr)
    }
  }
  
  # Write variable features to variablefeatures.tsv.gz
  gz4 <- gzfile(file.path(out_dir, "variablefeatures.tsv.gz"), "w")
  writeLines(VariableFeatures(seu), gz4)
  close(gz4)
}


#' @keywords internal
decode_param <- function(param) {
  bytes  <- param$bytes
  dtype  <- param$dtype
  shape  <- param$shape
  con    <- rawConnection(bytes, "rb")
  on.exit(close(con), add = TRUE)
  
  vals <- switch(dtype,
                 "F32" = readBin(con, "numeric", n = prod(shape), size = 4, endian = "little"),
                 "F64" = readBin(con, "numeric", n = prod(shape), size = 8, endian = "little"),
                 "I32" = readBin(con, "integer", n = prod(shape), size = 4, endian = "little"),
                 "I64" = readBin(con, "integer", n = prod(shape), size = 8, endian = "little"),
                 "F16" = read_f16(bytes),   # see note below
                 "BF16"= read_bf16(bytes),  # idem
                 stop("Unsupported dtype: ", dtype)
  )
  array(vals, dim = shape)
}




# --- helper: convert 16-bit bfloat16 to double -------------------------
#' @keywords internal
# ---------- IEEE-754 half-precision (float16) --------------------------
read_f16 <- function(raw_vec, endian = "little") {
  stopifnot(length(raw_vec) %% 2 == 0)
  
  ints <- readBin(raw_vec, "integer",
                  n      = length(raw_vec) / 2,
                  size   = 2,
                  endian = endian,
                  signed = FALSE)
  
  sign <- ifelse(bitwAnd(ints, 0x8000L) != 0L, -1, 1)
  exp  <- bitwShiftR(bitwAnd(ints, 0x7C00L), 10)          # <<— fixed
  frac <- bitwAnd(ints, 0x03FFL)
  
  val <- ifelse(
    exp == 0L,
    sign * 2^(-14) * (frac / 1024),
    ifelse(
      exp == 0x1FL,
      NaN,
      sign * 2^(exp - 15) * (1 + frac / 1024)
    )
  )
  as.numeric(val)
}

# ---------- bfloat16 ---------------------------------------------------
#' @keywords internal
read_bf16 <- function(raw_vec, endian = "little") {
  stopifnot(length(raw_vec) %% 2 == 0)
  
  ints <- readBin(raw_vec, "integer",
                  n      = length(raw_vec) / 2,
                  size   = 2,
                  endian = endian,
                  signed = FALSE)
  
  sign <- ifelse(bitwAnd(ints, 0x8000L) != 0L, -1, 1)
  exp  <- bitwShiftR(bitwAnd(ints, 0x7F80L), 7)           # <<— fixed
  mant <- bitwAnd(ints, 0x007FL)
  
  val <- ifelse(
    exp == 0L,
    sign * 2^(-126) * (mant / 128),
    ifelse(
      exp == 0xFFL,
      NaN,
      sign * 2^(exp - 127) * (1 + mant / 128)
    )
  )
  as.numeric(val)
}

#' Extract linear-layer weights and map them to feature / class names
#'
#' Reads a model exported by the Rust Burn pipeline together with its
#' companion metadata file and returns a tidy **weight matrix** whose rows
#' correspond to the original feature names and whose columns correspond to
#' the class labels.
#'
#' @param dir `character(1)`  Path to the artefact directory that contains
#'   *both* files produced by `run_custom()`:
#'
#'   * `model.mpk`  – tensor weights saved via
#'     `NamedMpkFileRecorder::<FullPrecisionSettings>`
#'   * `meta.mpk`   – MessagePack blob created by \code{save_artifacts()}
#'     holding `feature_names` and `class_labels`.
#'
#' @return A base-`data.frame` with dimension
#'   \eqn{(\#\;features) \times (\#\;classes)}, where
#'   `rownames(wmat)` are the feature names and
#'   `colnames(wmat)` are the class labels.  Cell *(i,j)* is the weight
#'   connecting feature *i* to logit *j*.
#'
#' @details
#' Internally the function:
#'
#' 1. deserialises the two MessagePack files with **msgpackR**;
#' 2. raw‐decodes the tensor bytes through \code{decode_param()};
#' 3. reshapes the flat vector into a column-major matrix using the stored
#'    shape (`[out_dim, in_dim]`);
#' 4. transposes it so that rows align with features;
#' 5. re-labels rows and columns from the metadata lists.
#'
#' The resulting object is ready for
#' `pheatmap()`, `corrplot()`, or `as.matrix()` for further analysis.
#'
#' @seealso
#' * `msgpack_read()` from **msgpackR** – generic MessagePack reader
#' * `decode_param()` – helper that converts Burn tensor blobs into R vectors
#'
#' @examples
#' \dontrun{
#' w <- get_weights("artifacts/run-42")
#' head(w[, 1:5])         # first 5 classes
#'
#' # visualise top positive / negative features for class 3
#' cls <- 3
#' w_sorted <- w[order(w[, cls]), cls]
#' barplot(tail(w_sorted, 10), horiz = TRUE, las = 1)
#' barplot(head(w_sorted, 10), horiz = TRUE, las = 1)
#' }
#'
#' @export
#' @importFrom RcppMsgPack msgpack_read
#' @keywords internal
#' 
get_weights <- function(dir){
  mod <- msgpack_read(file.path(dir, "model.mpk"), simplify = TRUE)
  meta <- msgpack_read(file.path(dir, "meta.mpk"), simplify = TRUE)
  weights <- decode_param(mod$item$linear1$weight$param)
  shape <- mod$item$linear1$weight$param$shape
  wmat <- data.frame(t(matrix(weights, nrow = shape[2])))
  rownames(wmat) <- meta$feature_names
  colnames(wmat) <- meta$class_labels
  wmat
}




