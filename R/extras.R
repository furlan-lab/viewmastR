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
