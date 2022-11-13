#' Project data from one object into the UMAP embedding a single cell object
#' 
#' @description Adapted from: ArchR: An integrative and scalable software package for single-cell chromatin accessibility analysis
#' Jeffrey M. Granja, M. Ryan Corces, Sarah E. Pierce, S. Tansu Bagdatli, Hani Choudhry, Howard Y. Chang, William J. Greenleaf
#' doi: https://doi.org/10.1101/2020.04.28.066498
#' 
#' @param projector cell_data_set object with a reduced dimension matrix (currently LSI supported) as specified in
#' reduced_dim argument and a model used to create a low dimensional embedding
#' @param projectee a SummarizedExperiment type object (cell_data_set currently supported) to be projected using 
#' the models contained in the projector
#' @param make_pseudo_single_cells whether to make pseudo-single cells from the data in the projectee (set this to true for bulk data)
#' @param ncells_coembedding number of cells in the projector to use in in the co-embedding with simulated single cells; default is 5000, will automatically
#' default to the total number of cells in the projector if less than this value.
#' @param reduced_dim A string specifying the reducedDim (currently LSI supported).
#' @param embedding A string specifying embedding type (currently UMAP supported).
#' @param n An integer specifying the number of subsampled "pseudo single cells" per bulk sample.  Note this is only relevant if 
#' make_pseudo_single_cells is TRUE
#' @param verbose A boolean value indicating whether to use verbose output during execution of this function. Can be set to FALSE for a cleaner output.
#' @param threads The number of threads used for parallel execution
#' @export

project_data <- function(
  projector = NULL,
  projectee = NULL,
  ncells_coembedding = 5000,
  scale=F,
  reduced_dim = "LSI",
  embedding = "UMAP",
  make_pseudo_single_cells = FALSE,
  features = c("annotation-based", "range-based"),
  n = 250,
  verbose = TRUE,
  threads = 6,
  seed=2020,
  force=F
){
  check_input(input = projector, name = "projector", valid = c("cell_data_set"))
  check_input(input = projectee, name = "projectee", valid = c("SummarizedExperiment"))
  check_input(input = ncells_coembedding , name = "ncells_coembedding ", valid = c("numeric"))
  check_input(input = reduced_dim, name = "reduced_dim", valid = c("character"))
  check_input(input = embedding, name = "embedding", valid = c("character", "null"))
  check_input(input = n, name = "n", valid = c("integer"))
  check_input(input = verbose, name = "verbose", valid = c("boolean"))
  check_input(input = threads, name = "threads", valid = c("integer"))
  
  #backwards compatible
  if(reduced_dim=="LSI"){
    reduced_dim_aux<-"iLSI"
  }else{
    reduced_dim_aux<-"PCA"
  }
  ##################################################
  # Extract data from bulk
  ##################################################
  rD<-reducedDims(projector)[[reduced_dim]]
  #rD_num_dum<-projector@preprocess_aux$iLSI$num_dim
  match(names(projector@reduce_dim_aux), embedding)
  embedding_num_dim<-projector@reduce_dim_aux[[embedding]]$num_dim
  
  sc_embedding<-reducedDims(projector)[[embedding]]
  rownames(sc_embedding)<-colnames(projector)
  
  if(features[1] %in% "annotation-based"){
    query<-projector@preprocess_aux[[reduced_dim_aux]]$features
    shared_rd<-extract_data(query, projectee)
  }
  
  if(features[1] %in% "range-based"){
    query<-projector@preprocess_aux[[reduced_dim_aux]]$granges
    shared_rd<-extract_data(query, projectee)
  }
  
  message(paste0("Overlap Ratio of Reduced Dims Features = ", round(shared_rd$overlap, 3)))
  
  if( (shared_rd$overlap) < 0.25 ){
    if(force){
      warning("Less than 25% of the features are present in this bulk RNA data set! Continuing since force = TRUE!")
    }else{
      stop("Less than 25% of the features are present in this bulk RNA data set! Set force = TRUE to continue!")
    }
  }
  
  ##################################################
  # Simulate single cells and project using original LSI/SVD model
  ##################################################
  if(make_pseudo_single_cells){
    depthN <- round(sum(projector@preprocess_aux[[reduced_dim_aux]]$row_sums / nrow(rD)))
    nRep <- 5
    n2 <- ceiling(n / nRep)
    ratios <- c(2, 1.5, 1, 0.5, 0.25) #range of ratios of number of fragments
  
    if(verbose) message(paste0("Simulating ", (n * dim(projectee)[2]), " single cells"))
    projRD <- pbmcapply::pbmclapply(seq_len(ncol(shared_rd$mat)), function(x){
      counts <- shared_rd$mat[, x]
      counts <- rep(seq_along(counts), counts)
      simMat <- lapply(seq_len(nRep), function(y){
        ratio <- ratios[y]
        simMat <- matrix(sample(x = counts, size = ceiling(ratio * depthN) * n2, replace = TRUE), ncol = n2)
        simMat <- Matrix::summary(as(simMat, "dgCMatrix"))[,-1,drop=FALSE]
        simMat[,1] <- simMat[,1] + (y - 1) * n2
        simMat
      }) %>%  Reduce("rbind", .)
      simMat <- Matrix::sparseMatrix(i = simMat[,2], j = simMat[,1], x = rep(1, nrow(simMat)), dims = c(nrow(shared_rd$mat), n2 * nRep))
      projRD <- as.matrix(projectLSI(simMat, LSI = projector@preprocess_aux[[reduced_dim_aux]], verbose = verbose))
      rownames(projRD) <- paste0(colnames(shared_rd$mat)[x], "#", seq_len(nrow(projRD)))
      projRD
    }, mc.cores =  threads) %>% Reduce("rbind", .)
  
    # Deal with NaN
    if(any(is.nan(projRD))){
      projRD[is.nan(projRD)]<-0
      warning("NaN calculated during single cell generation")
    }
    
    if(scale){
      projRD <- scale_dims(projRD)
    }
  }else{
    projRD <- as.matrix(projectLSI(shared_rd$mat, LSI = projector@preprocess_aux[[reduced_dim_aux]], verbose = verbose))
  }
  
  ##################################################
  # Check LSI and Embedding SVD columns
  ##################################################  
  
  
  if(embedding_num_dim != ncol(projRD)){
    stop("Error incosistency found with matching LSI dimensions to those used in embedding")
  }
  
  ##################################################
  # Get Previous UMAP Model
  ##################################################
  umap_model <- load_umap_model(projector@reduce_dim_aux[[embedding]]$model_file, embedding_num_dim)
  
  ##################################################
  # subsample
  ##################################################
  idx <- sort(sample(seq_len(nrow(rD)), min(nrow(rD), ncells_coembedding)))
  rD_ss <- rD[idx,,drop=FALSE]
  
  ##################################################
  # Project UMAP
  ##################################################
  if(verbose & make_pseudo_single_cells) message(paste0("Projecting simulated doublets onto manifold saved in: ", projector@reduce_dim_aux[[embedding]]$model_file))
  if(verbose & !make_pseudo_single_cells) message(paste0("Projecting projectee cells onto manifold saved in: ", projector@reduce_dim_aux[[embedding]]$model_file))
  set.seed(seed)
  #threads2 <- max(floor(threads/2), 1)
  simUMAP <- uwot::umap_transform(
    X = rbind(rD_ss, projRD), 
    model = umap_model, 
    verbose = verbose, 
    n_threads = threads
  )
  rownames(simUMAP) <- c(rownames(rD_ss), rownames(projRD))
  ##################################################
  # Check correlation of subsampled cells
  ##################################################
  c1 <- cor(simUMAP[rownames(rD_ss), 1], sc_embedding[rownames(rD_ss),1])
  c2 <- cor(simUMAP[rownames(rD_ss), 2], sc_embedding[rownames(rD_ss),2])
  if(min(c1, c2) < 0.8){
    message(paste0("Warning projection correlation is less than 0.8 (R = ", round(min(c1,c2), 4),").\nThese results may not be accurate because of the lack of heterogeneity in the single cell data."))
  }
  
  dfUMAP <- sc_embedding
  colnames(dfUMAP) <- c("UMAP1", "UMAP2")
  colnames(simUMAP) <- c("UMAP1", "UMAP2")
  dfUMAP <- DataFrame(dfUMAP)
  dfUMAP$Type <- Rle("single_cell", lengths = nrow(dfUMAP))
  
  simUMAP <- DataFrame(simUMAP[rownames(projRD),,drop=FALSE])
  simUMAP$Type <- Rle(stringr::str_split(rownames(simUMAP), pattern = "#", simplify = TRUE)[,1])
  
  out <- SimpleList(
    simulatedBulkUMAP = simUMAP,
    singleCellUMAP = dfUMAP,
    simulatedReducedDims = projRD
  )
  return(out)
}

#' Liftover projection
#' 
#' @description This function transfers labels from a projector to a projectee
#' @param projection projection object created by project_data function
#' @param projector cell_data_set object with a reduced dimension matrix (currently LSI supported) as specified in
#' reduced_dim argument and a model used to create a low dimensional embedding
#' @param projectee a SummarizedExperiment type object (cell_data_set currently supported) to be projected using 
#' the models contained in the projector
#' @param projector_col column name of projector to color cells by using nearest neighbor
#' @param projectee_col column name of projectee to color cells by
#' @param colors color palatte of projectee_col cells
#' @export
liftover_projection<-function(projection, projector, projectee, projector_col=NULL){
  closest<-RANN::nn2(data = data.frame(UMAP1=projection[[2]]$UMAP1, UMAP2=projection[[2]]$UMAP2), query = data.frame(UMAP1=projection[[1]]$UMAP1, UMAP2=projection[[1]]$UMAP2), k = 1)
  projectee$liftover<-projector[[projector_col]][closest$nn.idx]
  projectee
}


#' Plot a projection
#' 
#' @description This function plots a projection object created using the project_data function
#' @param projection projection object created by project_data function
#' @param projector cell_data_set object with a reduced dimension matrix (currently LSI supported) as specified in
#' reduced_dim argument and a model used to create a low dimensional embedding
#' @param projectee a SummarizedExperiment type object (cell_data_set currently supported) to be projected using 
#' the models contained in the projector
#' @param projector_col column name of projector to color cells by using nearest neighbor
#' @param projectee_col column name of projectee to color cells by
#' @param colors color palatte of projectee_col cells
#' @export
plot_projection<-function(projection, projector, projectee, projector_col=NULL, projectee_col=NULL){
  projection<-as.list(projection)
  closest<-RANN::nn2(data = data.frame(UMAP1=projection[[2]]$UMAP1, UMAP2=projection[[2]]$UMAP2), query = data.frame(UMAP1=projection[[1]]$UMAP1, UMAP2=projection[[1]]$UMAP2), k = 1)
  projection[[2]]$neighbor<-"0_scRNA"
  if(!is.null(projector_col)){
    projection[[1]]$neighbor<-projector[[projector_col]][closest$nn.idx]
  }
  if(!is.null(projectee_col)){
    projection[[1]]$neighbor<-colData(projectee)[[projectee_col]]
  }
  p<-as.data.frame(do.call(rbind, projection[2:1]))
  g<-ggplot(p, aes(UMAP1, UMAP2, color=neighbor))+geom_point(size=0.3)+SFtheme
}



#' Extract data using a set of features
#' 
#' @description This function extracts data from a subject object (a SummarizedExperiment-like 
#' object) using the features specified in query.  The function is parameterized to allow for finding features by either feature
#' name (annotation-based) or using GenomicRanges (range-based).  Annotation-based searching defaults
#' to the rownames of the subject, but this can be altered using annotation argument
#' which will search for these colnames in the subject's feature metadata.  The default behavior
#' of this function is to return a SummarizedExperiment object with the full feature-set of the query and thus
#' will add zeros for those features in the query not found in the subject. Set the full-output argument to TRUE
#' to modify this behavior
#' 
#' @param query a vector of feature names or a GrangesObject
#' @param subject a SummarizedExperiment based object
#' @param annotation = A string specifying the target feature search for oin subject; Default is "row.names"
#' @param duplicate_hits parameter for dealing with multiple subject hits in a range-based search.  Options are to select those
#' multiple hits based on the following: "max.mean", "max.var", "max.disp", "min.mean", "min.var", "min.disp". Default is "max.disp"
#' @param fill_output = A boolean indicating whether the function should return a subject object filled with zero data for 
#' features in query that were not found in the subject
#' @param ignore_strand Whether to ignore strand for range-based searches
#' @param verbose A boolean value indicating whether to use verbose output during execution of this function. Can 
#' be set to FALSE for a cleaner output.
#' @return a list cotaining 1) matrix of the extracted data, 2) the ratio of features
#' found in the subject to those present in the query, 3) those features not_found
#' @export
#' 
extract_data<-function(query, 
                       subject,
                       annotation="row.names",
                       verbose=T,
                       fill_output=T,
                       duplicate_hits="max.disp",
                       ignore_strand=T){
  ##################################################
  # Check Inputs
  ##################################################
  check_input(input = query, name = "query", valid = c("character", "Granges"))
  check_input(input = subject, name = "subject", valid = c("SummarizedExperiment"))
  check_input(input = annotation, name = "annotation", valid = c("NULL", "character"))
  check_input(input = verbose, name = "verbose", valid = c("boolean"))
  check_input(input = fill_output, name = "fill_output", valid = c("boolean"))
  
  ##################################################
  # parse duplicate_hits behavior
  ################################################## 
  FUN<-get(strsplit(duplicate_hits, "\\.")[[1]][1])
  Var1<-strsplit(duplicate_hits, "\\.")[[1]][2]
  
  if(class(query)=="GRanges") {
    #rename features as coordinates are no longer meaningful
    fs<-names(query)<-paste("f", seq_along(query))
    se_sub <- subsetByOverlaps( subject, query, ignore.strand = ignore_strand, type="any")
    #Remove dups
    hits<-data.table::as.data.table(findOverlaps(query, se_sub, ignore.strand = TRUE))
    dat<-as.matrix(get_assay(se_sub))
    hits$var<-rowVars(dat)[hits$subjectHits]
    hits$mean<-rowMeans(dat)[hits$subjectHits]
    hits$disp<-sqrt(hits$var)/hits$mean*100
    best_hits <- hits %>% group_by(queryHits) %>% filter(get(Var1) == FUN(get(Var1)))
    if(nrow(best_hits) == 0){
      stop(paste0("No overlap between query and subject found."))
    }
    mat<-dat[best_hits$subjectHits,]
    rownames(mat)<-names(query)[best_hits$queryHits]
    if(fill_output)  mat<-safe_subset(mat, subsetRows = fs)
    overlap=nrow(best_hits)/length(fs)
    not_found<-query[!best_hits$queryHits %in% 1:length(names(fs)),]
  }else{
    if(annotation=="row.names") {
      fidx <- which(rownames(subject) %in% query)
      if(length(fidx) == 0){
        stop(paste0("No overlap between query and subject found."))
      }
      fs<-query
      mat<-as(get_assay(subject), "dgCMatrix")[fidx,]
      if(is.null(rownames(mat))){
        rownames(mat)<-rownames(subject)[fidx]
      }
      if(fill_output)  mat<-safe_subset(mat, subsetRows = fs)
      overlap<-length(fidx)/length(query)
      not_found<-query[!fidx %in% 1:length(query)]
    }else{
      stop("not_implemented")
    }
  }
  return(list(mat=mat, overlap=overlap, notfound=not_found))
}


#'check_input helper
#'
#' @description
#' Adapted from: Jeffrey M. Granja, M. Ryan Corces, Sarah E. Pierce, S. Tansu Bagdatli, Hani Choudhry, Howard Y. Chang, William J. Greenleaf
#' doi: https://doi.org/10.1101/2020.04.28.066498
#' @export
check_input<-function (input = NULL, name = NULL, valid = NULL) 
{
  valid <- unique(valid)
  if (is.character(valid)) {
    valid <- tolower(valid)
  }
  else {
    stop("Validator must be a character!")
  }
  if (!is.character(name)) {
    stop("name must be a character!")
  }
  if ("null" %in% tolower(valid)) {
    valid <- c("null", valid[which(tolower(valid) != "null")])
  }
  av <- FALSE
  for (i in seq_along(valid)) {
    vi <- valid[i]
    if (vi == "integer" | vi == "wholenumber") {
      if (all(is.numeric(input))) {
        cv <- min(abs(c(input%%1, input%%1 - 1))) < .Machine$double.eps^0.5
      }
      else {
        cv <- FALSE
      }
    }
    else if (vi == "null") {
      cv <- is.null(input)
    }
    else if (vi == "bool" | vi == "boolean" | vi == "logical") {
      cv <- is.logical(input)
    }
    else if (vi == "numeric") {
      cv <- is.numeric(input)
    }
    else if (vi == "vector") {
      cv <- is.vector(input)
    }
    else if (vi == "matrix") {
      cv <- is.matrix(input)
    }
    else if (vi == "sparsematrix") {
      cv <- is(input, "dgCMatrix")
    }
    else if (vi == "character") {
      cv <- is.character(input)
    }
    else if (vi == "factor") {
      cv <- is.factor(input)
    }
    else if (vi == "cell_data_set") {
      cv <- is(input, "cell_data_set")
    }
    else if (vi == "rlecharacter") {
      cv1 <- is(input, "Rle")
      if (cv1) {
        cv <- is(input@values, "factor") || is(input@values, 
                                               "character")
      }
      else {
        cv <- FALSE
      }
    }
    else if (vi == "palette") {
      cv <- all(.isColor(input))
    }
    else if (vi == "timestamp") {
      cv <- is(input, "POSIXct")
    }
    else if (vi == "dataframe" | vi == "data.frame" | vi == 
             "df") {
      cv1 <- is.data.frame(input)
      cv2 <- is(input, "DataFrame")
      cv <- any(cv1, cv2)
    }
    else if (vi == "fileexists") {
      cv <- all(file.exists(input))
    }
    else if (vi == "direxists") {
      cv <- all(dir.exists(input))
    }
    else if (vi == "granges" | vi == "gr") {
      cv <- is(input, "GRanges")
    }
    else if (vi == "grangeslist" | vi == "grlist") {
      cv <- .isGRList(input)
    }
    else if (vi == "list" | vi == "simplelist") {
      cv1 <- is.list(input)
      cv2 <- is(input, "SimpleList")
      cv <- any(cv1, cv2)
    }
    else if (vi == "bsgenome") {
      cv1 <- is(input, "BSgenome")
      cv2 <- tryCatch({
        library(input)
        eval(parse(text = input))
      }, error = function(e) {
        FALSE
      })
      cv <- any(cv1, cv2)
    }
    else if (vi == "se" | vi == "summarizedexperiment") {
      cv <- is(input, "SummarizedExperiment")
    }
    else if (vi == "seurat" | vi == "seuratobject") {
      cv <- is(input, "Seurat")
    }
    else if (vi == "txdb") {
      cv <- is(input, "TxDb")
    }
    else if (vi == "orgdb") {
      cv <- is(input, "OrgDb")
    }
    else if (vi == "bsgenome") {
      cv <- is(input, "BSgenome")
    }
    else if (vi == "parallelparam") {
      cv <- is(input, "BatchtoolsParam")
    }
    else {
      stop("Validator is not currently supported")
    }
    if (cv) {
      av <- TRUE
      break
    }
  }
  if (av) {
    return(invisible(TRUE))
  }
  else {
    stop("Input value for '", name, "' is not a ", paste(valid, 
                                                         collapse = ","), ", (", name, " = ", class(input), 
         ") please supply valid input!")
  }
}

#' safe_subset helper
#'
#' @description
#' Adapted from: Jeffrey M. Granja, M. Ryan Corces, Sarah E. Pierce, S. Tansu Bagdatli, Hani Choudhry, Howard Y. Chang, William J. Greenleaf
#' doi: https://doi.org/10.1101/2020.04.28.066498
#' 
#' @export
safe_subset<-function (mat = NULL, subsetRows = NULL, subsetCols = NULL) 
{
  if (!is.null(subsetRows)) {
    idxNotIn <- which(!subsetRows %in% rownames(mat))
    if (length(idxNotIn) > 0) {
      subsetNamesNotIn <- subsetRows[idxNotIn]
      matNotIn <- Matrix::sparseMatrix(i = 1, j = 1, x = 0, 
                                       dims = c(length(idxNotIn), ncol = ncol(mat)))
      dim(matNotIn)
      dim(mat)
      rownames(matNotIn) <- subsetNamesNotIn
      mat <- rbind(mat, matNotIn)
    }
    mat <- mat[subsetRows, ]
  }
  if (!is.null(subsetCols)) {
    idxNotIn <- which(subsetCols %ni% colnames(mat))
    if (length(idxNotIn) > 0) {
      subsetNamesNotIn <- subsetCols[idxNotIn]
      matNotIn <- Matrix::sparseMatrix(i = 1, j = 1, x = 0, 
                                       dims = c(nrow(mat), ncol = length(idxNotIn)))
      colnames(matNotIn) <- subsetNamesNotIn
      mat <- cbind(mat, matNotIn)
    }
    mat <- mat[, subsetCols]
  }
  mat
}

#' get_assay helper
#'
#' @description
#' Adapted from: Jeffrey M. Granja, M. Ryan Corces, Sarah E. Pierce, S. Tansu Bagdatli, Hani Choudhry, Howard Y. Chang, William J. Greenleaf
#' doi: https://doi.org/10.1101/2020.04.28.066498
#' 
#' @export
get_assay<-function (se = NULL, assayName = NULL) 
{
  .assayNames <- function(se) {
    names(SummarizedExperiment::assays(se))
  }
  if (is.null(assayName)) {
    o <- SummarizedExperiment::assay(se)
  }
  else if (assayName %in% .assayNames(se)) {
    o <- SummarizedExperiment::assays(se)[[assayName]]
  }
  else {
    stop(sprintf("assayName '%s' is not in assayNames of se : %s", 
                 assayName, paste(.assayNames(se), collapse = ", ")))
  }
  return(o)
}

#' @export
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

#' Project LSI Function
#' @description
#' Adapted from: Jeffrey M. Granja, M. Ryan Corces, Sarah E. Pierce, S. Tansu Bagdatli, Hani Choudhry, Howard Y. Chang, William J. Greenleaf
#' doi: https://doi.org/10.1101/2020.04.28.066498
#' @export
projectLSI<-function (mat = NULL, LSI = NULL, returnModel = FALSE, verbose = FALSE, seed=2020) 
{
  out2 <- tryCatch({
    require(Matrix)
    set.seed(seed)
    if(verbose) message(sprintf("Projecting LSI, Input Matrix = %s GB", 
                                round(object.size(mat)/10^9, 3)))
    if(verbose) message("Subsetting by Non-Zero features in inital Matrix")
    #mat <- mat[LSI$idx, ] I don't see what this does after safeSubset
    if (LSI$binarize) {
      if(verbose) message("Binarizing Matrix")
      mat@x[mat@x > 0] <- 1
    }
    if(verbose) message("Computing Term Frequency")
    colSm <- Matrix::colSums(mat)
    if (any(colSm == 0)) {
      exclude <- which(colSm == 0)
      mat <- mat[, -exclude]
      colSm <- colSm[-exclude]
    }
    mat@x <- mat@x/rep.int(colSm, Matrix::diff(mat@p))
    if (LSI$LSI_method == 1) {
      #Adapted from Casanovich et al.
      
      if(verbose) message("Computing Inverse Document Frequency")
      idf   <- as(log(1 + nrow(LSI$svd$v) / LSI$row_sums), "sparseVector")
      if(verbose) message("Computing TF-IDF Matrix")
      mat <- as(Matrix::Diagonal(x = as.vector(idf)), "sparseMatrix") %*% 
        mat
    }
    else if (LSI$LSI_method == 2) {
      #Adapted from Stuart et al.
      if(verbose) message("Computing Inverse Document Frequency")
      idf   <- as( nrow(LSI$svd$v) / LSI$row_sums, "sparseVector")
      if(verbose) message("Computing TF-IDF Matrix")
      mat <- as(Matrix::Diagonal(x = as.vector(idf)), "sparseMatrix") %*% 
        mat
      mat@x <- log(mat@x * LSI$scale_to + 1)
    }else if (LSI$LSI_method == 3) {
      mat@x <- log(mat@x + 1)
      if(verbose) message("Computing Inverse Document Frequency")
      idf <- as(log(1 + nrow(LSI$svd$v) /LSI$row_sums), "sparseVector")
      if(verbose) message("Computing TF-IDF Matrix")
      mat <- as(Matrix::Diagonal(x = as.vector(idf)), "sparseMatrix") %*% 
        mat
    }else {
      stop("LSIMethod unrecognized please select valid method!")
    }
    gc()
    idxNA <- Matrix::which(is.na(mat), arr.ind = TRUE)
    if (length(idxNA) > 0) {
      if(verbose) message((sprintf("Zeroing %s NA elements", length(idxNA))))
      mat[idxNA] <- 0
    }
    if(verbose) message("Calculating V Matrix")
    V <- Matrix::t(mat) %*% LSI$svd$u %*% Matrix::diag(1/LSI$svd$d)
    if(verbose) message("Computing Projected Coordinates")
    svdDiag <- matrix(0, nrow = LSI$num_dim, ncol = LSI$num_dim)
    diag(svdDiag) <- LSI$svd$d
    matSVD <- Matrix::t(svdDiag %*% Matrix::t(V))
    matSVD <- as.matrix(matSVD)
    rownames(matSVD) <- colnames(mat)
    colnames(matSVD) <- paste0("LSI", seq_len(ncol(matSVD)))
    if (returnModel) {
      if(verbose) message("Calculating Re-Projected Matrix")
      X <- LSI$svd$u %*% diag(LSI$svd$d) %*% t(V)
      out <- list(matSVD = matSVD, V = V, X = X)
    }
    else {
      out <- matSVD
    }
    out
  }, error = function(e) {
    errorList <- list(mat = mat, colSm = if (exists("colSm", 
                                                    inherits = FALSE)) colSm else "Error with colSm!", 
                      idf = if (exists("idf", inherits = FALSE)) idf else "Error with idf!", 
                      V = if (exists("V", inherits = FALSE)) V else "Error with V!", 
                      matSVD = if (exists("matSVD", inherits = FALSE)) matSVD else "Error with matSVD!")
    errorLog(e, fn = "projectLSI", info = "", errorList = errorList)
  })
  out2
}

#' @export
scale_dims<-function(x, scale_max = NULL){
  if(!is.null(scale_max)){
    row_z_scores(m=x, min=-scale_max, max = scale_max, limit = TRUE)
  }else{
    row_z_scores(m=x)
  }
}

#' @export
row_z_scores<-function(m = NULL, min = -2, max = 2, limit = FALSE){
  z <- sweep(m - Matrix::rowMeans(m), 1, matrixStats::rowSds(m),`/`)
  if(limit){
    z[z > max] <- max
    z[z < min] <- min
  }
  return(z)
}

#' @export
errorLog<-function(
  e = NULL,
  fn = NULL,
  info = NULL, 
  errorList = NULL,
  throwError = TRUE
){
  
  header <- "************************************************************"
  if(!is.null(errorList)){
    tryCatch({
      saveRDS(errorList, "Save-Error.rds")
      message("Saving a list of errors to Save-Error.rds")
    }, error = function(e){
      message("Error recording errorList")
    })
  }
  print(e)
  cat(sprintf("\n%s\n\n", header))
  if(throwError) stop("Exiting See Error Above")
}
