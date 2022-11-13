#' Compute a projection of a cell_data_set object into a lower dimensional
#' space with non-linear dimension reduction methods
#'
#' @description Monocle3 aims to learn how cells transition through a
#' biological program of gene expression changes in an experiment. Each cell
#' can be viewed as a point in a high-dimensional space, where each dimension
#' describes the expression of a different gene. Identifying the program of
#' gene expression changes is equivalent to learning a \emph{trajectory} that
#' the cells follow through this space. However, the more dimensions there are
#' in the analysis, the harder the trajectory is to learn. Fortunately, many
#' genes typically co-vary with one another, and so the dimensionality of the
#' data can be reduced with a wide variety of different algorithms. Monocle3
#' provides two different algorithms for dimensionality reduction via
#' \code{reduce_dimensions} (UMAP and tSNE). The function
#' \code{reduce_dimensions} is the second step in the trajectory building
#' process after \code{preprocess_cds}.
#'
#' UMAP is implemented from the package uwot.
#'
#' @param cds the cell_data_set upon which to perform this operation.
#' @param max_components the dimensionality of the reduced space. Default is 2.
#' @param reduction_method A character string specifying the algorithm to use
#'   for dimensionality reduction. Currently "UMAP", "tSNE", and "PCA" are
#'   supported.
#'@param num_dim Numeric indicating the number of prinicipal components to be 
#'    in downstream ordering.  Default value is NULL which will result in use 
#'    of all PCs
#' @param preprocess_method A string indicating the preprocessing method used
#'   on the data. Options are "PCA" and "LSI". Default is "LSI".
#' @param umap.metric A string indicating the distance metric to be used when
#'   calculating UMAP. Default is "cosine". See uwot package's
#'   \code{\link[umap]{umap}} for details.
#' @param umap.min_dist Numeric indicating the minimum distance to be passed to
#'   UMAP function. Default is 0.1.See uwot package's \code{\link[umap]{umap}}
#'   for details.
#' @param umap.n_neighbors Integer indicating the number of neighbors to use
#'   during kNN graph construction. Default is 15L. See uwot package's
#'   \code{\link[umap]{umap}} for details.
#' @param umap.fast_sgd Logical indicating whether to use fast SGD. Default is
#'   TRUE. See uwot package's \code{\link[umap]{umap}} for details.
#' @param umap.nn_method String indicating the nearest neighbor method to be
#'   used by UMAP. Default is "annoy". See uwot package's
#'   \code{\link[umap]{umap}} for details.
#' @param umap.save_model path to save umap model. Default NULL (don't save a model); See uwot package's \code{\link[umap]{umap}} for details.
#' @param cores Number of compute cores to use.
#' @param verbose Logical, whether to emit verbose output.
#' @param ... additional arguments to pass to the dimensionality reduction
#'   function.
#' @return an updated cell_data_set object
#' @references UMAP: McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation
#'   and Projection for Dimension Reduction, ArXiv e-prints 1802.03426, 2018
#' @references tSNE: Laurens van der Maaten and Geoffrey Hinton. Visualizing
#'   data using t-SNE. J. Mach. Learn. Res., 9(Nov):2579– 2605, 2008.
#' @references monocle3 - this function only differs from that found in monocle3 in \
#' that it allows for selection of num_dim
#' @export
reduce_dimension <- function(cds,
                             max_components=2,
                             reduction_method=c("UMAP", 'tSNE', 'PCA'),
                             preprocess_method=c("PCA", "LSI"),
                             umap.metric = "cosine",
                             umap.min_dist = 0.1,
                             umap.n_neighbors = 15L,
                             umap.fast_sgd = FALSE,
                             umap.nn_method = "annoy",
                             umap.save_model = NULL,
                             verbose=FALSE,
                             cores=1,
                             num_dim=NULL,
                             seed=2020,
                             ...){
  extra_arguments <- list(...)
  
  assertthat::assert_that(
    tryCatch(expr = ifelse(match.arg(reduction_method) == "",TRUE, TRUE),
             error = function(e) FALSE),
    msg = "reduction_method must be one of 'UMAP', 'PCA' or 'tSNE'")
  
  assertthat::assert_that(
    tryCatch(expr = ifelse(match.arg(preprocess_method) == "",TRUE, TRUE),
             error = function(e) FALSE),
    msg = "preprocess_method must be one of 'PCA' or 'LSI'")
  
  reduction_method <- match.arg(reduction_method)
  preprocess_method <- match.arg(preprocess_method)
  
  assertthat::assert_that(assertthat::is.count(max_components))
  
  assertthat::assert_that(!is.null(reducedDims(cds)[[preprocess_method]]),
                          msg = paste("Data has not been preprocessed with",
                                      "chosen method:", preprocess_method,
                                      "Please run preprocess_cds with",
                                      "method =", preprocess_method,
                                      "before running reduce_dimension."))
  if(reduction_method == "PCA") {
    assertthat::assert_that(preprocess_method == "PCA",
                            msg = paste("preprocess_method must be 'PCA' when",
                                        "reduction_method = 'PCA'"))
    assertthat::assert_that(!is.null(reducedDims(cds)[["PCA"]]),
                            msg = paste("When reduction_method = 'PCA', the",
                                        "cds must have been preprocessed for",
                                        "PCA. Please run preprocess_cds with",
                                        "method = 'PCA' before running",
                                        "reduce_dimension with",
                                        "reduction_method = 'PCA'."))
  }
  
  #ensure results from RNG sensitive algorithms are the same on all calls
  set.seed(seed)
  
  if (reduction_method=="UMAP" && (umap.fast_sgd == TRUE || cores > 1)){
    message(paste("Note: reduce_dimension will produce slightly different",
                  "output each time you run it unless you set",
                  "'umap.fast_sgd = FALSE' and 'cores = 1'"))
  }
  
  preprocess_mat <- reducedDims(cds)[[preprocess_method]]
  if(!is.null(num_dim)){
    preprocess_mat<-preprocess_mat[,1:num_dim]
  }else{
    num_dim<-ncol(preprocess_mat)
  }
  
  if(reduction_method == "PCA") {
    if (verbose) message("Returning preprocessed PCA matrix")
  } else if (reduction_method == "tSNE") {
    if (verbose) message("Reduce dimension by tSNE ...")
    
    tsne_res <- Rtsne::Rtsne(as.matrix(preprocess_mat), dims = max_components,
                             pca = F, check_duplicates=FALSE, ...)
    
    tsne_data <- tsne_res$Y[, 1:max_components]
    row.names(tsne_data) <- colnames(tsne_data)
    
    reducedDims(cds)$tSNE <- tsne_data
    
  } else if (reduction_method == c("UMAP")) {
    if (verbose)
      message("Running Uniform Manifold Approximation and Projection")
    if(is.null(umap.save_model))
      {
        umap.ret_model=F
        umap.ret_nn = F
    }else{
        umap.ret_model=T
        umap.ret_nn = T
      }
    umap_res = uwot::umap(as.matrix(preprocess_mat),
                          n_components = max_components,
                          metric = umap.metric,
                          min_dist = umap.min_dist,
                          n_neighbors = umap.n_neighbors,
                          fast_sgd = umap.fast_sgd,
                          n_threads=cores,
                          verbose=verbose,
                          nn_method = umap.nn_method,
                          ret_model = umap.ret_model,
                          ret_nn = umap.ret_nn,
                          ...)
    
    
    if(!umap.ret_model) {
      row.names(umap_res) <- colnames(cds)
      reducedDims(cds)$UMAP <- umap_res
    }else{
      reducedDims(cds)$UMAP <- umap_res$embedding
      model_file <- save_umap_model(umap_res, umap.save_model)
      cds@reduce_dim_aux <-SimpleList(UMAP=SimpleList(scale_info=umap_res$scale_info, model_file=model_file, num_dim=num_dim))
    }
  }
  
  ## Clear out any old graphs:
  cds@principal_graph_aux[[reduction_method]] <- NULL
  cds@principal_graph[[reduction_method]] <- NULL
  cds@clusters[[reduction_method]] <- NULL
  
  cds
}




#' Iterative LSI
#'
#' @description This function aims to both minimize batch effects and accentuate
#' cell type differences in a single cell experiment.  This function was implemented
#' using Monocle3 but takes inspiration from the Granja et. al. reference cited below which took inspiration from the fly ATAC paper. At it's heart
#' this function iterates through three main steps: 1) Using TFIDF transformation and SVD
#' to normalize data 2) Clustering this normalized data using leiden clustering in high dimensional
#' space and 3) identifying those features that are over-represented in the resulting clusters using
#' a simple counting method.  These three steps are repeated using features identified in step 3 to subset 
#' the normalization matrix in step 1 and repeating through the process.  TFIDF transformation is 
#' supplied in this package.  SVD is performed using the irilba package.  Leiden clustering is performed using
#' the monocle3 implementation and finally the counting per cluster is performed using the edgeR cpm function.  This
#' function takes as its input a cell_data_set and will iterate through n number of iterations.  The output of this function
#' is then appropriately input into dimensionality reduction methods such as UMAP or tSNE.  The number of iterations
#' is set by the number of resolution parameters specified.
#' 
#' @param cds the cell_data_set upon which to perform this operation.
#' @param num_dim Numeric indicating the number of prinicipal components to be 
#'    in downstream ordering.  Default value is NULL which will result in use 
#'    of all PCs
#' @param resolution vector of resolution values for leiden clustering
#' @param binarize boolean whether to binarize data prior to TFIDF transformation
#' @param scale_to numeric value to scale data
#' @param seed numeric seed
#' @param return_iterations boolean whether to return iterations; funciton will then output a list contianing the final cds and 
#' all SVD matrices, clusters and features used in each iteration
#' @param num_features number of features to use for dimensionality reduction (default 3000).  To use different numbers
#' of features for different iterations, supply a vector that is the same length as the resolution vector.
#' @param exclude_features character vector of features (rownames of assay(cds))
#' @return an updated cell_data_set object with a reduced dimension LSI object and clusters object
#' @references Granja, J. M.et al. (2019). Single-cell multiomic analysis identifies regulatory programs in mixed-phenotype 
#' acute leukemia. Nature Biotechnology, 37(12), 1458–1465.
#' @references UMAP: McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation
#'   and Projection for Dimension Reduction, ArXiv e-prints 1802.03426, 2018
#' @references tSNE: Laurens van der Maaten and Geoffrey Hinton. Visualizing
#'   data using t-SNE. J. Mach. Learn. Res., 9(Nov):2579– 2605, 2008.
#' @references Cusanovich, D. A., Reddington, J. P., Garfield, D. A., Daza, R. M., Aghamirzaie, D., Marco-Ferreres, R., et al. (2018). The 
#'   cis-regulatory dynamics of embryonic development at single-cell resolution. Nature, 555(7697), 538–542.
#' @export
iterative_LSI <- function(cds,
                          num_dim=25, starting_features = NULL,
                          resolution=c(1e-4, 3e-4, 5e-4), do_tf_idf=T,
                          num_features=c(3000,3000,3000), exclude_features=NULL,
                          binarize=FALSE, scale=T, log_transform=T,
                          LSI_method=1, partition_qval = 0.05, 
                          seed=2020, scale_to = 10000, leiden_k=20, leiden_weight=FALSE, leiden_iter=1, verbose=F, return_iterations=F,
                          ...){
  extra_arguments <- list(...)
  if(!is.null(starting_features)){
    if(length(num_features)!=length(resolution)){
      num_features<-c(length(starting_features), num_features)
    }
  }
  if(length(num_features)!=length(resolution)){
    message("Numbers of elements for resolution and num_features do not match.  Will use num_features[1]...")
    num_features<-rep(num_features, length(resolution))
  }
  if(!is.null(exclude_features)){
    mat<-assay(cds)
    mat<-mat[!rownames(mat) %in% exclude_features,]
  }else{
    mat<-assay(cds)
  }
  
  original_features<-rownames(mat)
  set.seed(seed)
  if(binarize){
    message("Binarizing...")
    mat@x[mat@x > 0] <- 1
  }
  outlist<-list()
  if(scale){
    matNorm <- t(t(mat)/Matrix::colSums(mat)) * scale_to
  }else{
    matNorm<-mat
  }
  
  if(log_transform){
    matNorm@x <- log2(matNorm@x + 1)
  }

  message("Performing LSI/SDF for iteration 1....")
  if(!is.null(starting_features)){
    if(!all(starting_features %in% rownames(mat))){stop("Not all starting features found in data")}
    f_idx<-which(starting_features %in% rownames(mat))
  }else{
    f_idx<-head(order(sparseRowVariances(matNorm),decreasing=TRUE), num_features[1])
  }
  if(do_tf_idf){
    tf<-tf_idf_transform(mat[f_idx,], method = LSI_method)
    row_sums<-Matrix::rowSums(mat[f_idx,])
    tf@x[is.na(tf@x)] <- 0
  }else{
    tf<-mat[f_idx,]
    row_sums<-Matrix::rowSums(mat[f_idx,])
  }
  
  svd_list<-svd_lsi(tf, num_dim, mat_only=FALSE)
  cluster_result <- monocle3:::leiden_clustering(data = svd_list$matSVD, 
                                      pd = colData(cds), k = leiden_k, weight = leiden_weight, num_iter = leiden_iter, 
                                      resolution_parameter = resolution[1], random_seed = seed, 
                                      verbose = verbose, ...)
  clusters <- factor(igraph::membership(cluster_result$optim_res))
  clusterMat <- edgeR::cpm(groupSums(mat, clusters, sparse = TRUE), log=TRUE, prior.count = 3)
  if(length(resolution)==1){
    reducedDims(cds)[["LSI"]]<-svd_list$matSVD
    irlba_rotation = svd_list$svd$u
    row.names(irlba_rotation) = original_features[f_idx]
    iLSI<-SimpleList(svd=svd_list$svd, features=original_features[f_idx], 
                     row_sums = row_sums, seed=seed, binarize=binarize, 
                     scale_to=scale_to, num_dim=num_dim, resolution=resolution, 
                     granges=rowRanges(cds)[f_idx], LSI_method=LSI_method, outliers=NULL)
    pp_aux <- SimpleList(iLSI=iLSI, gene_loadings=irlba_rotation, features=original_features[f_idx])
    cds@preprocess_aux <- pp_aux
    if (length(unique(cluster_result$optim_res$membership)) > 
        1) {
      cluster_graph_res <- monocle3:::compute_partitions(cluster_result$g, 
                                              cluster_result$optim_res, partition_qval, verbose)
      partitions <- igraph::components(cluster_graph_res$cluster_g)$membership[cluster_result$optim_res$membership]
      partitions <- as.factor(partitions)
    }
    else {
      partitions <- rep(1, nrow(colData(cds)))
    }
    names(partitions) <- row.names(rownames(colData(cds)))
    cds@clusters[["LSI"]] <- list(cluster_result = cluster_result, 
                                             partitions = partitions, clusters = clusters)
    if(return_iterations){
      outlist[["iteration_1"]]=list(matSVD=svd_list$matSVD, features=original_features[f_idx], clusters=clusters)
      return(list(cds=cds, iterationlist=outlist))
    }else{
      return(cds)
    }
  }
  for(iteration in 2:length(resolution)){
    message("Performing LSI/SDF for iteration ", iteration, "....")
    f_idx<-head(order(rowVars(clusterMat), decreasing=TRUE), num_features[iteration])
    tf<-tf_idf_transform(mat[f_idx,], method = LSI_method)
    tf@x[is.na(tf@x)] <- 0
    row_sums<-Matrix::rowSums(mat[f_idx,])
    svd_list<-svd_lsi(tf, num_dim, mat_only=FALSE)
    cluster_result <- monocle3:::leiden_clustering(data = svd_list$matSVD, 
                                                   pd = colData(cds), k = leiden_k, weight = leiden_weight, num_iter = leiden_iter, 
                                                   resolution_parameter = resolution[iteration], random_seed = seed, 
                                                   verbose = verbose, ...)
    clusters <- factor(igraph::membership(cluster_result$optim_res))
    clusterMat <- edgeR::cpm(groupSums(mat, clusters, sparse = TRUE), log=TRUE, prior.count = 3)
    if(iteration!=length(resolution)){
      if(return_iterations){
        it_count<-paste0("iteration_", iteration)
        outlist[[it_count]]=list(matSVD=svd_list$matSVD, features=original_features[f_idx], clusters=clusters)
        return(list(cds=cds, iterationlist=outlist))
      }
      next
    }else{
      reducedDims(cds)[["LSI"]]<-svd_list$matSVD
      irlba_rotation = svd_list$svd$u
      row.names(irlba_rotation) = original_features[f_idx]
      iLSI<-SimpleList(svd=svd_list$svd, features=original_features[f_idx], 
                       row_sums = row_sums, seed=seed, binarize=binarize, 
                       scale_to=scale_to, num_dim=num_dim, resolution=resolution, 
                       granges=rowRanges(cds)[f_idx], LSI_method=LSI_method, outliers=NULL)
      pp_aux <- SimpleList(iLSI=iLSI, gene_loadings=irlba_rotation, features=original_features[f_idx])
      cds@preprocess_aux <- pp_aux
      if (length(unique(cluster_result$optim_res$membership)) > 
          1) {
        cluster_graph_res <- monocle3:::compute_partitions(cluster_result$g, 
                                                           cluster_result$optim_res, partition_qval, verbose)
        partitions <- igraph::components(cluster_graph_res$cluster_g)$membership[cluster_result$optim_res$membership]
        partitions <- as.factor(partitions)
      }
      else {
        partitions <- rep(1, nrow(colData(cds)))
      }
      names(partitions) <- row.names(rownames(colData(cds)))
      cds@clusters[["LSI"]] <- list(cluster_result = cluster_result, 
                                    partitions = partitions, clusters = clusters)
      if(return_iterations){
        it_count<-paste0("iteration_", iteration)
        outlist[[it_count]]=list(matSVD=svd_list$matSVD, features=original_features[f_idx], clusters=clusters)
        return(list(cds=cds, iterationlist=outlist))
      }else{
        return(cds)
      }
    }
    # if(!is.null(update_clusters_in_embedding)){
    #   cds@clusters[[update_clusters_in_embedding]]$clusters[colnames(exprs(cds))]<-as.character(cds@clusters[["LSI"]]$optim_res$membership)
    # }
  }
}

#' Helper function for summing sparse matrix groups
#' @references Granja, J. M.et al. (2019). Single-cell multiomic analysis identifies regulatory programs in mixed-phenotype 
#' acute leukemia. Nature Biotechnology, 37(12), 1458–1465.
#' @export
groupSums <- function (mat, groups = NULL, na.rm = TRUE, sparse = FALSE){
  stopifnot(!is.null(groups))
  stopifnot(length(groups) == ncol(mat))
  gm <- lapply(unique(groups), function(x) {
    if (sparse) {
      Matrix::rowSums(mat[, which(groups == x), drop = F], na.rm = na.rm)
    }
    else {
      rowSums(mat[, which(groups == x), drop = F], na.rm = na.rm)
    }
  }) %>% Reduce("cbind", .)
  colnames(gm) <- unique(groups)
  return(gm)
}


#' Helper function for summing sparse matrix groups
#' @references Granja, J. M.et al. (2019). Single-cell multiomic analysis identifies regulatory programs in mixed-phenotype 
#' acute leukemia. Nature Biotechnology, 37(12), 1458–1465.
#' @export
sparseRowVariances <- function (m){
  rM <- Matrix::rowMeans(m)
  rV <- computeSparseRowVariances(m@i + 1, m@x, rM, ncol(m))
  return(rV)
}
# 
# 
# #Sparse Variances Rcpp
# Rcpp::sourceCpp(code='
#   #include <Rcpp.h>
#   using namespace Rcpp;
#   using namespace std;
#   // [[Rcpp::export]]
#   Rcpp::NumericVector computeSparseRowVariances(IntegerVector j, NumericVector val, NumericVector rm, int n) {
#     const int nv = j.size();
#     const int nm = rm.size();
#     Rcpp::NumericVector rv(nm);
#     Rcpp::NumericVector rit(nm);
#     int current;
#     // Calculate RowVars Initial
#     for (int i = 0; i < nv; ++i) {
#       current = j(i) - 1;
#       rv(current) = rv(current) + (val(i) - rm(current)) * (val(i) - rm(current));
#       rit(current) = rit(current) + 1;
#     }
#     // Calculate Remainder Variance
#     for (int i = 0; i < nm; ++i) {
#       rv(i) = rv(i) + (n - rit(i))*rm(i)*rm(i);
#     }
#     rv = rv / (n - 1);
#     return(rv);
#   }'
# )

#New save_umap
#' @export
save_umap_model <- function(model, file){
  if (!is.null(model$nn_index$ann)) {
    save_umap_model_new(model = model, file = file)
  }
  else {
    save_umap_model_depracated(model = model, file = file) #backwards to previous version
  }
}

#' @export
save_umap_model_new<-function(model, file){
  file <- file.path(normalizePath(dirname(file)), basename(file))
  wd <- getwd()
  mod_dir <- tempfile(pattern = "dir")
  dir.create(mod_dir)
  uwot_dir <- file.path(mod_dir, "uwot")
  dir.create(uwot_dir)
  model_tmpfname <- file.path(uwot_dir, "model")
  saveRDS(model, file = model_tmpfname)
  metrics <- names(model$metric)
  n_metrics <- length(metrics)
  for (i in seq_len(n_metrics)) {
    nn_tmpfname <- file.path(uwot_dir, paste0("nn", i))
    if (n_metrics == 1) {
      model$nn_index$ann$save(nn_tmpfname)
      model$nn_index$ann$unload()
      model$nn_index$ann$load(nn_tmpfname)
    }
    else {
      model$nn_index[[i]]$ann$save(nn_tmpfname)
      model$nn_index[[i]]$ann$unload()
      model$nn_index[[i]]$ann$load(nn_tmpfname)
    }
  }
  setwd(mod_dir)
  system2("tar", "-cvf uwot.tar uwot", stdout = NULL, stderr = NULL)
  o <- file_rename("uwot.tar", file)
  setwd(wd)
  if (file.exists(mod_dir)) {
    unlink(mod_dir, recursive = TRUE)
  }
  return(o)
}



#' @export
save_umap_model_depracated<-function(model, file){
  file <- file.path(normalizePath(dirname(file)), basename(file))
  wd <- getwd()
  mod_dir <- tempfile(pattern = "dir")
  dir.create(mod_dir)
  uwot_dir <- file.path(mod_dir, "uwot")
  dir.create(uwot_dir)
  model_tmpfname <- file.path(uwot_dir, "model")
  saveRDS(model, file = model_tmpfname)
  metrics <- names(model$metric)
  n_metrics <- length(metrics)
  for (i in seq_len(n_metrics)) {
    nn_tmpfname <- file.path(uwot_dir, paste0("nn", i))
    if (n_metrics == 1) {
      model$nn_index$save(nn_tmpfname)
      model$nn_index$unload()
      model$nn_index$load(nn_tmpfname)
    }
    else {
      model$nn_index[[i]]$save(nn_tmpfname)
      model$nn_index[[i]]$unload()
      model$nn_index[[i]]$load(nn_tmpfname)
    }
  }
  setwd(mod_dir)
  system2("tar", "-cvf uwot.tar uwot", stdout = NULL, stderr = NULL)
  o <- file_rename("uwot.tar", file)
  setwd(wd)
  if (file.exists(mod_dir)) {
    unlink(mod_dir, recursive = TRUE)
  }
  return(o)
}

#' @export
file_rename<-function(from = NULL, to = NULL){
  
  if(!file.exists(from)){
    stop("Input file does not exist!")
  }
  
  tryCatch({
    
    o <- .suppressAll(file.rename(from, to))
    
    if(!o){
      stop("retry with mv")
    }
    
  }, error = function(x){
    
    tryCatch({
      
      system(paste0("mv '", from, "' '", to, "'"))
      
      return(to)
      
    }, error = function(y){
      
      stop("File Moving/Renaming Failed!")
      
    })
    
  })
  
}


#New load_umap_model
#' @export
load_umap_model <- function(file, num_dim = NULL){
    tryCatch({
      load_umap_model_new(file = file, num_dim = num_dim) 
    }, error = function(x){
      load_umap_model_depracated(file = file, num_dim = num_dim)
    })
}

#' @export
load_umap_model_new<-function(file, num_dim = NULL){
  model <- NULL
  tryCatch({
    mod_dir <- tempfile(pattern = "dir")
    dir.create(mod_dir)
    utils::untar(file, exdir = mod_dir)
    model_fname <- file.path(mod_dir, "uwot/model")
    if (!file.exists(model_fname)) {
      stop("Can't find model in ", file)
    }
    model <- readRDS(file = model_fname)
    metrics <- names(model$metric)
    n_metrics <- length(metrics)
    for (i in seq_len(n_metrics)){
      nn_fname <- file.path(mod_dir, paste0("uwot/nn", i))
      if (!file.exists(nn_fname)) {
        stop("Can't find nearest neighbor index ", nn_fname, " in ", file)
      }
      metric <- metrics[[i]]
      if(length(model$metric[[i]]) == 0){
        if(!is.null(num_dim)){
          num_dim2 <- num_dim
        }else{
          num_dim2 <- length(model$metric[[i]])
        }
      }
      if(!is.null(num_dim)){
        num_dim2 <- num_dim
      }
      ann <- uwot:::create_ann(metric, ndim = num_dim2)
      ann$load(nn_fname)
      if (n_metrics == 1) {
        model$nn_index$ann <- ann
      }else{
        model$nn_index[[i]]$ann <- ann
      }
    }
  }, finally = {
    if (file.exists(mod_dir)) {
      unlink(mod_dir, recursive = TRUE)
    }
  })
  model 
}

#' @export
load_umap_model_depracated<-function(file, num_dim = NULL){
  model <- NULL
  tryCatch({
    mod_dir <- tempfile(pattern = "dir")
    dir.create(mod_dir)
    utils::untar(file, exdir = mod_dir)
    model_fname <- file.path(mod_dir, "uwot/model")
    if (!file.exists(model_fname)) {
      stop("Can't find model in ", file)
    }
    model <- readRDS(file = model_fname)
    metrics <- names(model$metric)
    n_metrics <- length(metrics)
    for (i in seq_len(n_metrics)){
      nn_fname <- file.path(mod_dir, paste0("uwot/nn", i))
      if (!file.exists(nn_fname)) {
        stop("Can't find nearest neighbor index ", nn_fname, " in ", file)
      }
      metric <- metrics[[i]]
      if(length(model$metric[[i]]) == 0){
        if(!is.null(num_dim)){
          num_dim2 <- num_dim
        }else{
          num_dim2 <- length(model$metric[[i]])
        }
      }
      if(!is.null(num_dim)){
        num_dim2 <- num_dim
      }
      ann <- uwot:::create_ann(metric, ndim = num_dim2)
      ann$load(nn_fname)
      if (n_metrics == 1) {
        model$nn_index <- ann
      }else{
        model$nn_index[[i]] <- ann
      }
    }
  }, finally = {
    if (file.exists(mod_dir)) {
      unlink(mod_dir, recursive = TRUE)
    }
  })
  model 
}
#' Cluster LSI
#'
#' @description This function extracts clustering from the last iteration of LSI (see \code{iterativeLSI})
#' cell type differences in a single cell experiment.  This function uses the leiden clustering as implemented in monocle3, then finds
#' less granular clusters in the data using partitions (monocle3) using the reduced dimension LSI input from the last iteration of LSI used.
#' 
#' @param cds the cell_data_set upon which to perform this operation.
#' @param k Nnteger number of nearest neighbors to use when creating the k nearest neighbor graph for Leiden clustering. k is 
#' related to the resolution of the clustering result, a bigger k will result in lower resolution and vice versa. Default is 20.
#' @param weight A logical argument to determine whether or not to use Jaccard coefficients for two nearest neighbors (based on the 
#' overlapping of their kNN) as the weight used for Louvain clustering. Default is FALSE
#' @param binarize boolean whether to binarize data prior to TFIDF transformation
#' @param num_iter 	Integer number of iterations used for Leiden clustering. The clustering result giving the largest modularity 
#' score will be used as the final clustering result. Default is 1. Note that if num_iter is greater than 1, the random_seed argument will be ignored for the louvain method.
#' @param resolution Parameter that controls the resolution of clustering. If NULL (Default), the parameter is determined automatically.
#' @param random_seed The seed used by the random number generator in louvain-igraph package. This argument will be ignored if num_iter is larger than 1.
#' @param verbose A logic flag to determine whether or not we should print the run details.
#' @param partition_qval Numeric, the q-value cutoff to determine when to partition. Default is 0.05.
#' @references Granja, J. M.et al. (2019). Single-cell multiomic analysis identifies regulatory programs in mixed-phenotype 
#' acute leukemia. Nature Biotechnology, 37(12), 1458–1465.
#' @references Cusanovich, D. A., Reddington, J. P., Garfield, D. A., Daza, R. M., Aghamirzaie, D., Marco-Ferreres, R., et al. (2018). The 
#'   cis-regulatory dynamics of embryonic development at single-cell resolution. Nature, 555(7697), 538–542.
#' @references  Vincent D. Blondel, Jean-Loup Guillaume, Renaud Lambiotte, Etienne Lefebvre: Fast unfolding of communities in large 
#' networks. J. Stat. Mech. (2008) P10008
#' @references V. A. Traag and L. Waltman and N. J. van Eck: From Louvain to Leiden: guaranteeing well-connected communities. 
#' Scientific Reports, 9(1) (2019). doi: 10.1038/s41598-019-41695-z.
#' @references Jacob H. Levine and et. al. Data-Driven Phenotypic Dissection of AML Reveals Progenitor-like Cells that 
#' Correlate with Prognosis. Cell, 2015.
#' @export
cluster_LSI<-function(cds, 
                      k = 20, 
                      weight = F, 
                      num_iter = 1, 
                      resolution_parameter = NULL, 
                      random_seed = 2020, 
                      verbose = T, 
                      partition_q_value = 0.05)
{
  cluster_result<-monocle3:::leiden_clustering(data = as.matrix(reducedDims(cds)$LSI), 
                                               pd = colData(cds), k = k, weight = weight, 
                                               num_iter = num_iter, 
                                               resolution_parameter = resolution_parameter, 
                                               random_seed = random_seed, verbose = verbose)
  cluster_graph_res<-monocle3:::compute_partitions(cluster_result$g, 
                                                   cluster_result$optim_res, partition_q_value, verbose=t)
  partitions <- as.factor(igraph::components(cluster_graph_res$cluster_g)$membership[cluster_result$optim_res$membership])
  clusters <- factor(igraph::membership(cluster_result$optim_res))
  cds@clusters[["UMAP"]] <- list(cluster_result = cluster_result, 
                                 partitions = partitions, clusters = clusters)
  cds
}



