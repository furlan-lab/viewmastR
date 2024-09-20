setClass("training_item", slots=c(data="numeric", target="numeric"))
setClass("training_set", slots=c(name="character", items="list", labels="character", features="character"))

dimension_check <- function (obj){
  nm <-deparse(substitute(obj))
  if(is.null(dim(obj))){
    if(class(obj)=="list"){
      llen <- sapply(obj, length)
      if(!is.null(var(llen))){
        if (var(llen)==0){
          paste0(nm, "-dim1: ", llen[1], " non-ragged")
        } else {
          paste0(nm, "-dim1: ", mean(llen), " (mean) ragged")
        }
      } else {
        message("not numeric")
      }
    } else {
      paste0(nm, "-dim1: ", length(obj))
    }
  } else {
    nd <- length(dim(obj))
    dv <- paste0("-dim", 1:nd, ": ")
    paste0(nm, paste0(dv, dim(obj)))
  }
}

#' Run viewmastR using the new Rust implementation
#' 
#' This function runs viewmastR using the new Rust implementation to generate various forms of training and test data from the given query and reference cell data sets.
#' 
#' @param query_cds A cell data set (cds) to query.
#' @param ref_cds A reference cell data set (cds).
#' @param ref_celldata_col The column in the reference cell data set containing cell data.
#' @param query_celldata_col The column in the query cell data set containing cell data. Defaults to "viewmastR_pred" if not provided.
#' @param FUNC The machine learning model to use (multinomial linear regression - 'mlr' or multilayer neural network - 'nn'). Default is "mlr".
#' @param norm_method The normalization method to use. Options are "log", "binary", "size_only", or "none".
#' @param selected_genes A vector of pre-selected genes for analysis. If NULL, common features between reference and query are used.
#' @param train_frac The fraction of data to use for training. Default is 0.8.
#' @param tf_idf Boolean indicating whether to perform TF-IDF transformation. Default is FALSE.
#' @param scale Boolean indicating whether to scale the data. Default is FALSE.
#' @param hidden_layers A vector specifying the number of neurons in hidden layers for neural network models. Default is c(500, 100).  To use one hidden layer supply a vector of length 1.
#' @param learning_rate The learning rate for training the model. Default is 1e-3.
#' @param max_epochs The maximum number of epochs for training the model. Default is 10.
#' @param LSImethod The method for Latent Semantic Indexing. Default is 1.
#' @param verbose Boolean indicating whether to display verbose output. Default is TRUE.
#' @param dir The directory to save output files. Default is "/tmp/sc_local".
#' @param return_type The type of output to return. Options are "object" or "list". Default is "object".
#' @param debug Run in debug mode
#' @param ... Additional arguments.
#' 
#' @return If return_type is "object", returns the modified query cell data set or seurat object. If return_type is "list", returns a list containing the query cell data set and training output.
#' 
#' @export

viewmastR <-function(query_cds, 
                     ref_cds, 
                     ref_celldata_col, 
                     query_celldata_col=NULL, 
                     FUNC=c("mlr", "nn", 'nb'),
                     norm_method=c("log", "binary", "size_only", "none"),
                     selected_genes=NULL,
                     train_frac = 0.8,
                     tf_idf=F,
                     scale=F,
                     hidden_layers = c(500,100),
                     learning_rate = 1e-3,
                     max_epochs = 10,
                     LSImethod=1,
                     verbose = T,
                     backend = c("wgpu", "nd", "candle"),
                     threshold = NULL,
                     keras_model = NULL, 
                     dir = "/tmp/sc_local",
                     return_probs = F,
                     return_type = c("object", "list"), 
                     debug = F, ...){
  return_type <- match.arg(arg = NULL, return_type)
  backend <- match.arg(arg = NULL, backend)
  FUNC <-match.arg(arg = NULL, FUNC)
  # if(return_type=="object" && return_probs == T) {stop("Cannot return both probabilities and a single cell object; rerun changing return_type to list if probabilities are sought.")}
  if(!length(hidden_layers) %in% c(1,2)){stop("Only 1 or 2 hidden layers are allowed.")}
  if (debug){
    message("Dimension check:")
    message(paste0("\t", dimension_check(query_cds)))
    message(paste0("\t", dimension_check(ref_cds)))
    message(paste0("\t", dimension_check(selected_genes)))
  }
  training_list<-setup_training(query_cds, 
                                ref_cds, 
                                tf_idf=tf_idf,
                                scale=scale,
                                LSImethod=LSImethod,
                                ref_celldata_col = ref_celldata_col, 
                                selected_genes = selected_genes, 
                                verbose=verbose, 
                                return_type = "list")
  # use_sparse = F)
  if (debug){
    message("Dimension check:")
    message(paste0("\t", dimension_check(training_list[["train"]][[1]])))
    message(paste0("\t", dimension_check(training_list[["test"]][[1]])))
    message(paste0("\t", dimension_check(training_list[["query"]][[1]])))
    message(paste0("\t", dimension_check(training_list[["labels"]])))
  }
  if(!file.exists(dir)){
    dir.create(dir)
  }
  if(FUNC=="mlr"){
    export_list<-process_learning_obj_mlr(train = training_list[["train"]], 
                                          test = training_list[["test"]], 
                                          query = training_list[["query"]], 
                                          labels = training_list[["labels"]], 
                                          learning_rate = learning_rate, num_epochs = max_epochs, 
                                          directory = dir, verbose = verbose, backend = backend)
  } 
  if(FUNC=="nn"){
    export_list<-process_learning_obj_ann(train = training_list[["train"]], 
                                          test = training_list[["test"]], 
                                          query = training_list[["query"]], 
                                          labels = training_list[["labels"]],
                                          hidden_size = hidden_layers,
                                          learning_rate = learning_rate, num_epochs = max_epochs, 
                                          directory = dir, verbose = verbose, backend = backend)
  }
  if(FUNC=="nb"){
    export_list<-process_learning_obj_nb(train = training_list[["train"]], 
                                          test = training_list[["test"]], 
                                          query = training_list[["query"]])
  }
  
  if(is.null(query_celldata_col)){
    query_celldata_col<-"viewmastR_pred"
  }
  
  log_odds = unlist(export_list$probs[[1]])
  if(length(log_odds) == dim(query_cds)[2]*length(training_list[["labels"]])){
    log_odds = matrix(log_odds, ncol = dim(query_cds)[2])
    log_odds = t(log_odds)
    colnames(log_odds) <- paste0("prob_", training_list[["labels"]])
  } else {
    stop("Error in log odds dimensions of function output")
  }
  export_list$probs = plogis(log_odds)
  query_cds[[query_celldata_col]]<-training_list[["labels"]][apply(log_odds, 1, which.max)]
  if(return_probs){
    query_cds@meta.data <- cbind(query_cds@meta.data, export_list$probs)
  }
  if (return_type=="object") {
    query_cds
  } else {
    list(object=query_cds, training_output = export_list)
  }
}


#' Setup training datasets
#' 
#' This function sets up training datasets for use in machine learning models.
#' 
#' @param query_cds A cell data set (cds) to query.
#' @param ref_cds A reference cell data set (cds).
#' @param ref_celldata_col The column in the reference cell data set containing cell data.
#' @param norm_method The normalization method to use. Options are "log", "binary", "size_only", or "none".
#' @param selected_genes A vector of pre-selected genes for analysis.
#' @param train_frac The fraction of data to use for training. 
#' @param tf_idf Boolean indicating whether to perform TF-IDF transformation.
#' @param scale Boolean indicating whether to scale the data.
#' @param LSImethod The method for Latent Semantic Indexing.
#' @param verbose Boolean indicating whether to display verbose output.
#' @param addbias Boolean indicating whether to add bias.
#' @param return_type The type of output to return. Options are "list", "matrix", or "S4obj".
#' @param ... Additional arguments.
#' 
#' @return A list, matrix, or S4 object containing the training datasets.
#' 
#' @importFrom Matrix colSums
#' @export
#' @keywords internal

setup_training <-function(query_cds, 
                          ref_cds, 
                          ref_celldata_col, 
                          norm_method=c("log", "binary", "size_only", "none"),
                          selected_genes=NULL,
                          train_frac = 0.8,
                          tf_idf=F,
                          scale=F,
                          LSImethod=1,
                          verbose = T,
                          addbias = F,
                          # use_sparse = F, 
                          return_type = c("list", "matrix", "S4obj"),
                          debug = F,
                          ...){
  
  if(verbose){
    message("Checking arguments and input")
  }
  if (debug){
    message("Dimension check:")
    message(paste0("\t", dimension_check(training_list[["train"]][[1]])))
    message(paste0("\t", dimension_check(training_list[["test"]][[1]])))
    message(paste0("\t", dimension_check(training_list[["query"]][[1]])))
    message(paste0("\t", dimension_check(training_list[["labels"]])))
  }
  #capture args for specific fxns
  argg <- c(as.list(environment()), list(...))
  layers=F
  
  #deal with conflicting and other args
  if(tf_idf & scale){
    warning("Both tf_idf and scale selected. Cannot do this as they are both scaling methods. Using tf_idf alone")
    scale<-F
  }
  norm_method=match.arg(norm_method)
  
  return_type <- match.arg(return_type)
  
  #get class of object
  if(class(query_cds) != class(ref_cds)){stop("input objects must be of the same class")}
  software<-NULL
  if(class(query_cds)=="Seurat"){
    software<-"seurat"
    labf<-as.factor(ref_cds@meta.data[[ref_celldata_col]])
  }
  if(class(query_cds)=="cell_data_set"){
    software<-"monocle3"
    labf<-as.factor(colData(ref_cds)[[ref_celldata_col]])
  }
  if(is.null(software)){stop("software not found for input objects")}
  
  
  #find common features
  if(verbose){
    message("Finding common features between reference and query")
  }
  common_list<-common_features(list(ref_cds, query_cds))
  names(common_list)<-c("ref", "query")
  rm(ref_cds)
  gc()
  
  if(is.null(selected_genes)){
    selected_common<-rownames(common_list[['query']])
    selected_common<-selected_commmon[selected_common %in% rownames(common_list[['ref']])]
  }else{
    if(verbose){
      message("Subsetting by pre-selected features")
    }
    selected_common<-selected_genes
    selected_common<-selected_common[selected_common %in% rownames(common_list[['query']])]
    selected_common<-selected_common[selected_common %in% rownames(common_list[['ref']])]
  }
  
  #make final X and query normalizing along the way
  # #no tf_idf
  if(norm_method!="none"){
    if(verbose){
      message("Calculated normalized counts")
    }
    query<-get_norm_counts(common_list[['query']], norm_method = norm_method)[selected_common,]
    X<-get_norm_counts(common_list[['ref']], norm_method = norm_method)[rownames(query),]
  }else{
    query<-get_norm_counts(common_list[['query']], )[selected_common,]
    X<-get_norm_counts(common_list[['ref']], )[rownames(query),]
  }
  rm(common_list)
  gc()
  
  #performe scaling methods
  if(tf_idf){
    if(verbose){
      message("Performing TF-IDF")
    }
    X<-tf_idf_transform(X, LSImethod)
    query<-tf_idf_transform(query, LSImethod)
  }else{
    if(scale){
      X<-scale(X)
      query <-scale(query)
    }
  }
  
  #densifying adding bias
  # if(!use_sparse){
  #   if(verbose){
  #     message("Converting to dense matrix :(")
  #   }
  #   if(addbias){
  #     if(verbose){
  #       message("Adding bias")
  #     }
  #     X <-rbind(rep(1, ncol(Xtrain)), X)
  #     query <-rbind(rep(1, ncol(query)), query)
  #   }
  #   X<-as.matrix(X)
  #   query<-as.matrix(query)
  # } else {
  #   if(addbias){
  #     if(verbose){
  #       message("Adding bias")
  #     }
  #     X <-rbind(rep(1, ncol(X)), X)
  #     query <-rbind(rep(1, ncol(query)), query)
  #   }
  #   X<-as(X, "RsparseMatrix")
  #   query<-as(query, "RsparseMatrix")
  # }
  
  if(verbose){
    message("Converting to dense matrix :(")
  }
  if(addbias){
    if(verbose){
      message("Adding bias")
    }
    X <-rbind(rep(1, ncol(Xtrain)), X)
    query <-rbind(rep(1, ncol(query)), query)
  }
  X<-as.matrix(X)
  query<-as.matrix(query)
  gc()
  
  
  #prep Y and deal with rownames
  Ylab<-as.numeric(labf)-1
  labels<-levels(labf)
  Y<-matrix(model.matrix(~0+labf), ncol = length(labels))
  colnames(Y)<-NULL
  features<-rownames(X)
  rownames(X)<-NULL
  colnames(X)<-NULL
  
  #create test/train indices
  train_idx<-sample(1:dim(X)[2], round(train_frac*dim(X)[2]))
  test_idx<-which(!1:dim(X)[2] %in% train_idx)
  # if(return_type == "matrices" && !use_sparse){
  if(return_type == "matrix"){
    return(list(Xtrain_data = t(X[,train_idx]), 
                Xtest_data = t(X[,test_idx]), 
                Ytrain_label = Y[train_idx,], 
                Ytest_label = Y[test_idx,],
                query = t(query),
                label_text = labels,
                features = features))
    # } else if(return_type == "matrices" && use_sparse){
    #   return(list(Xtrain_data = t_shallow(X[,train_idx]), 
    #               Xtest_data = t_shallow(X[,test_idx]), 
    #               Ytrain_label = Y[train_idx,], 
    #               Ytest_label = Y[test_idx,],
    #               query = t_shallow(query),
    #               label_text = labels,
    #               features = features))
  } else if(return_type == "list") {
    # } else if(return_type == "list" && !use_sparse) {
    return(list(
      train = lapply(train_idx, function(idx){
        list(data= t(X[,idx])[1,], target = Ylab[idx])
      }),
      test = lapply(test_idx, function(idx){
        list(data= t(X[,idx])[1,], target = Ylab[idx])
      }),
      query = lapply(1:dim(query)[2], function(idx){
        list(data = t(query[,idx])[1,])
      }),
      labels = labels,
      features = features))
    # } else if(return_type == "list" && use_sparse) {
    #   return(list(
    #     train = lapply(train_idx, function(idx){
    #       slice <- X[,idx]
    #       indices <- which(!slice == 0)
    #       list(values= slice[ indices] , indices=indices,
    #            target = Ylab[idx])
    #     }),
    #     test = lapply(test_idx, function(idx){
    #       slice <- X[,idx]
    #       indices <- which(!slice == 0)
    #       list(values= slice[ indices] , indices=indices,
    #            target = Ylab[idx])
    #     }),
    #     query = lapply(1:dim(query)[2], function(idx){
    #       slice <- query[,idx]
    #       indices <- which(!slice == 0)
    #       list(values= slice[ indices] , indices=indices)
    #     }),
    #     labels = labels,
    #     features = features))
  } else if (return_type == "S4obj") {
    # setClass("training_item", slots=c(data="numeric", target="numeric"))
    # setClass("training_set", slots=c(name="character", items="list", labels="character"))
    training_set<-new("training_set", 
                      name="train", 
                      items=lapply(train_idx, function(idx){
                        new("training_item", data = t(X[,idx])[1,], target = Ylab[idx])
                      }),
                      labels=labels,
                      features=features)
    test_set<-new("training_set", 
                  name="test", 
                  items=lapply(test_idx, function(idx){
                    new("training_item", data = t(X[,idx])[1,], target = Ylab[idx])
                  }),
                  labels=labels,
                  features=features)
    query_set<-new("training_set", 
                   name="query", 
                   items=lapply(1:dim(query)[2], function(idx){
                     new("training_item", data = t(query[,idx])[1,], target = 0)
                   }),
                   labels="Unknown",
                   features=features)
    return(list(training_set, test_set, query_set))
  }
}



get_norm_counts<-function (cds, norm_method = c("log", "binary", "size_only", "none"), 
                           pseudocount = 1) 
{
  software<-NULL
  norm_method = match.arg(norm_method)
  if(class(cds)=="Seurat"){software<-"seurat"}
  if(class(cds)=="cell_data_set"){software<-"monocle3"}
  if(is.null(software)){stop("software not found for input")}
  if(software=="monocle3"){
    norm_mat = SingleCellExperiment::counts(cds)
    if(norm_method=="none"){
      return(norm_mat)
    }
    sf<-size_factors(cds)
  }
  if(software=="seurat"){
    get_counts_seurat(cds)
    if(norm_method=="none"){
      return(norm_mat)
    }
    sf<-seurat_size_factors(cds)
  }
  if (norm_method == "binary") {
    norm_mat = norm_mat > 0
    if (is_sparse_matrix(norm_mat)) {
      norm_mat = methods::as(norm_mat, "dgCMatrix")
    }
  }
  else {
    if (is_sparse_matrix(norm_mat)) {
      norm_mat@x = norm_mat@x/rep.int(sf, 
                                      diff(norm_mat@p))
      if (norm_method == "log") {
        if (pseudocount == 1) {
          norm_mat@x = log10(norm_mat@x + pseudocount)
        }
        else {
          stop("Pseudocount must equal 1 with sparse expression matrices")
        }
      }
    }
    else {
      norm_mat = Matrix::t(Matrix::t(norm_mat)/sf)
      if (norm_method == "log") {
        norm_mat@x <- log10(norm_mat + pseudocount)
      }
    }
  }
  return(norm_mat)
}


seurat_size_factors<-function (cds, round_exprs = TRUE, method = c("mean-geometric-mean-total", 
                                                                   "mean-geometric-mean-log-total")) 
{
  method <- match.arg(method)
  mat<-get_counts_seurat(cds)
  if (any(Matrix::colSums(mat) == 
          0)) {
    warning("Your CDS object contains cells with zero reads. ", 
            "This causes size factor calculation to fail. Please remove ", 
            "the zero read cells using ", "cds <- cds[,Matrix::colSums(exprs(cds)) != 0] and then ", 
            "run cds <- estimate_size_factors(cds)")
    return(cds)
  }
  if (is_sparse_matrix(mat)) {
    sf <- monocle3:::estimate_sf_sparse(mat, 
                                        round_exprs = round_exprs, method = method)
  }
  else {
    sf <- monocle3:::estimate_sf_dense(mat, 
                                       round_exprs = round_exprs, method = method)
  }
  return(sf)
}

is_sparse_matrix<-function (x) 
{
  class(x) %in% c("dgCMatrix", "dgTMatrix", "lgCMatrix")
}


get_norm_counts<-function (cds, norm_method = c("log", "binary", "size_only", "none"), 
                           pseudocount = 1) 
{
  software<-NULL
  norm_method = match.arg(norm_method)
  if(class(cds)=="Seurat"){software<-"seurat"}
  if(class(cds)=="cell_data_set"){software<-"monocle3"}
  if(is.null(software)){stop("software not found for input")}
  if(software=="monocle3"){
    norm_mat = SingleCellExperiment::counts(cds)
    if(norm_method=="none"){
      return(norm_mat)
    }
    sf<-size_factors(cds)
  }
  if(software=="seurat"){
    norm_mat <- get_counts_seurat(cds)
    if(norm_method=="none"){
      return(norm_mat)
    }
    sf<-seurat_size_factors(cds)
  }
  if (norm_method == "binary") {
    norm_mat = norm_mat > 0
    if (is_sparse_matrix(norm_mat)) {
      norm_mat = methods::as(norm_mat, "dgCMatrix")
    }
  }
  else {
    if (is_sparse_matrix(norm_mat)) {
      norm_mat@x = norm_mat@x/rep.int(sf, 
                                      diff(norm_mat@p))
      if (norm_method == "log") {
        if (pseudocount == 1) {
          norm_mat@x = log10(norm_mat@x + pseudocount)
        }
        else {
          stop("Pseudocount must equal 1 with sparse expression matrices")
        }
      }
    }
    else {
      norm_mat = Matrix::t(Matrix::t(norm_mat)/sf)
      if (norm_method == "log") {
        norm_mat@x <- log10(norm_mat + pseudocount)
      }
    }
  }
  return(norm_mat)
}

get_counts_seurat <- function(cds){
  GetAssayData(object = cds, assay = cds@active.assay, slot = "counts")
  # res<-tryCatch(
  # {cds@assays[[cds@active.assay]]@counts},
  # error = function() {
  #   message(paste("data not found in the counts slot; trying list type"))
  #   # Choose a return value in case of error
  #   tryCatch(
  #       {cds@assays[[cds@active.assay]]$counts},
  #       error = function() {
  #         message(paste("data not found in the counts list type - leaving function"))
  #       })
  # }
  # )
  # res
}


#' Common Variant Genes
#' 
#' This function finds common variant genes between two cell data sets.
#' 
#' @param cds1 The first cell data set.
#' @param cds2 The second cell data set.
#' @param top_n The number of top genes to consider. Default is 2000.
#' @param logmean_ul The upper limit for mean expression.
#' @param logmean_ll The lower limit for mean expression.
#' @param row_data_column The column in the feature data corresponding to the gene symbol.
#' @param unique_data_column The column in the feature data corresponding to the unique id.
#' @param verbose Boolean indicating whether to display verbose output.
#' @param plot Boolean indicating whether to plot the results.
#' 
#' @return A vector of similarly variant genes.
#' 
#' @export

common_variant_genes <-function(cds1, 
                                cds2,
                                top_n=2000,
                                logmean_ul = 2, 
                                logmean_ll = -6,
                                row_data_column = "gene_short_name",
                                unique_data_column = "id",
                                verbose = T,
                                plot=F){
  if(class(cds1) != class(cds2)){stop("input objects must be of the same class")}
  software<-NULL
  if(class(cds1)=="Seurat"){
    software<-"seurat"
  }
  if(class(cds1)=="cell_data_set"){
    software<-"monocle3"
  }
  if(software=="monocle3"){
    common_variant_m3(cds1, cds2, top_n,logmean_ul, logmean_ll, 
                      row_data_column, unique_data_column, verbose, plot)
  }
  if(software=="seurat"){
    common_variant_seurat(cds1, cds2, top_n,logmean_ul, logmean_ll, 
                          verbose, plot)
  }
}



common_variant_seurat <-function(cds1, 
                                 cds2,
                                 top_n=2000,
                                 logmean_ul = 2, 
                                 logmean_ll = -6,
                                 verbose = T,
                                 plot=F){
  if(is.null(cds1@misc$dispersion)){
    cds1<-calculate_gene_dispersion(cds1)
  }
  cds1<-select_genes(cds1, top_n = top_n, logmean_ul = logmean_ul, logmean_ll = logmean_ll)
  if(plot){
    if(verbose) {message("Plotting feature dispersion for first object")}
    p<-plot_gene_dispersion(cds1)
    print(p)
  }
  qsel<-get_selected_genes(cds1)
  if(is.null(cds2@misc$dispersion)){
    cds2<-calculate_gene_dispersion(cds2)
  }
  if(plot){
    if(verbose) {message("Plotting feature dispersion (unselected) for second object")}
    p<-plot_gene_dispersion(cds2)
    print(p)
  }
  cds2<-select_genes(cds2, top_n = top_n, logmean_ul = logmean_ul, logmean_ll = logmean_ll)
  if(plot){
    if(verbose) {message("Plotting gene dispersion for second object")}
    p<-plot_gene_dispersion(cds2)
    print(p)
  }
  rsel<-get_selected_genes(cds2)
  selected_common<-intersect(qsel, rsel)
  selected_common
}


common_variant_m3 <-function(cds1, 
                             cds2,
                             top_n=2000,
                             logmean_ul = 2, 
                             logmean_ll = -6,
                             row_data_column = "gene_short_name",
                             unique_data_column = "id",
                             verbose = T,
                             plot=F){
  if(verbose) {message("Calculating feature dispersion for monocle3 object")}
  cds1<-calculate_gene_dispersion(cds1)
  cds1<-select_genes(cds1, top_n = top_n, logmean_ul = logmean_ul, logmean_ll = logmean_ll)
  if(plot){
    if(verbose) {message("Plotting feature dispersion for first object")}
    p<-plot_gene_dispersion(cds1)
    print(p)
  }
  qsel<-rowData(cds1)[[row_data_column]][rowData(cds1)[[unique_data_column]] %in% get_selected_genes(cds1)]
  cds2<-calculate_gene_dispersion(cds2)
  if(plot){
    if(verbose) {message("Plotting feature dispersion (unselected) for second object")}
    p<-plot_gene_dispersion(cds2)
    print(p)
  }
  cds2<-select_genes(cds2, top_n = top_n, logmean_ul = logmean_ul, logmean_ll = logmean_ll)
  if(plot){
    if(verbose) {message("Plotting gene dispersion for second object")}
    p<-plot_gene_dispersion(cds2)
    print(p)
  }
  if(verbose) {message("Returning shared features")}
  rsel<-rowData(cds2)[[row_data_column]][rowData(cds2)[[unique_data_column]] %in% get_selected_genes(cds2)]
  selected_common<-intersect(qsel, rsel)
  selected_common
}


#' Calculate dispersion genes in a cell_data_set object
#' 
#' This function calculates dispersion genes in a cell_data_set object for downstream analysis.
#' 
#' @param cds The cell data set upon which to perform this operation.
#' @param q The polynomial degree.
#' @param id_tag The name of the feature data column corresponding to the unique id.
#' @param symbol_tag The name of the feature data column corresponding to the gene symbol.
#' @param upper_lim The upper limit of dispersion to consider.
#' @param verbose Boolean indicating whether to display verbose output.
#' 
#' @return A vector of dispersion genes.
#' 
#' @export

calculate_gene_dispersion<-function(cds, q=3, id_tag="id", symbol_tag="gene_short_name", method="m3addon", removeOutliers=T){
  software<-NULL
  if(class(cds)=="Seurat"){
    software<-"seurat"
  }
  if(class(cds)=="cell_data_set"){
    software<-"monocle3"
  }
  if(is.null(software)){stop("software not found for input objects")}
  if(software=="monocle3"){
    cds@int_metadata$dispersion<-NULL
    if(method=="m2"){
      df<-data.frame(calc_dispersion_m2(obj = cds, min_cells_detected = 1, min_exprs=1, id_tag=id_tag))
      fdat<-fData(cds)
      if (!is.list(df)) 
        stop("Parametric dispersion fitting failed, please set a different lowerDetectionLimit")
      disp_table <- subset(df, is.na(mu) == FALSE)
      res <- monocle:::parametricDispersionFit(disp_table, verbose = T)
      fit <- res[[1]]
      coefs <- res[[2]]
      if (removeOutliers) {
        CD <- cooks.distance(fit)
        cooksCutoff <- 4/nrow(disp_table)
        message(paste("Removing", length(CD[CD > cooksCutoff]), 
                      "outliers"))
        outliers <- union(names(CD[CD > cooksCutoff]), setdiff(row.names(disp_table), 
                                                               names(CD)))
        res <- monocle:::parametricDispersionFit(disp_table[row.names(disp_table) %in% 
                                                              outliers == FALSE, ], verbose=T)
        fit <- res[[1]]
        coefs <- res[[2]]
        names(coefs) <- c("asymptDisp", "extraPois")
        ans <- function(q) coefs[1] + coefs[2]/q
        attr(ans, "coefficients") <- coefs
      }
      res <- list(disp_table = disp_table, disp_func = ans)
      cds@int_metadata$dispersion<-res
      return(cds)
    }
    if(method=="m3addon"){
      ncounts<-Matrix::t(Matrix::t(exprs(cds))/monocle3::size_factors(cds))
      m<-Matrix::rowMeans(ncounts)
      sd<-sqrt(sparseRowVariances(ncounts))
      fdat<-fData(cds)
      cv<-sd/m*100
      df<-data.frame(log_dispersion=log(cv), log_mean=log(m))
      df[[id_tag]]<-fdat[[id_tag]]
      df<-df[is.finite(df$log_dispersion),]
      model <- lm(data = df, log_dispersion ~ log_mean + poly(log_mean, degree=q))
      prd <- data.frame(log_mean = df$log_mean)
      err<-suppressWarnings(predict(model, newdata= prd, se.fit = T))
      prd$lci <- err$fit - 1.96 * err$se.fit
      prd$fit <- err$fit
      prd$uci <- err$fit + 1.96 * err$se.fit
      prd$log_dispersion<-df$log_dispersion
      prd[[id_tag]]<-df[[id_tag]]
      cds@int_metadata$dispersion<-prd
      return(cds)
    }
  }
  if(software=="seurat"){
    if(method=="m2"){
      stop("m2 method not supported for seurat objects")
    }
    if(method=="m3addon"){
      ncounts<-get_norm_counts(cds)
      m<-Matrix::rowMeans(ncounts)
      sd<-sqrt(sparseRowVariances(ncounts))
      #fdat<-fData(cds)
      cv<-sd/m*100
      df<-data.frame(log_dispersion=log(cv), log_mean=log(m))
      df[[id_tag]]<-rownames(cds)
      df<-df[is.finite(df$log_dispersion),]
      model <- lm(data = df, log_dispersion ~ log_mean + poly(log_mean, degree=q))
      prd <- data.frame(log_mean = df$log_mean)
      err<-suppressWarnings(predict(model, newdata= prd, se.fit = T))
      prd$lci <- err$fit - 1.96 * err$se.fit
      prd$fit <- err$fit
      prd$uci <- err$fit + 1.96 * err$se.fit
      prd$log_dispersion<-df$log_dispersion
      prd[[id_tag]]<-df[[id_tag]]
      #prd$gene_short_name<-fdat[[symbol_tag]][match(prd[[id_tag]], fdat[[id_tag]])]
      cds@misc$dispersion<-prd
      return(cds)
    }
  }
}


#' Helper function for summing sparse matrix groups
#' @references Granja, J. M.et al. (2019). Single-cell multiomic analysis identifies regulatory programs in mixed-phenotype 
#' acute leukemia. Nature Biotechnology, 37(12), 1458–1465.
#' @export
#' @keywords internal
sparseRowVariances <- function (m){
  rM <- Matrix::rowMeans(m)
  rV <- computeSparseRowVariances(m@i + 1, m@x, rM, ncol(m))
  return(rV)
}
