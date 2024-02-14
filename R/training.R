setClass("training_item", slots=c(data="numeric", target="numeric"))
setClass("training_set", slots=c(name="character", items="list", labels="character"))

#' Running viewmastR using the new Rust implementation
#' @description see viewmastR
#' @param query_cds cds to query
#' @param ref_cds reference cds
#' @return various forms of training test data and query
#' @export

viewmastR <-function(query_cds, 
                     ref_cds, 
                     ref_celldata_col, 
                     query_celldata_col=NULL, 
                     FUNC=c("softmax_regression"),
                     norm_method=c("log", "binary", "size_only", "none"),
                     selected_genes=NULL,
                     train_frac = 0.8,
                     tf_idf=F,
                     scale=F,
                     hidden_layers = c(500,100),
                     learning_rate = 1e-3,
                     batch_size = 100,
                     max_epochs = 10,
                     max_error = 0.5,
                     lambda = 1.0,
                     iterations = 1000,
                     LSImethod=1,
                     verbose = T,
                     device = 0,
                     threshold = NULL,
                     keras_model = NULL, 
                     dir = "/tmp/sc_local", 
                     return_type = c("object", "list"), ...){
  
  return_type <- match.arg(arg = NULL, return_type)
  training_list<-setup_training(query_cds, ref_cds, ref_celldata_col = ref_celldata_col, selected_genes = selected_genes, verbose=verbose, return_type = "list", use_sparse = F)
  dir.create(dir)
  export_list<-process_learning_obj_mlr(train = training_list[["train"]], 
                                        test = training_list[["test"]], 
                                        query = training_list[["query"]], 
                                        labels = training_list[["labels"]], 
                                        learning_rate = learning_rate, num_epochs = max_epochs, 
                                        directory = dir, verbose = verbose, backend = "wgpu")
  if(is.null(query_celldata_col)){
    query_celldata_col<-"viewmastR_smr"
  }
  query_cds[[query_celldata_col]]<-training_list[["labels"]][export_list$predictions[[1]]+1]
  if (return_type=="object") {
    query_cds
  } else {
    list(object=query_cds, training_output = export_list)
  }
}


#' Setup training datasets
#' @description see viewmastR
#' @param query_cds cds to query
#' @param ref_cds reference cds
#' @return various forms of training test data and query
#' @importFrom Matrix colSums
#' @importFrom MatrixExtra t_shallow
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
                          use_sparse = F, 
                          return_type = c("list", "matrix", "S4obj"),
                          ...){
  
  if(verbose){
    message("Checking arguments and input")
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
  if(!use_sparse){
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
  } else {
    if(addbias){
      if(verbose){
        message("Adding bias")
      }
      X <-rbind(rep(1, ncol(X)), X)
      query <-rbind(rep(1, ncol(query)), query)
    }
    X<-as(X, "RsparseMatrix")
    query<-as(query, "RsparseMatrix")
  }
  
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
  if(return_type == "matrices" && !use_sparse){
    return(list(Xtrain_data = t(X[,train_idx]), 
                Xtest_data = t(X[,test_idx]), 
                Ytrain_label = Y[train_idx,], 
                Ytest_label = Y[test_idx,],
                query = t(query),
                label_text = labels,
                features = features))
  } else if(return_type == "matrices" && use_sparse){
    return(list(Xtrain_data = t_shallow(X[,train_idx]), 
                Xtest_data = t_shallow(X[,test_idx]), 
                Ytrain_label = Y[train_idx,], 
                Ytest_label = Y[test_idx,],
                query = t_shallow(query),
                label_text = labels,
                features = features))
  } else if(return_type == "list" && !use_sparse) {
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
  } else if(return_type == "list" && use_sparse) {
    return(list(
      train = lapply(train_idx, function(idx){
        slice <- X[,idx]
        indices <- which(!slice == 0)
        list(values= slice[ indices] , indices=indices,
             target = Ylab[idx])
      }),
      test = lapply(test_idx, function(idx){
        slice <- X[,idx]
        indices <- which(!slice == 0)
        list(values= slice[ indices] , indices=indices,
             target = Ylab[idx])
      }),
      query = lapply(1:dim(query)[2], function(idx){
        slice <- query[,idx]
        indices <- which(!slice == 0)
        list(values= slice[ indices] , indices=indices)
      }),
      labels = labels,
      features = features))
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
#' @description Find common variant genes between two cds objects
#' @param cds1 cds 
#' @param cds2 
#' @return a vector of similarly variant genes
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
#' \code{calculate_dispersion} is an optional step in the trajectory building
#' process before \code{preprocess_cds}.  After calculating dispersion for
#' a cell_data_set using the \code{calculate_gene_dispersion} function, the 
#' \code{select_genes} function allows the user to identify a set of genes
#' that will be used in downstream dimensionality reduction methods.  These
#' genes and their disperion and mean expression can be plotted using the 
#' \code{plot_gene_dispersion} function.
#'
#'
#' @param cds the cell_data_set upon which to perform this operation.
#' @param q the polynomial degree; default = 3.
#' @param id_tag the name of the feature data column corresponding to 
#' the unique id - typically ENSEMBL id; default = "id".
#' @param symbol_tag the name of the feature data column corresponding to 
#' the gene symbol; default = "gene_short_name".
#' @return an updated cell_data_set object with dispersion and mean expression saved
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
