#' ViewmastR
#' @description in progress
#' @param query_cds cds to query
#' @param ref_cds reference cds
#' @return a cell_data_set object or a list of items if unfiltered data is returned (see unfiltered)
#' @importFrom Matrix colSums
#' @export

viewmastR <-function(query_cds, 
                     ref_cds, 
                     ref_celldata_col, 
                     query_celldata_col=NULL, 
                     FUNC=c("naive_bayes", "neural_network", "bagging","softmax_regression", 
                            "logistic_regression", "deep_belief_nn", "perceptron", "keras_nn",
                            "xgboost", "lasso"),
                     norm_method=c("log", "binary", "size_only", "none"),
                     selected_genes=NULL,
                     train_frac = 0.8,
                     tf_idf=F,
                     scale=F,
                     hidden_layers = c(500,100),
                     learning_rate = 0.5,
                     batch_size = 100,
                     max_epochs = 250,
                     max_error = 0.5,
                     lambda = 1.0,
                     iterations = 1000,
                     LSImethod=1,
                     verbose = T,
                     device = 0,
                     threshold = NULL,
                     keras_model = NULL, ...){
  
  #capture args for specific fxns
  argg <- c(as.list(environment()), list(...))
  layers=F
  
  #deal with conflicting and other args
  if(tf_idf & scale){
    warning("Both tf_idf and scale selected. Cannot do this as they are both scaling methods. Using tf_idf alone")
    scale<-F
  }
  norm_method=match.arg(norm_method)

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
  
  #set fxn
  FUNC=match.arg(FUNC)
  switch(FUNC, 
         naive_bayes={FUNC = naive_bayes
         funclabel="naive_bayes_"
         output = "labels"},
         neural_network={FUNC = af_nn
         funclabel="nn_"
         layers=T
         output = "probs"},
         softmax_regression={FUNC = smr
         funclabel="smr_"
         output = "probs"},
         deep_belief_nn={FUNC = af_dbn
         funclabel="dbnn_"
         output = "probs"},
         logistic_regression={FUNC = lr
         funclabel="lr_"
         output = "probs"},
         bagging={FUNC = bagging
         funclabel="bagging_"
         output = "labels"},
         perceptron={FUNC = perceptron
         funclabel="perceptron_"
         output = "probs"},
         keras_nn={FUNC = keras_helper
         funclabel="keras_"
         output = "probs"},
         xgboost={FUNC = xgboost_helper
         funclabel="xgboost_"
         output = "labels"},
         lasso={FUNC = lasso_helper
         funclabel="lasso_"
         output = "probs"},
  )
  
  #deal with null celltype label
  if(is.null(query_celldata_col)){
    coldata_label<-paste0(funclabel, "celltype")
  }else{
    coldata_label = query_celldata_col
  }
  
  #find common features
  common_list<-common_features(list(ref_cds, query_cds))
  names(common_list)<-c("ref", "query")
  rm(ref_cds)
  gc()
  if(is.null(selected_genes)){
    selected_common<-rownames(common_list[['query']])
    selected_common<-selected_commmon[selected_common %in% rownames(common_list[['ref']])]
  }else{
    selected_common<-selected_genes
    selected_common<-selected_common[selected_common %in% rownames(common_list[['query']])]
    selected_common<-selected_common[selected_common %in% rownames(common_list[['ref']])]
  }
  
  #make final X and query normalizing along the way
  # #no tf_idf
  if(norm_method!="none"){
    query_mat<-get_norm_counts(common_list[['query']], norm_method = norm_method)[selected_common,]
    ref_mat<-get_norm_counts(common_list[['ref']], norm_method = norm_method)[rownames(query_mat),]
  }else{
    query_mat<-get_norm_counts(common_list[['query']], norm_method = )[selected_common,]
    ref_mat<-get_norm_counts(common_list[['ref']], norm_method = )[rownames(query_mat),]
  }
  rm(common_list)
  gc()
  X<-as.matrix(ref_mat)
  query<-as.matrix(query_mat)
  rm(ref_mat, query_mat)
  gc()
  
  #performe scaling methods
  if(tf_idf){
    X<-as.matrix(tf_idf_transform(X, LSImethod))
    query<-as.matrix(tf_idf_transform(query, LSImethod))
  }else{
    if(scale){
      X<-scale(X)
    }
  }
  
  #prep Y
  Ylab<-as.numeric(labf)-1
  labels<-levels(labf)
  Y<-matrix(model.matrix(~0+labf), ncol = length(labels))
  colnames(Y)<-NULL
  rownames(X)<-NULL
  colnames(X)<-NULL
  
  #create test/train indices
  train_idx<-sample(1:dim(X)[2], round(train_frac*dim(X)[2]))
  test_idx<-which(!1:dim(X)[2] %in% train_idx)

  
  #set specific args to run probs fxn
  if(output=="probs"){
    args<-list(X[,train_idx], 
               X[,test_idx], 
               Y[train_idx,], 
               Y[test_idx,], 
               length(labels), 
               query,
               learning_rate = as.double(learning_rate),
               verbose = verbose,
               device = device)
    if(funclabel=="smr_"){
      args$learning_rate=learning_rate
      args$iterations = as.integer(iterations)
      args$lambda = as.integer(lambda)
      args$max_error = as.integer(max_error)
    }
    if(funclabel=="nn_"){
      args$learning_rate=learning_rate
      args$layers = c(as.integer(dim(X[,train_idx])[1]), sapply(hidden_layers, as.integer), as.integer(length(labels)))
      args$max_epochs = as.integer(max_epochs)
      args$batch_size = as.integer(batch_size)
      args$max_error = as.integer(max_error)
    }
    if(funclabel=="perceptron_"){
      args$learning_rate=NULL
    }
    if(funclabel=="dbnn_"){
      if(is.null(argg$rbm_learning_rate)){args$rbm_learning_rate = 0.2} else {args$rbm_learning_rate = argg$rbm_learning_rate}
      if(is.null(argg$nn_learning_rate)){args$nn_learning_rate = 4.0} else {args$nn_learning_rate = argg$nn_learning_rate}
      if(is.null(argg$rbm_epochs)){args$rbm_epochs = 15} else {args$rbm_epochs = argg$rbm_epochs}
      if(is.null(argg$nn_epochs)){args$nn_epochs = 250} else {args$nn_epochs = argg$nn_epochs}
      args$learning_rate=NULL
    }
    if(funclabel=="keras_"){
      args$layers = c(as.integer(dim(X[,train_idx])[1]), sapply(hidden_layers, as.integer), as.integer(length(labels)))
      args$max_epochs = as.integer(max_epochs)
      args$batch_size = as.integer(batch_size)
      args$keras_model = keras_model
    }
    if(funclabel=="lasso_"){
      args$argg<-argg
    }
    
    ##run FXN for probs output
    out<-do.call(FUNC, args)
    colnames(out)<-labels
    
    ##process probs output
    return(process_probabilities(out, query_cds, threshold, software, coldata_label))
  }
  
  #set specific args to run labels fxn
  if(output=="labels"){
    args<-list(X[,train_idx], 
               X[,test_idx], 
               Ylab[train_idx], 
               Ylab[test_idx], 
               length(labels), 
               query, 
               verbose = verbose)
    if(funclabel=="xgboost_"){
      args$argg<-argg
    }
    
    ##run FXN for labels output
    out<-do.call(FUNC, args)
  
    #deal with xgboost
    if(funclabel=="xgboost_"){
      colnames(out)<-labels
      return(process_probabilities(out, query_cds, threshold, software, coldata_label))
    }
    
    #process labels output
    return(process_labels(out, query_cds, software, coldata_label, labels))
  }
}


## helper fxn for dealing with labels output from procedures invoked by viewmastR
process_labels<-function(out, query_cds, software, coldata_label, labels){
  if(software=="seurat"){
    query_cds@meta.data[[coldata_label]]<-labels[out+1]
  }
  if(software=="monocle3"){
    colData(query_cds)[[coldata_label]]<-labels[out+1]
  }
  return(query_cds)
}

## helper fxn for dealing with probs output from procedures invoked by viewmastR
process_probabilities<-function(out, query_cds, threshold=NULL, software, coldata_label){
  if(is.null(threshold)){
    if(software=="seurat"){
      query_cds@meta.data[[coldata_label]]<-colnames(out)[apply(as.data.frame(out),1,which.max)]
    }
    if(software=="monocle3"){
      colData(query_cds)[[coldata_label]]<-colnames(out)[apply(as.data.frame(out),1,which.max)]
    }
    return(query_cds)
  }else{
    if(threshold > 1 & threshold <= 0)stop("thresh must be value between 0 and 1")
    out[out<threshold]<-NA
    outd<-apply(as.data.frame(out),1,which.max)
    outv<-sapply(outd, function(out){
      if(length(out)==0){
        NA
      }else{
        names(out)
      }
    })
    if(software=="seurat"){
      query_cds@meta.data[[coldata_label]]<-outv
    }
    if(software=="monocle3"){
      colData(query_cds)[[coldata_label]]<-outv
    }
    return(query_cds)
  }
}

#' xgboost helper
#' @description A function for input of viewmastR data into xgboost for training and evaluation of a query
#' @return model evaluation of query
#' @importFrom xgboost xgboost
#' 

xgboost_helper<-function(
  x_train, 
      x_test, 
      y_train, 
      y_test, 
      num_classes, 
      query,
      verbose,
      argg){
  if(verbose){verbose<-2}
  if(is.null(argg$nrounds)){argg$nrounds<-20}
  if(is.null(argg$objective)){argg$objective<-"multi:softprob"}
  if(is.null(argg$cores)){argg$cores<-1}
  x_test<-t(x_test)
  x_train<-t(x_train)
  query<-t(query)
  if(verbose){
    message(paste0("Running XGBoost with ", argg$cores, " cores"))
    message(paste0("Train feature dims:\n", paste0(dim(x_train), collapse=" ")))
    message(paste0("Test feature dims:\n", paste0(dim(x_test), collapse=" ")))
    message(paste0("Query dims:\n", paste0(dim(query), collapse=" ")))
    message(paste0("Num classes:\n", num_classes))
    message(paste0("Num rounds ", argg$nrounds))
    message(paste0("Objective function: ", argg$objective))
  }
  bstSparse <- xgboost(data = x_train, label = y_train, nthread = argg$cores, nrounds = argg$nrounds, num_class=num_classes, objective = argg$objective, verbose=verbose)
  Yhat<-t(matrix(predict(bstSparse, x_test), ncol=dim(x_test)[1], nrow=num_classes))
  yhatTest<-apply(Yhat, 1, which.max)-1
  if(verbose){message(paste("XGbBoost test accuracy: ", (length(yhatTest)-table(yhatTest==y_test)[1])/length(yhatTest)))}
  return(t(matrix(predict(bstSparse, query), ncol=dim(query)[1], nrow=num_classes)))
}



#' lasso helper
#' @description A function for input of viewmastR data into lasso for training and evaluation of a query
#' @return model evaluation of query
#' @importFrom glmnet cv.glmnet
#' @importFrom doMC registerDoMC
#' 

lasso_helper<-function(
    x_train, 
    x_test, 
    y_train, 
    y_test, 
    num_classes, 
    query,
    learning_rate,
    verbose,
    device,
    argg){

  x_test<-t(x_test)
  x_train<-t(x_train)
  query <-t(query)
  if(is.null(argg$cores)){argg$cores<-1}
  if(argg$cores>1){
    parallel <- T
    registerDoMC(cores = argg$cores)
  } else {
    parallel <- F
  }
  if(verbose){
    message(paste0("Running Lasso with ", argg$cores, " core(s)"))
    message(paste0("Train feature dims:\n", paste0(dim(x_train), collapse=" ")))
    message(paste0("Test feature dims:\n", paste0(dim(x_test), collapse=" ")))
    message(paste0("Train labels dims:\n", paste0(dim(y_train), collapse=" ")))
    message(paste0("Test labels dims:\n", paste0(dim(y_test), collapse=" ")))
    message(paste0("Query dims:\n", paste0(dim(query), collapse=" ")))
    message(paste0("Num classes:\n", num_classes))
  }
  startTime <- Sys.time()
  cv.lasso <- cv.glmnet(x_train, y_train, family = "multinomial",  parallel = parallel, trace.it=verbose) #lasso
  endTime <- Sys.time()
  if(verbose){
    message(paste0("Training with CV took: ", endTime - startTime))
    message(paste0("Lambda min:  ", cv.lasso$lambda.min))
    plot(cv.lasso)
  }
  Yhat<-predict(cv.lasso, newx = x_test, s = c("lambda.min"))
  Yhat_labels<-apply(Yhat, 1, which.max)-1
  y_test_label <- apply(y_test, 1, which.max)-1
  if(verbose){message(paste("Lasso test accuracy: ", (length(Yhat_labels)-table(Yhat_labels==y_test_label)[1])/length(Yhat_labels)))}
  pred<-predict(cv.lasso, newx = query, s = c("lambda.min"))
  colnames(pred)<-colnames(y_train)
  return(pred)
}

#' Keras helper
#' @description A function for input of viewmastR data into keras for training and evaluation of a query
#' @return model evaluation of query
#' @import keras
#' @import reticulate
#' 

keras_helper<-function(
    x_train, 
    x_test, 
    y_train, 
    y_test, 
    num_classes, 
    query,
    learning_rate,
    verbose,
    layers = layers,
    max_epochs = max_epochs,
    batch_size = batch_size,
    device = device,
    keras_model = NULL
){
  ##check python
  py_config()
  if(!py_available("keras")){stop("keras not found")}
  if(!py_available("tensorflow")){stop("tensorflow not found")}
  x_train = t(x_train)
  x_test = t(x_test)
  message("device = keras")
  message(paste0("Train feature dims:\n", paste0(dim(x_train), collapse=" ")))
  message(paste0("Test feature dims:\n", paste0(dim(x_test), collapse=" ")))
  message(paste0("Train labels dims:\n", paste0(dim(y_train), collapse=" ")))
  message(paste0("Test labels dims:\n", paste0(dim(y_test), collapse=" ")))
  message(paste0("Query dims:\n", paste0(dim(query), collapse=" ")))
  message(paste0("Num classes:\n", num_classes))
  message(paste0("Max epochs:\n", max_epochs))
  message(paste0("Batch size:\n", batch_size))
  if(is.null(keras_model)){
    message(paste0("Creating network with the following layers:\n", paste0(layers, collapse=" ")))
    model <- keras_model_sequential() %>%
      layer_dense(units = layers[2], activation = 'relu', input_shape = dim(x_train)[2]) %>%
      layer_dense(units = layers[3], activation = 'relu') %>% 
      layer_dense(units = num_classes, activation = 'softmax')
    
    # Compile model
    model %>% compile(
      loss = loss_categorical_crossentropy,
      optimizer = optimizer_adadelta(),
      metrics = c('accuracy')
    )}else{
      message("using precompiled keras model")
    }
  
  summary(model)
  
  # Train model
  model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = max_epochs,
    validation_split = 0.2
  )
  
  
  scores <- model %>% evaluate(
    x_test, y_test, verbose = 0
  )
  
  
  # Output metrics
  cat('Test loss:', scores[[1]], '\n')
  cat('Test accuracy:', scores[[2]], '\n')
  
  return(model %>% predict(t(query)))
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
    cds2<-calculate_gene_dispersion(cds1)
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
#' Aubment data
#' @description This function takes a seurat object and finds cells that are not sufficiently abundant when grouped by the
#' column parameter, then simulates data to augment cell number to a level of the parameter - norm_number
#' @param column column from the metadata that designates cell group (i.e. celltype)
#' @param norm_number cell number to augment data to for cells that are not sufficiently abundant in the 
#' @return a seurat object augmented with simulated cells such that all cell groups are present at a level of norm_number of cells
#' @importFrom pbmcapply pbmclapply
#' @importFrom parallel detectCores
#' @export
#' 
augment_data<-function(obj, column, norm_number=2000, assay="RNA"){
  message("Extracting less abundant celltypes")
  splitparam<-as.character(obj[[column]][,1])
  deficiency<-table(splitparam) - setNames(rep(norm_number, length(levels(factor(splitparam)))), levels(factor(splitparam)))
  to_synthesize<-deficiency[deficiency<0]*-1
  spmat<-obj@assays[[assay]]@counts
  universe<-rownames(spmat)
  #type<-to_synthesize[2]
  tsl<-lapply(1:length(to_synthesize), function(n) setNames(as.numeric(to_synthesize[n]), names(to_synthesize[n])))
  message("Simulating cells")
  dl<-pbmclapply(tsl, function(type){
    N<-as.numeric(type)
    type<-names(type)
    #namevec<-c(namevec, rep(type, N))
    rsums<-rowSums(spmat[,splitparam %in% type])
    sizes<-colSums(spmat[,splitparam %in% type])
    # ggplot(data.frame(x=sizes), aes(x=x))+geom_histogram(aes(y = ..density..),
    #                colour = 1, fill = "white")+geom_density()
    den<-density(sizes)
    newsizes <- sample(sizes, N, replace=TRUE) + rnorm(N*2, 0, den$bw)
    # ggplot(data.frame(x=newsizes), aes(x=x))+geom_histogram(aes(y = ..density..),
    #                colour = 1, fill = "white")+geom_density()
    trimmed_newdata<-round(newsizes[newsizes>min(sizes) & max(sizes)], 0)
    final_newsizes<-sample(trimmed_newdata, N)
    # ggplot(data.frame(x=final_newsizes), aes(x=x))+geom_histogram(aes(y = ..density..),
    #                colour = 1, fill = "white")+geom_density()
    splat <- names(rsums)[rep(seq_along(rsums), rsums)]
    dl<-lapply(final_newsizes, function(i){
      tab<-table(sample(splat, newsizes[1]))
      nf<-universe[!universe %in% names(tab)]
      all<-c(tab, setNames(rep(0, length(nf)), nf))
      all[match(universe, names(all))]
    })
    as.sparse(do.call(cbind, dl))
  }, mc.cores = min(c(detectCores(), length(to_synthesize))))
  message("Merging simulated cells")
  sm<-do.call(cbind, dl)
  nvl<-lapply(tsl, function(type){
    N<-as.numeric(type)
    type<-names(type)
    rep(type, N)})
  colnames(sm)<-make.unique(paste0("stimcell_", do.call(c, nvl)))
  seuS<-CreateSeuratObject(counts=sm)
  seuS[[column]]<-do.call(c, nvl)
  message("Returning merged objects")
  merge(obj, seuS)
}


#' Pseudo-singlets
#' @description ip
#' @param se Summarized Experiment
#' @param cds reference cds
#' @return a cell_data_set object or a list of items if unfiltered data is returned (see unfiltered)
#' @importFrom monocle3 new_cell_data_set
#' @importFrom Matrix rowSums
#' @export

pseudo_singlets <- function(se, 
                            sc_cds, 
                            assay_name="logcounts", 
                            logtransformed=T, 
                            selected_genes=NULL,
                            ncells_per_col=50, 
                            threads = detectCores()){
  if(!assay_name %in% names(assays(se))){stop("Assay name not found in Summarized Experiment object;  Run: 'names(assays(se))' to see available assays")}
  mat<-as.matrix(assays(se)[[assay_name]])
  if(logtransformed){
    cmat<-ceiling(exp(mat)-1)
  }else{
    cmat<-mat
  }
  if(is.null(selected_genes)){
    selected_genes<-rownames(se)
  }
  exprs_bulk<-cmat[selected_genes,]
  exprs_sc<-counts(sc_cds)[selected_genes[selected_genes %in% rownames(sc_cds)],]
  depth <- round(sum(rowSums(exprs_sc) / ncol(exprs_sc)))
  nRep <- 5
  n2 <- ceiling(ncells_per_col / nRep)
  ratios <- c(2, 1.5, 1, 0.5, 0.25) #range of ratios of number of fragments
  message(paste0("Simulating ", (ncells_per_col * dim(se)[2]), " single cells"))
  syn_mat <- pbmcapply::pbmclapply(seq_len(ncol(exprs_bulk)), function(x){
    counts <- exprs_bulk[, x]
    counts <- rep(seq_along(as.numeric(counts)), as.numeric(counts))
    simMat <- lapply(seq_len(nRep), function(y){
      ratio <- ratios[y]
      simMat <- matrix(sample(x = counts, size = ceiling(ratio * depth) * n2, replace = TRUE), ncol = n2)
      simMat <- Matrix::summary(as(simMat, "dgCMatrix"))[,-1,drop=FALSE]
      simMat[,1] <- simMat[,1] + (y - 1) * n2
      simMat
    }) %>%  Reduce("rbind", .)
    simMat <- Matrix::sparseMatrix(i = simMat[,2], j = simMat[,1], x = rep(1, nrow(simMat)), dims = c(nrow(exprs_bulk), n2 * nRep))
    colnames(simMat) <- paste0(colnames(exprs_bulk)[x], "#", seq_len(ncol(simMat)))
    rownames(simMat)<-rownames(exprs_bulk)
    simMat}, mc.cores =  threads)
  syn_mat <- Reduce("cbind", syn_mat)
  if(any(is.nan(syn_mat@x))){
    syn_mat@x[is.nan(syn_mat@x)]<-0
    warning("NaN calculated during single cell generation")
  }
  slice<-rep.int(1:nrow(colData(se)), ncells_per_col)
  slice<-slice[order(slice)]
  sim_meta_data<-colData(se)[slice,]
  rownames(sim_meta_data)<-colnames(syn_mat)
  new_cell_data_set(syn_mat, sim_meta_data, DataFrame(row.names = rownames(syn_mat), gene_short_name = rownames(syn_mat), id = rownames(syn_mat)))
}



#' Seurat to Monocle3
#' @description Conver Seurat to Monocle3
#' @param seu Seurat Object
#' @param seu_rd Reduced dimname for seurat ('i.e. UMAP'); use NULLto not copy dimensionality reduction
#' @param assay_name Name of data slot ('i.e. RNA')
#' @param mon_rd Reduced dimname for monocle3 ('i.e. UMAP'); use NULL to not copy dimensionality reduction
#' @import Seurat
#' @import monocle3
#' @return a cell_data_set object
#' @export


seurat_to_monocle3 <-function(seu, seu_rd="umap", mon_rd="UMAP", assay_name="RNA"){
  cds<-new_cell_data_set(seu@assays[[assay_name]]@counts, 
                         cell_metadata = seu@meta.data, 
                         gene_metadata = DataFrame(
                           row.names = rownames(seu@assays[[assay_name]]@counts), 
                           id=rownames(seu@assays[[assay_name]]@counts), 
                           gene_short_name=rownames(seu@assays[[assay_name]]@counts)))
  if(!is.null(seu_rd)){
    if(!is.null(mon_rd)){
      reducedDims(cds)[[mon_rd]]<-seu@reductions[[seu_rd]]@cell.embeddings
    }
  }
  cds
}

#' Monocle3 to Seurat
#' @description Conver Monocle3 cell data set to a Seurat object.  For a variety of reasons, the recommendations are to use this funciont
#' only to generate skeleton Seurat objects that can be used for plotting and not much else.  The resulting object will not contain PCA reducitons or 
#' nearest neighbor graphs.
#' @param seu Seurat Object
#' @param seu_rd Reduced dimname for seurat ('i.e. UMAP')
#' @param assay_name Name of data slot ('i.e. RNA')
#' @param mon_rd Reduced dimname for monocle3 ('i.e. UMAP')
#' @param row.names rowData column to use as rownames for Seurat object
#' @import Seurat
#' @import monocle3
#' @return a cell_data_set object
#' @export


monocle3_to_seurat <-function(cds, seu_rd="umap", mon_rd="UMAP", assay_name="RNA", row.names="gene_short_name", normalize=T){
  warning("this function will create a Seurat object with only 1 reduced dimension; currently only UMAP is supported")
  counts <- exprs(cds)
  rownames(counts) <- rowData(cds)[[row.names]]
  seu<-CreateSeuratObject(counts, meta.data = data.frame(colData(cds)))
  keyname<-paste0(seu_rd, "_")
  colnames(reducedDims(cds)[[mon_rd]])<-paste0(keyname, 1:dim(reducedDims(cds)[[mon_rd]])[2])
  seu@reductions[[seu_rd]]<-Seurat::CreateDimReducObject(embeddings = reducedDims(cds)[[mon_rd]], key = keyname, assay = assay_name, )
  if(normalize){seu<-NormalizeData(seu)}
  seu
}


#' Franken
#' @description Will prepare monocle3 objects for use across species
#' @param cds cell_data_set
#' @param rowdata_col rowData column to lookup 
#' @param from_species currently only mouse ("mm") and human ("hs") symbols supported
#' @import monocle3
#' @return a cell_data_set object
#' @export

franken<-function(cds, rowdata_col="gene_short_name", from_species="mm", to_species="hs", trim=T){
  message("Currently only human and mgi symbols supported")
  if(from_species==to_species){return(cds)}
  labels<-data.frame(mm=c(dataset="mmusculus_gene_ensembl", prefix="mgi"), hs=c(dataset="hsapiens_gene_ensembl", prefix="hgnc"))
  from_X<-rowData(cds)[[rowdata_col]]
  from_dataset=labels[,from_species][1]
  to_dataset=labels[,to_species][1]
  from_type = paste0(labels[,from_species][2], "_symbol")
  to_type = paste0(labels[,to_species][2], "_symbol")
  from_labelout<-paste0(toupper(labels[,from_species][2]), ".symbol")
  to_labelout<-paste0(toupper(labels[,to_species][2]), ".symbol")
  franken_table<-franken_helper(from_X, from_dataset, to_dataset, from_type, to_type)
  new_column_name<-paste0("franken_", from_species, "_to_", to_species)
  rowData(cds)[[new_column_name]]<-franken_table[[to_labelout]][match(from_X, franken_table[[from_labelout]])]
  rowData(cds)[[new_column_name]][is.na(rowData(cds)[[new_column_name]])]<-"Unknown"
  if(trim){
    cds<-cds[!rowData(cds)[[new_column_name]] %in% "Unknown",]
  }
  rownames(cds)<-rowData(cds)[[new_column_name]]
  rowData(cds)[[rowdata_col]]<-rowData(cds)[[new_column_name]]
  cds
}


#' @importFrom biomaRt getLDS
#' @export
franken_helper <- function(x, from_dataset="mmusculus_gene_ensembl", to_dataset="hsapiens_gene_ensembl", from_type="mgi_symbol", to_type="hgnc_symbol"){
  from_mart = useMart("ensembl", dataset = from_dataset)
  to_mart = useMart("ensembl", dataset = to_dataset)
  genesV2 = getLDS(attributes = c(from_type), filters = from_type, values = x , mart = from_mart, attributesL = c(to_type), martL = to_mart, uniqueRows=F)
  return(genesV2)
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
    norm_mat = cds@assays[[cds@active.assay]]@counts
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
  if (any(Matrix::colSums(cds@assays[[cds@active.assay]]@counts) == 
          0)) {
    warning("Your CDS object contains cells with zero reads. ", 
            "This causes size factor calculation to fail. Please remove ", 
            "the zero read cells using ", "cds <- cds[,Matrix::colSums(exprs(cds)) != 0] and then ", 
            "run cds <- estimate_size_factors(cds)")
    return(cds)
  }
  if (is_sparse_matrix(cds@assays[[cds@active.assay]]@counts)) {
    sf <- monocle3:::estimate_sf_sparse(cds@assays[[cds@active.assay]]@counts, 
                                        round_exprs = round_exprs, method = method)
  }
  else {
    sf <- monocle3:::estimate_sf_dense(cds@assays[[cds@active.assay]]@counts, 
                                       round_exprs = round_exprs, method = method)
  }
  return(sf)
}

#' demo
#' @description run classification demo
#' @param perc percent to holdout
#' @param FUNC algorithm to use
#' @importFrom devtools session_info
#' @export
vm_demo<-function(perc = 80, 
               FUNC=c("naive_bayes", 
                      "neural_network", 
                      "softmax_regression",
                      "deep_belief_nn", 
                      "logistic_regression",
                      "bagging", 
                      "perceptron",
                      "keras"), device = 0)
  {
  d<-session_info()
  lib_path <- as.character(d$packages$library[d$packages$package=="viewmastR"])
  FUNC=match.arg(FUNC)
  switch(FUNC, 
         naive_bayes={FUNC = naive_bayes_demo},
         neural_network={FUNC = ann_demo},
         softmax_regression={FUNC = smr_demo_helper},
         deep_belief_nn={FUNC = dbn_demo},
         logistic_regression={FUNC = lr_demo},
         bagging={FUNC = bagging_demo},
         perceptron={FUNC = perceptron_demo},
         keras={FUNC = keras_demo},
         # xgboost={FUNC = xgboost_helper},
         # lasso={FUNC = lasso_helper},
  )
  FUNC(lib_path, perc=perc, device = device)
}


keras_demo<-function(lib_path, perc = 80, device){
  model <- keras_model_sequential() %>%
    layer_dense(units = 256, activation = "relu", input_shape = c(784)) %>%
    layer_dropout(rate = 0.25) %>% 
    layer_dense(units = 128, activation = "relu") %>%
    layer_dropout(rate = 0.25) %>% 
    layer_dense(units = 64, activation = "relu") %>%
    layer_dropout(rate = 0.25) %>%
    layer_dense(units = 10, activation = "softmax")
  summary(model)
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(),
    metrics = c("accuracy")
  )
  
  l<-get_mnist(lib_path = lib_path, perc = perc)
  
  X_train <- t(l[[1]])
  X_test <- t(l[[2]])
  y_train <-to_categorical(l[[3]], num_classes = 10)
  y_test <- to_categorical(l[[4]], num_classes = 10)
  
  time1 <- Sys.time()
  history <- model %>% 
    fit(X_train, y_train, epochs = 50, batch_size = 128, validation_split = 0.15)
  time2 <- Sys.time()
  time2 - time1
}
