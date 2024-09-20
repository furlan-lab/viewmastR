
#' 
#' @keywords internal
infer_prep<-function(object, variable_features, software){
  qcounts<-t(as.matrix(get_norm_counts(object)[variable_features,]))
  query = lapply(1:dim(qcounts)[1], function(idx){
    list(data = as.numeric(qcounts[idx,]))
  })
}


#' Function to infer cell labels using a trained model
#' 
#' This function infers cell labels using a trained model and updates the input dataset with the inferred labels.
#' 
#' @param query_cds Seurat or cell_data_set object - The dataset for which cell labels are to be inferred.
#' @param model_path character path to the trained model file.
#' @param vg character vector - Features used for inference (must be the same used during model creation).
#' @param query_celldata_col character vector - names of the column to store inferred cell labels in the query dataset. Default is "viewmastR_inferred".
#' @param labels character vector - optional labels corresponding to the class indices. Default is NULL.
#' @param verbose bool - show messaging
#' @param return_probs logical If TRUE, returns the class probabilities. Default is FALSE.
#' 
#' @return If return_probs is TRUE, returns a data frame containing class probabilities.
#' If return_probs is FALSE, updates the query_cds object with inferred cell labels and returns it.
#' @importFrom RcppMsgPack msgpackRead
#' @export

viewmastR_infer<-function(query_cds, 
                          model_path, 
                          vg, 
                          query_celldata_col = "viewmastR_inferred", 
                          labels = NULL, 
                          verbose = T, 
                          return_probs=F, 
                          return_type = c("object", "list")
                          ){
  return_type <- match.arg(arg = NULL, return_type)
  software<-NULL
  if(class(query_cds)=="Seurat"){
    software<-"seurat"
  }
  if(class(query_cds)=="cell_data_set"){
    software<-"monocle3"
  }
  if(is.null(software)){stop("Only seurat and monocle3 objects supported")}
  if(verbose){message("Preparing query")}
  query<-infer_prep(query_cds, vg, software)
  mod<-msgpackRead(model_path, simplify = T)
  num_classes <- mod$item$linear1$weight$param$shape[2]
  export_list <- viewmastR:::infer_from_model(model_path,  query = query, num_classes = num_classes, num_features = length(vg), verbose = verbose)
  log_odds = unlist(export_list$probs)
  if(is.integer(length(log_odds) %% dim(query_cds)[2])){
    log_odds = matrix(log_odds, ncol = dim(query_cds)[2])
    log_odds = t(log_odds)
    if(is.null(labels)){
      labels = paste0("porb_celltype_", 1:dim(log_odds)[1])
    }
    colnames(log_odds) <- paste0("prob_", labels)
  } else {
    stop("Error in log odds dimensions of function output")
  }
  export_list$probs = plogis(log_odds)
  query_cds[[query_celldata_col]]<-labels[apply(log_odds, 1, which.max)]
  if(return_probs){
    query_cds@meta.data <- cbind(query_cds@meta.data, export_list$probs)
  }
  if (return_type=="object") {
    query_cds
  } else {
    list(object=query_cds, training_output = export_list)
  }
}
  
