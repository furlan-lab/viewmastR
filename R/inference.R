
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

viewmastR_infer<-function(query_cds, model_path, vg, query_celldata_col = "viewmastR_inferred", labels = NULL, verbose = T, return_probs=F){
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
  d<-viewmastR:::infer_from_model(model_path,  query = query, num_classes = num_classes, num_features = length(vg), verbose = verbose)
  if(return_probs){
    d
  } else {
    if(is.null(labels)){
      query_cds[[query_celldata_col]]<-d$predictions+1
      query_cds
    } else {
      query_cds[[query_celldata_col]]<-labels[d$predictions+1]
      query_cds
    }
  }
}
  
