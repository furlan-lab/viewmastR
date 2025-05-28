
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
  # mod<-msgpackRead(model_path, simplify = T)
  #num_classes <- get_modelpak_info()
  # num_classes <- mod$item$linear1$weight$param$shape[2]
  df <- extract_mpk_shapes(model_path)
  num_classes <- as.numeric(df[nrow(df),]$shape)
  if(nrow(df)==2){
    model_type="mlr"
    hidden_layer1 = NULL
    hidden_layer2 = NULL
  } else if (nrow(df)==4) {
    model_type="ann1"
    hidden_layer1 = as.numeric(df[2,4])
    hidden_layer2 = NULL
  } else if (nrow(df)==6) {
    model_type="ann2"
    hidden_layer1 = as.numeric(df[2,4])
    hidden_layer2 = as.numeric(df[4,4])
  } else {
    stop("Model type not found")
  }
  export_list <- viewmastR:::infer_from_model(model_path,  query = query, num_classes = as.integer(num_classes), num_features = as.integer(length(vg)), model_type = model_type, hidden1 = as.integer(hidden_layer1), hidden2 = as.integer(hidden_layer2), verbose = verbose)
  log_odds = unlist(export_list$probs)
  if(length(log_odds) == dim(query_cds)[2]*num_classes){
    log_odds = matrix(log_odds, ncol = dim(query_cds)[2])
    log_odds = t(log_odds)
    if(is.null(labels)){
      labels <- 1:num_classes
    }
    colnames(log_odds) <- paste0("prob_", labels)
  } else {
    stop("Error in log odds dimensions of function output")
  }
  
  softmax_rows <- function(mat) {
    shifted <- mat - apply(mat, 1, max)  # stability
    exp_shifted <- exp(shifted)
    exp_shifted / rowSums(exp_shifted)
  }
  
  #logit_mat <- matrix(log_odds, nrow = num_cells, ncol = num_classes, byrow = TRUE)
  prob_mat  <- softmax_rows(log_odds)
  export_list$probs = prob_mat
  query_cds[[query_celldata_col]]<-labels[apply(log_odds, 1, which.max)]
  if(return_probs){
    query_cds@meta.data <- cbind(query_cds@meta.data, export_list$probs)
  }
  if (return_type=="object") {
    return(query_cds)
  } else {
    return(list(object=query_cds, training_output = export_list))
  }

}

#' Extract the Shape of Every Tensor in a *burn* `.mpk` Checkpoint
#'
#' @description
#' Walks the arbitrary-depth module tree inside a
#' **burn**‐generated MessagePack checkpoint (`*.mpk`) and returns
#' a tidy data frame listing each tensor (e.g., `weight`, `bias`)
#' and its dimensions.  Handles any number of linear (or other) layers
#' and preserves nested sub-module names via a dot-separated path.
#'
#' @param file name of model.mpk file
#'   ```
#'
#' @return A **data frame** with one row per tensor and columns:
#' \describe{
#'   \item{`layer`}{Dot-separated module path (e.g. `"linear1"` or
#'         `"encoder.block1.linear"`).}
#'   \item{`tensor`}{Name of the tensor inside that module
#'         (e.g. `weight`, `bias`).}
#'   \item{`rank`}{Tensor rank (length of the dimension vector).}
#'   \item{`dims`}{A list column – each element is an integer vector of
#'         dimensions.}
#'   \item{`shape`}{Human-readable string, e.g. `"384×1000"` for
#'         matrices or `"128"` for vectors.}
#' }
#'
#' @examples
#' \dontrun{
#' shapes <- extract_mpk_shapes(file)
#' print(shapes)
#' }
#' @importFrom RcppMsgPack msgpackRead
#' @export

extract_mpk_shapes <- function(file) {
  pack <- msgpackRead(file, simplify=T)
  #–– recursive helper ––----------------------------------------------------
  scrape <- function(x, prefix = character(0)) {
    out <- list()
    for (nm in names(x)) {
      cur <- x[[nm]]
      if (is.null(cur)) next                     # skip NULLs (e.g. activation)
      if (is.list(cur) && !is.null(cur$param$shape)) {
        # tensor leaf
        out[[length(out) + 1]] <- list(
          layer  = paste(prefix, collapse = "."),
          tensor = nm,
          shape  = cur$param$shape
        )
      } else if (is.list(cur)) {
        # recurse into sub-module
        out <- c(out, scrape(cur, c(prefix, nm)))
      }
    }
    out
  }
  
  raw_shapes <- scrape(pack$item)
  
  #–– build tidy data frame ––-----------------------------------------------
  df <- do.call(rbind, lapply(raw_shapes, function(rec) {
    dims_vec <- as.integer(rec$shape)
    data.frame(
      layer  = rec$layer,
      tensor = rec$tensor,
      rank   = length(dims_vec),
      dims   = I(list(dims_vec)),
      shape  = paste(dims_vec, collapse = "×"),
      stringsAsFactors = FALSE
    )
  }))
  rownames(df) <- NULL
  df
}


