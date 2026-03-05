
################################################################################
# FILE: R/training.R
# STATUS: Clean
# ------------------------------------------------------------------------------
# Functions:
# [ ] dimension_check           (Not Exported)
# [x] optimize_backend          (Exported - Internal)
# [x] viewmastR                 (Exported)
# [x] setup_training            (Exported - Internal)
# [ ] seurat_size_factors       (Not Exported)
# [ ] is_sparse_matrix          (Not Exported)
# [ ] get_norm_counts           (Not Exported)
# [ ] get_seurat_version        (Not Exported)
# [ ] get_counts_seurat         (Not Exported)
# [x] sparseRowVariances        (Exported - Internal)
# [x] call_process_learning_obj_sparse (Exported - Internal)
################################################################################


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

#' @keywords internal
optimize_backend <- function(){
  if(Sys.info()["machine"]=="arm64" & Sys.info()['sysname'] == "Darwin"){
    return("nd")
  }
  if(Sys.info()["machine"]=="x86_64" & Sys.info()['sysname'] == "Linux"){
    return("candle")
  }
  else {
    return("wgpu")
  }
}

#' Integrate and Train Models on Reference Dataset and (Optionally) Infer on Query Datasets
#'
#' The \code{viewmastR} function preprocesses one or two single-cell datasets (a reference and an optional query), 
#' splits the reference data into training and test sets, and optionally includes the ability to run inference on a query dataset 
#' for downstream analysis. It then applies specified modeling functions (e.g., MLR, NN, NB) to train and optionally predict on the 
#' query data.
#'
#' @param query_cds A \code{Seurat} or \code{cell_data_set} object representing the query dataset. If \code{NULL}, 
#'   the function will operate in "reference-only" mode, using the reference dataset for training and testing only.
#' @param ref_cds A \code{Seurat} or \code{cell_data_set} object representing the reference dataset. This is required.
#' @param ref_celldata_col A character string specifying the metadata column in \code{ref_cds} that contains the cell labels.
#' @param query_celldata_col A character string specifying a metadata column name in \code{query_cds} (or reference in 
#'   reference-only mode) where predicted labels should be stored. If \code{NULL}, defaults to \code{"viewmastR_pred"}.
#' @param FUNC A character string specifying the modeling function to apply. One of \code{"mlr"}, \code{"nn"}, or \code{"nb"}.
#' @param norm_method Character string indicating the normalization method. One of \code{"log"}, \code{"binary"}, 
#'   \code{"size_only"}, or \code{"none"}.
#' @param selected_features A character vector specifying genes to subset. If \code{NULL}, uses the set of common features 
#'   (if query is provided) or selected genes directly (if reference-only).
#' @param train_frac A numeric value between 0 and 1 specifying the fraction of reference cells to use for training. 
#'   The remainder are used for testing.
#' @param tf_idf Logical, whether to apply TF-IDF transformation after normalization.
#' @param scale Logical, whether to scale the data. If both \code{tf_idf} and \code{scale} are \code{TRUE}, TF-IDF takes precedence.
#' @param hidden_layers A numeric vector indicating the size of hidden layers (for the NN model). Only 1 or 2 layers are allowed.
#' @param learning_rate Numeric, learning rate for model training.
#' @param max_epochs Integer, the maximum number of epochs for model training.
#' @param LSImethod Integer, specifying the TF-IDF method variant if using TF-IDF.
#' @param verbose Logical, whether to print progress messages.
#' @param backend A character string specifying the backend to use. One of \code{"wgpu"}, \code{"nd"}, \code{"candle"}.
#' @param threshold Currently unused. Can be \code{NULL}.
#' @param keras_model Currently unused. Can be \code{NULL}.
#' @param model_dir A character string specifying the directory to store model artifacts.
#' @param return_probs Logical, whether to return predicted probabilities (softmax) in the object's metadata.
#' @param return_logits Logical, whether to return raw logits (pre-softmax scores) in the object's metadata.
#' @param return_type A character string specifying the return type. One of \code{"object"} or \code{"list"}.
#'   If \code{"object"}, returns the updated \code{query_cds}. If \code{"list"}, returns a list containing
#'   \code{object} and \code{training_output}.
#' @param debug Logical, whether to print debugging messages and dimension checks.
#' @param train_only Logical, if \code{TRUE}, only the reference data is processed and no query data is included.
#'
#' @details 
#' The function first calls \code{\link{setup_training}} to preprocess and split the data into training, testing, and 
#' optionally query subsets. Then, based on the selected \code{FUNC}, it calls one of the model training and prediction 
#' functions (\code{process_learning_obj_mlr}, \code{process_learning_obj_ann}, \code{process_learning_obj_nb}). 
#' If \code{train_only = TRUE}, the query portion is skipped and no query predictions are made.
#'
#' For \code{"mlr"} and \code{"nn"} functions, predicted log odds are converted to probabilities using the logistic function. 
#' Predicted cell labels are assigned to the \code{query_cds} (or \code{ref_cds} if query is not provided).
#'
#' @return 
#' Depending on \code{return_type}, returns either:
#' \itemize{
#'   \item \code{return_type = "object"}: the input \code{query_cds} (or \code{ref_cds} if \code{query_cds = NULL}) with predicted 
#'   labels (and optionally probabilities) appended.
#'   \item \code{return_type = "list"}: a list containing:
#'   \describe{
#'     \item{object}{The updated \code{query_cds} (or \code{ref_cds}).}
#'     \item{training_output}{The output from the model training process, including probabilities if applicable.}
#'   }
#' }
#'
#' @examples
#' \dontrun{
#' # Training and predicting with reference and query data:
#' res <- viewmastR(
#'   query_cds = query_seurat_obj,
#'   ref_cds = ref_seurat_obj,
#'   ref_celldata_col = "cell_type",
#'   FUNC = "mlr",
#'   norm_method = "log",
#'   train_frac = 0.8,
#'   backend = "wgpu",
#'   verbose = TRUE,
#'   return_type = "object"
#' )
#'
#' # Reference-only scenario:
#' res_ref <- viewmastR(
#'   query_cds = NULL,
#'   ref_cds = ref_cds_obj,
#'   ref_celldata_col = "cell_type",
#'   FUNC = "nn",
#'   norm_method = "none",
#'   train_frac = 0.7,
#'   scale = TRUE,
#'   train_only = TRUE,
#'   return_type = "list"
#' )
#' }
#'
#' @export

viewmastR <- function(query_cds, 
                      ref_cds, 
                      ref_celldata_col, 
                      query_celldata_col = NULL, 
                      FUNC = c("mlr", "nn", "nb"),
                      norm_method = c("log", "binary", "size_only", "none"),
                      selected_features = NULL,
                      train_frac = 0.8,
                      tf_idf = FALSE,
                      scale = FALSE,
                      hidden_layers = c(as.integer(500), as.integer(100)),
                      learning_rate = 1e-3,
                      max_epochs = 5,
                      LSImethod = 1,
                      verbose = TRUE,
                      backend = c("auto", "wgpu", "nd", "candle"),
                      threshold = NULL,
                      keras_model = NULL, 
                      model_dir = "/tmp/sc_local",
                      return_probs = FALSE,
                      return_logits = FALSE,
                      return_type = c("object", "list"),
                      debug = FALSE,
                      train_only = FALSE
) {
  FUNC <- match.arg(FUNC)
  return_type <- match.arg(return_type)
  backend <- match.arg(backend)
  norm_method <- match.arg(norm_method)
  
  if (backend == "auto") {
    backend <- optimize_backend()
  }
  
  # Set default query_celldata_col

if (is.null(query_celldata_col)) {
    query_celldata_col <- "viewmastR_pred"
  }
  
  # Create model directory if needed
  if (!file.exists(model_dir)) {
    dir.create(model_dir, recursive = TRUE)
  }
  
  # Determine model type for MLR/NN
  model <- NULL
  if (FUNC == "nn") {
    if (length(hidden_layers) == 2) {
      model <- "ann2"
    } else if (length(hidden_layers) == 1) {
      model <- "ann"
    } else {
      stop("Number of hidden layers can only be 1 or 2")
    }
  } else if (FUNC == "mlr") {
    model <- "mlr"
  }
  
  # ============================================================================
  # NAIVE BAYES PATH - uses old list format
  # ============================================================================
  if (FUNC == "nb") {
    if (train_only) {
      message("Naive Bayes is currently not implemented for training only")
      return(NULL)
    }
    
    # NB requires the old list format
    training_list <- setup_training(
      query_cds = query_cds,
      ref_cds = ref_cds,
      ref_celldata_col = ref_celldata_col,
      norm_method = norm_method,
      selected_features = selected_features,
      train_frac = train_frac,
      tf_idf = tf_idf,
      scale = scale,
      LSImethod = LSImethod,
      verbose = verbose,
      addbias = FALSE,
      return_type = "list",
      debug = debug
    )
    
    if (debug) {
      message("Dimension check (Naive Bayes):")
      message(paste0("\t", dimension_check(training_list[["train"]][[1]])))
      message(paste0("\t", dimension_check(training_list[["test"]][[1]])))
      message(paste0("\t", dimension_check(training_list[["query"]][[1]])))
      message(paste0("\t", dimension_check(training_list[["labels"]])))
    }
    
    export_list <- process_learning_obj_nb(
      train = training_list[["train"]],
      test = training_list[["test"]],
      query = training_list[["query"]]
    )
    
    if (return_probs) {
      message("Probabilities from multinomial naive bayes not implemented yet")
    }
    
    query_cds[[query_celldata_col]] <- training_list[["labels"]][export_list$predictions[[1]] + 1]
    
    if (return_type == "object") {
      return(query_cds)
    } else {
      return(list(
        object = query_cds, 
        training_output = list(
          history = export_list$history, 
          duration = export_list$duration, 
          params = export_list$params
        ),
        model_dir = model_dir
      ))
    }
  }
  
  # ============================================================================
  # MLR / NN PATH - uses new sparse format
  # ============================================================================
  
  ## TRAIN ONLY MODE (no query)
  if (train_only) {
    training_list <- setup_training(
      query_cds = NULL,
      ref_cds = ref_cds,
      ref_celldata_col = ref_celldata_col,
      norm_method = norm_method,
      selected_features = selected_features,
      train_frac = train_frac,
      tf_idf = tf_idf,
      scale = scale,
      LSImethod = LSImethod,
      verbose = verbose,
      addbias = FALSE,
      return_type = "sparse_mat",
      debug = debug
    )
    
    if (debug) {
      message("Dimension check (Training Only - Sparse):")
      message(paste0("\ttrain_mat: ", nrow(training_list$train_mat), " x ", ncol(training_list$train_mat)))
      message(paste0("\ttest_mat: ", nrow(training_list$test_mat), " x ", ncol(training_list$test_mat)))
      message(paste0("\tlabels: ", length(training_list$labels)))
    }
    
    export_list <- call_process_learning_obj_sparse(
      model_type = model,
      train_mat = training_list$train_mat,
      train_labels = training_list$train_labels,
      train_sf = training_list$train_sf,
      test_mat = training_list$test_mat,
      test_labels = training_list$test_labels,
      test_sf = training_list$test_sf,
      query_mat = NULL,
      query_sf = NULL,
      labels = training_list$labels,
      feature_names = training_list$features,
      hidden_size = as.integer(hidden_layers),
      learning_rate = learning_rate,
      num_epochs = max_epochs,
      directory = model_dir,
      verbose = verbose,
      backend = backend
    )
    
    return(list(
      object = NULL, 
      training_output = list(
        history = export_list$history, 
        duration = export_list$duration, 
        params = export_list$params
      ), 
      model_dir = model_dir
    ))
  }
  
  ## TRAIN + QUERY MODE
  training_list <- setup_training(
    query_cds = query_cds,
    ref_cds = ref_cds,
    ref_celldata_col = ref_celldata_col,
    norm_method = norm_method,
    selected_features = selected_features,
    train_frac = train_frac,
    tf_idf = tf_idf,
    scale = scale,
    LSImethod = LSImethod,
    verbose = verbose,
    addbias = FALSE,
    return_type = "sparse_mat",
    debug = debug
  )
  
  if (debug) {
    message("Dimension check (Train + Query - Sparse):")
    message(paste0("\ttrain_mat: ", nrow(training_list$train_mat), " x ", ncol(training_list$train_mat)))
    message(paste0("\ttest_mat: ", nrow(training_list$test_mat), " x ", ncol(training_list$test_mat)))
    message(paste0("\tquery_mat: ", nrow(training_list$query_mat), " x ", ncol(training_list$query_mat)))
    message(paste0("\tlabels: ", length(training_list$labels)))
  }
  
  export_list <- call_process_learning_obj_sparse(
    model_type = model,
    train_mat = training_list$train_mat,
    train_labels = training_list$train_labels,
    train_sf = training_list$train_sf,
    test_mat = training_list$test_mat,
    test_labels = training_list$test_labels,
    test_sf = training_list$test_sf,
    query_mat = training_list$query_mat,
    query_sf = training_list$query_sf,
    labels = training_list$labels,
    feature_names = training_list$features,
    hidden_size = as.integer(hidden_layers),
    learning_rate = learning_rate,
    num_epochs = max_epochs,
    directory = model_dir,
    verbose = verbose,
    backend = backend
  )
  
  # Process predictions
  log_odds <- unlist(export_list$probs)
  
  # Check dimensions
  expected_length <- ncol(query_cds) * length(training_list$labels)
  if (length(log_odds) != expected_length) {
    stop(sprintf(
      "Error in log odds dimensions: expected %d, got %d", 
      expected_length, length(log_odds)
    ))
  }
  
  log_odds <- matrix(log_odds, ncol = ncol(query_cds))
  log_odds <- t(log_odds)
  colnames(log_odds) <- paste0("logit_", training_list$labels)

  # Softmax conversion
  softmax_rows <- function(mat) {
    shifted <- mat - apply(mat, 1, max)  # numerical stability
    exp_shifted <- exp(shifted)
    exp_shifted / rowSums(exp_shifted)
  }

  prob_mat <- softmax_rows(log_odds)
  colnames(prob_mat) <- paste0("prob_", training_list$labels)
  query_cds[[query_celldata_col]] <- training_list$labels[apply(prob_mat, 1, which.max)]

  if (return_logits) {
    query_cds@meta.data <- cbind(query_cds@meta.data, log_odds)
  }

  if (return_probs) {
    query_cds@meta.data <- cbind(query_cds@meta.data, prob_mat)
  }

  if (return_type == "object") {
    return(query_cds)
  } else {
    return(list(
      object = query_cds,
      training_output = list(
        labels = training_list$labels,
        logits = log_odds,
        probs = prob_mat,
        history = export_list$history,
        duration = export_list$duration,
        params = export_list$params
      ),
      model_dir = model_dir
    ))
  }
}

#' Set up training, testing, and optional query datasets for model training
#'
#' This function prepares normalized and optionally scaled data matrices from reference and (optionally) query datasets. 
#' If a \code{query_cds} is provided, it identifies common features between the reference and query datasets, normalizes 
#' them, optionally performs TF-IDF or scaling, and returns both training/test splits from the reference and a query dataset. 
#' If no \code{query_cds} is provided, the function behaves like a reference-only setup, returning just training and 
#' testing splits.
#'
#' @param query_cds A \code{\link{Seurat}} or \code{\link{cell_data_set}} object representing the query dataset. 
#'   If \code{NULL}, no query data is processed.
#' @param ref_cds A \code{\link{Seurat}} or \code{\link{cell_data_set}} object representing the reference dataset.
#' @param ref_celldata_col A character string indicating the metadata column in \code{ref_cds} to use as labels.
#' @param norm_method A character string specifying the normalization method to use. One of \code{"log"}, 
#'   \code{"binary"}, \code{"size_only"}, or \code{"none"}.
#' @param selected_features A character vector of gene names to subset. If \code{NULL}, uses all common features 
#'   (if query is provided) or all selected features (if only reference is provided).
#' @param train_frac A numeric value between 0 and 1 indicating the fraction of reference cells to use for training. 
#'   The rest are used for testing.
#' @param tf_idf A logical indicating whether to perform TF-IDF normalization on the count matrices.
#' @param scale A logical indicating whether to scale the data. If \code{TRUE} and \code{tf_idf = TRUE}, 
#'   TF-IDF takes precedence and scaling is ignored.
#' @param LSImethod An integer specifying the TF-IDF method variant to use (passed to \code{\link{tf_idf_transform}}).
#' @param verbose A logical indicating whether to print progress messages.
#' @param addbias A logical indicating whether to add a bias row (ones) to the data matrices.
#' @param return_type A character string specifying the return format. One of \code{"sparse_mat"}, \code{"list"}, 
#'   \code{"matrix"}, or \code{"S4obj"}.
#' @param debug A logical indicating whether to print debugging messages.
#'
#' @details 
#' This function handles two scenarios:
#' \itemize{
#'   \item If \code{query_cds} is provided, it extracts common features between \code{ref_cds} and \code{query_cds}, 
#'   normalizes both datasets, performs optional TF-IDF or scaling, and returns training/testing splits from the reference 
#'   along with a query dataset.
#'   \item If \code{query_cds} is \code{NULL}, it behaves like a reference-only setup, returning just training and 
#'   testing splits from \code{ref_cds}.
#' }
#'
#' The returned object depends on \code{return_type}:
#' \itemize{
#'   \item \code{"sparse_mat"}: A list containing sparse matrices (\code{dgCMatrix}) for train/test/query, 
#'   labels (0-indexed integers), size factors, and metadata.
#'   \item \code{"matrix"}: A list of matrices containing \code{Xtrain_data}, \code{Xtest_data}, \code{Ytrain_label}, 
#'   \code{Ytest_label}, optionally \code{query}, as well as \code{label_text} and \code{features}.
#'   \item \code{"list"}: A list of lists, where each cell is represented as a \code{list} with \code{data} and 
#'   \code{target} elements.
#'   \item \code{"S4obj"}: A list of S4 objects \code{training_set}, \code{test_set}, and optionally \code{query_set}, 
#'   each containing training items and metadata.
#' }
#'
#' @return A list containing either matrices, lists of data items, or S4 objects, depending on \code{return_type}.
#'
#' @examples
#' \dontrun{
#' # Example with both reference and query data:
#' result <- setup_training(
#'   query_cds = query_seurat_obj,
#'   ref_cds = ref_seurat_obj,
#'   ref_celldata_col = "cell_type",
#'   norm_method = "log",
#'   train_frac = 0.8,
#'   tf_idf = TRUE,
#'   scale = FALSE,
#'   return_type = "sparse_mat"
#' )
#'
#' # Example with reference only:
#' result_ref <- setup_training(
#'   query_cds = NULL,
#'   ref_cds = ref_cds_obj,
#'   ref_celldata_col = "cell_type",
#'   norm_method = "none",
#'   train_frac = 0.7,
#'   scale = TRUE,
#'   return_type = "matrix"
#' )
#' }
#' @keywords internal
#' @export

setup_training <- function(query_cds = NULL, 
                           ref_cds, 
                           ref_celldata_col, 
                           norm_method = c("log", "binary", "size_only", "none"),
                           selected_features = NULL,
                           train_frac = 0.8,
                           tf_idf = FALSE,
                           scale = FALSE,
                           LSImethod = 1,
                           verbose = TRUE,
                           addbias = FALSE,
                           return_type = c("sparse_mat", "list", "matrix", "S4obj"),
                           debug = FALSE) {
  
  if (verbose) {
    message("Checking arguments and input")
  }
  
  norm_method <- match.arg(norm_method)
  return_type <- match.arg(return_type)
  
  # If both tf_idf and scale are TRUE, disable scale
  if (tf_idf & scale) {
    warning("Both tf_idf and scale selected. Using tf_idf alone.")
    scale <- FALSE
  }
  
  # Determine the software type and extract labels
  software <- NULL
  if (class(ref_cds) == "Seurat") {
    software <- "seurat"
    labf <- as.factor(ref_cds@meta.data[[ref_celldata_col]])
  } else if (class(ref_cds) == "cell_data_set") {
    software <- "monocle3"
    labf <- as.factor(colData(ref_cds)[[ref_celldata_col]])
  }
  if (is.null(software)) { 
    stop("Software not found for input objects") 
  }
  
  # ============================================================================
  # QUERY + REFERENCE MODE
  # ============================================================================
  if (!is.null(query_cds)) {
    if (class(query_cds) != class(ref_cds)) {
      stop("Input objects (ref_cds, query_cds) must be of the same class")
    }
    
    if (verbose) {
      message("Finding common features between reference and query")
    }
    common_list <- common_features(list(ref_cds, query_cds))
    names(common_list) <- c("ref", "query")
    
    # Select genes from common features
    if (is.null(selected_features)) {
      selected_common <- rownames(common_list[["query"]])
      selected_common <- selected_common[selected_common %in% rownames(common_list[["ref"]])]
    } else {
      if (verbose) {
        message("Subsetting by pre-selected features")
      }
      selected_common <- selected_features
      selected_common <- selected_common[selected_common %in% rownames(common_list[["query"]])]
      selected_common <- selected_common[selected_common %in% rownames(common_list[["ref"]])]
    }
    
    # Normalize counts
    if (norm_method != "none") {
      if (verbose) {
        message("Calculating normalized counts")
      }
      query <- get_norm_counts(common_list[["query"]], norm_method = norm_method)[selected_common, ]
      X <- get_norm_counts(common_list[["ref"]], norm_method = norm_method)[rownames(query), ]
    } else {
      query <- get_norm_counts(common_list[["query"]])[selected_common, ]
      X <- get_norm_counts(common_list[["ref"]])[rownames(query), ]
    }
    rm(common_list)
    gc()
    
    # Apply transformations
    if (tf_idf) {
      if (verbose) {
        message("Performing TF-IDF")
      }
      X <- tf_idf_transform(X, LSImethod)
      query <- tf_idf_transform(query, LSImethod)
    } else if (scale) {
      X <- scale(X)
      query <- scale(query)
    }
    
    # Prepare labels
    Ylab <- as.numeric(labf) - 1
    labels <- levels(labf)
    features <- rownames(X)
    
    # Split into train/test
    train_idx <- sample(1:ncol(X), round(train_frac * ncol(X)))
    test_idx <- which(!1:ncol(X) %in% train_idx)
    
    # --------------------------------------------------------------------------
    # Return based on type
    # --------------------------------------------------------------------------
    if (return_type == "sparse_mat") {
      X <- as(X, "dgCMatrix")
      query <- as(query, "dgCMatrix")
      
      # Data is already normalized by get_norm_counts()
      # Pass size_factor = 1.0 so Rust doesn't re-normalize
      n_ref <- ncol(X)
      n_query <- ncol(query)
      
      return(list(
        train_mat = X[, train_idx, drop = FALSE],
        train_labels = Ylab[train_idx],
        train_sf = rep(1.0, length(train_idx)),
        test_mat = X[, test_idx, drop = FALSE],
        test_labels = Ylab[test_idx],
        test_sf = rep(1.0, length(test_idx)),
        query_mat = query,
        query_sf = rep(1.0, n_query),
        labels = labels,
        features = features
      ))
      
    } else if (return_type == "matrix") {
      if (verbose) {
        message("Converting to dense matrix")
      }
      if (addbias) {
        X <- rbind(rep(1, ncol(X)), X)
        query <- rbind(rep(1, ncol(query)), query)
      }
      X <- as.matrix(X)
      query <- as.matrix(query)
      
      Y <- matrix(model.matrix(~0 + labf), ncol = length(labels))
      colnames(Y) <- NULL
      rownames(X) <- NULL
      colnames(X) <- NULL
      
      return(list(
        Xtrain_data = t(X[, train_idx]), 
        Xtest_data  = t(X[, test_idx]), 
        Ytrain_label = Y[train_idx, ], 
        Ytest_label = Y[test_idx, ],
        query = t(query),
        label_text = labels,
        features = features
      ))
      
    } else if (return_type == "list") {
      if (verbose) {
        message("Converting to dense matrix")
      }
      if (addbias) {
        X <- rbind(rep(1, ncol(X)), X)
        query <- rbind(rep(1, ncol(query)), query)
      }
      X <- as.matrix(X)
      query <- as.matrix(query)
      rownames(X) <- NULL
      colnames(X) <- NULL
      
      return(list(
        train = lapply(train_idx, function(idx) {
          list(data = X[, idx], target = Ylab[idx])
        }),
        test = lapply(test_idx, function(idx) {
          list(data = X[, idx], target = Ylab[idx])
        }),
        query = lapply(1:ncol(query), function(idx) {
          list(data = query[, idx])
        }),
        labels = labels,
        features = features
      ))
      
    } else if (return_type == "S4obj") {
      if (verbose) {
        message("Converting to dense matrix")
      }
      if (addbias) {
        X <- rbind(rep(1, ncol(X)), X)
        query <- rbind(rep(1, ncol(query)), query)
      }
      X <- as.matrix(X)
      query <- as.matrix(query)
      
      training_set <- new("training_set", 
                          name = "train", 
                          items = lapply(train_idx, function(idx) {
                            new("training_item", data = X[, idx], target = Ylab[idx])
                          }),
                          labels = labels,
                          features = features)
      test_set <- new("training_set", 
                      name = "test", 
                      items = lapply(test_idx, function(idx) {
                        new("training_item", data = X[, idx], target = Ylab[idx])
                      }),
                      labels = labels,
                      features = features)
      query_set <- new("training_set", 
                       name = "query", 
                       items = lapply(1:ncol(query), function(idx) {
                         new("training_item", data = query[, idx], target = 0)
                       }),
                       labels = "Unknown",
                       features = features)
      return(list(training_set, test_set, query_set))
    }
    
  } else {
    # ==========================================================================
    # REFERENCE ONLY MODE (query_cds is NULL)
    # ==========================================================================
    # Handle selected_features - if NULL, use all features
    if (is.null(selected_features)) {
      selected_features <- rownames(ref_cds)
      if (verbose) {
        message(sprintf("Using all %d features from reference", length(selected_features)))
      }
    }
    # Normalize counts
    if (norm_method != "none") {
      if (verbose) {
        message("Calculating normalized counts")
      }
      X <- get_norm_counts(ref_cds[selected_features, ], norm_method = norm_method)
    } else {
      X <- get_norm_counts(ref_cds[selected_features, ])
    }
    gc()
    
    # Apply transformations
    if (tf_idf) {
      if (verbose) {
        message("Performing TF-IDF")
      }
      X <- tf_idf_transform(X, LSImethod)
    } else if (scale) {
      X <- scale(X)
    }
    
    # Prepare labels
    Ylab <- as.numeric(labf) - 1
    labels <- levels(labf)
    features <- rownames(X)
    
    # Split into train/test
    train_idx <- sample(1:ncol(X), round(train_frac * ncol(X)))
    test_idx <- which(!1:ncol(X) %in% train_idx)
    
    # --------------------------------------------------------------------------
    # Return based on type
    # --------------------------------------------------------------------------
    if (return_type == "sparse_mat") {
      X <- as(X, "dgCMatrix")
      
      return(list(
        train_mat = X[, train_idx, drop = FALSE],
        train_labels = Ylab[train_idx],
        train_sf = rep(1.0, length(train_idx)),
        test_mat = X[, test_idx, drop = FALSE],
        test_labels = Ylab[test_idx],
        test_sf = rep(1.0, length(test_idx)),
        query_mat = NULL,
        query_sf = NULL,
        labels = labels,
        features = features
      ))
      
    } else if (return_type == "matrix") {
      if (verbose) {
        message("Converting to dense matrix")
      }
      if (addbias) {
        X <- rbind(rep(1, ncol(X)), X)
      }
      X <- as.matrix(X)
      
      Y <- matrix(model.matrix(~0 + labf), ncol = length(labels))
      colnames(Y) <- NULL
      rownames(X) <- NULL
      colnames(X) <- NULL
      
      return(list(
        Xtrain_data = t(X[, train_idx]), 
        Xtest_data  = t(X[, test_idx]), 
        Ytrain_label = Y[train_idx, ], 
        Ytest_label = Y[test_idx, ],
        query = NULL,
        label_text = labels,
        features = features
      ))
      
    } else if (return_type == "list") {
      if (verbose) {
        message("Converting to dense matrix")
      }
      if (addbias) {
        X <- rbind(rep(1, ncol(X)), X)
      }
      X <- as.matrix(X)
      rownames(X) <- NULL
      colnames(X) <- NULL
      
      return(list(
        train = lapply(train_idx, function(idx) {
          list(data = X[, idx], target = Ylab[idx])
        }),
        test = lapply(test_idx, function(idx) {
          list(data = X[, idx], target = Ylab[idx])
        }),
        query = NULL,
        labels = labels,
        features = features
      ))
      
    } else if (return_type == "S4obj") {
      if (verbose) {
        message("Converting to dense matrix")
      }
      if (addbias) {
        X <- rbind(rep(1, ncol(X)), X)
      }
      X <- as.matrix(X)
      
      training_set <- new("training_set",
                          name = "train",
                          items = lapply(train_idx, function(idx) {
                            new("training_item", data = X[, idx], target = Ylab[idx])
                          }),
                          labels = labels,
                          features = features)
      test_set <- new("training_set", 
                      name = "test",
                      items = lapply(test_idx, function(idx) {
                        new("training_item", data = X[, idx], target = Ylab[idx])
                      }),
                      labels = labels,
                      features = features)
      return(list(training_set, test_set, NULL))
    }
  }
}


#' Call the sparse training function in Rust
#' 
#' Internal wrapper to pass sparse matrices to Rust for MLR/ANN training.
#' 
#' @param model_type Character: "mlr", "ann", or "ann2"
#' @param train_mat dgCMatrix of training data (genes x cells)
#' @param train_labels Integer vector of 0-indexed labels
#' @param train_sf Numeric vector of size factors
#' @param test_mat dgCMatrix of test data
#' @param test_labels Integer vector of test labels
#' @param test_sf Numeric vector of test size factors
#' @param query_mat dgCMatrix of query data (or NULL)
#' @param query_sf Numeric vector of query size factors (or NULL)
#' @param labels Character vector of class names
#' @param feature_names Character vector of feature/gene names
#' @param hidden_size Integer vector of hidden layer sizes
#' @param learning_rate Numeric learning rate
#' @param num_epochs Integer number of epochs
#' @param directory Character path to model directory
#' @param verbose Logical for verbose output
#' @param backend Character: "nd", "wgpu", or "candle"
#' 
#' @return List with params, probs, history, duration
#' @keywords internal
#' @export
call_process_learning_obj_sparse <- function(
    model_type,
    train_mat,
    train_labels,
    train_sf,
    test_mat,
    test_labels,
    test_sf,
    query_mat = NULL,
    query_sf = NULL,
    labels,
    feature_names,
    hidden_size = NULL,
    learning_rate = 0.001,
    num_epochs = 10,
    directory = "/tmp/sc_local",
    verbose = TRUE,
    backend = "nd"
) {
  # Ensure dgCMatrix format
  train_mat <- as(train_mat, "dgCMatrix")
  test_mat <- as(test_mat, "dgCMatrix")
  
  # Handle query (may be NULL)
  if (!is.null(query_mat)) {
    query_mat <- as(query_mat, "dgCMatrix")
    query_x <- query_mat@x
    query_i <- query_mat@i
    query_p <- query_mat@p
    query_dims <- c(nrow(query_mat), ncol(query_mat))
  } else {
    query_x <- NULL
    query_i <- NULL
    query_p <- NULL
    query_dims <- NULL
    query_sf <- NULL
  }
  
  # Call Rust function
  viewmastR:::process_learning_obj_sparse(
    model_type = model_type,
    # Train
    train_x = train_mat@x,
    train_i = train_mat@i,
    train_p = train_mat@p,
    train_dims = c(nrow(train_mat), ncol(train_mat)),
    train_labels = as.integer(train_labels),
    train_size_factors = train_sf,
    # Test
    test_x = test_mat@x,
    test_i = test_mat@i,
    test_p = test_mat@p,
    test_dims = c(nrow(test_mat), ncol(test_mat)),
    test_labels = as.integer(test_labels),
    test_size_factors = test_sf,
    # Query
    query_x = query_x,
    query_i = query_i,
    query_p = query_p,
    query_dims = query_dims,
    query_size_factors = query_sf,
    # Metadata
    labels = labels,
    feature_names = feature_names,
    # Model params
    hidden_size = hidden_size,
    learning_rate = learning_rate,
    num_epochs = num_epochs,
    directory = directory,
    verbose = verbose,
    backend = backend
  )
}


# ==============================================================================
# Helper functions (unchanged)
# ==============================================================================

seurat_size_factors <- function(cds, round_exprs = TRUE, 
                                 method = c("mean-geometric-mean-total", 
                                            "mean-geometric-mean-log-total")) {
  method <- match.arg(method)
  mat <- get_counts_seurat(cds)
  if (any(Matrix::colSums(mat) == 0)) {
    warning("Your CDS object contains cells with zero reads. ", 
            "This causes size factor calculation to fail. Please remove ", 
            "the zero read cells using ", 
            "cds <- cds[,Matrix::colSums(exprs(cds)) != 0] and then ", 
            "run cds <- estimate_size_factors(cds)")
    return(cds)
  }
  if (is_sparse_matrix(mat)) {
    sf <- monocle3:::estimate_sf_sparse(mat, round_exprs = round_exprs, method = method)
  } else {
    sf <- monocle3:::estimate_sf_dense(mat, round_exprs = round_exprs, method = method)
  }
  return(sf)
}

is_sparse_matrix <- function(x) {
  class(x) %in% c("dgCMatrix", "dgTMatrix", "lgCMatrix")
}


get_norm_counts <- function(cds, norm_method = c("log", "binary", "size_only", "none"), 
                            pseudocount = 1) {
  software <- NULL
  norm_method <- match.arg(norm_method)
  if (class(cds) == "Seurat") { software <- "seurat" }
  if (class(cds) == "cell_data_set") { software <- "monocle3" }
  if (is.null(software)) { stop("Software not found for input") }
  
  if (software == "monocle3") {
    norm_mat <- SingleCellExperiment::counts(cds) %>% Seurat::as.sparse()
    if (norm_method == "none") {
      return(norm_mat)
    }
    sf <- size_factors(cds)
  }
  if (software == "seurat") {
    norm_mat <- get_counts_seurat(cds)
    if (norm_method == "none") {
      return(norm_mat)
    }
    sf <- seurat_size_factors(cds)
  }
  
  if (norm_method == "binary") {
    norm_mat <- norm_mat > 0
    if (is_sparse_matrix(norm_mat)) {
      norm_mat <- methods::as(norm_mat, "dgCMatrix")
    }
  } else {
    if (is_sparse_matrix(norm_mat)) {
      norm_mat@x <- norm_mat@x / rep.int(sf, diff(norm_mat@p))
      if (norm_method == "log") {
        if (pseudocount == 1) {
          norm_mat@x <- log10(norm_mat@x + pseudocount)
        } else {
          stop("Pseudocount must equal 1 with sparse expression matrices")
        }
      }
    } else {
      norm_mat <- Matrix::t(Matrix::t(norm_mat) / sf)
      if (norm_method == "log") {
        norm_mat@x <- log10(norm_mat + pseudocount)
      }
    }
  }
  return(norm_mat)
}


get_seurat_version <- function(obj) {
  return(obj@version)
}

get_counts_seurat <- function(cds, assay = "RNA") {
  sversion <- packageVersion("Seurat")
  oversion <- tryCatch(get_seurat_version(cds), error = function(e) NULL)
  
  # Scenario: User has v4 installed, but Object is v5
  if (!is.null(oversion) && 
      compareVersion(as.character(sversion), "5.0.0") == -1 && 
      !compareVersion(as.character(oversion), "5.0.0") == -1) {
    warning(sprintf(
      "Version Mismatch: System has Seurat v%s, but object is v%s. \n",
      "Seurat v4 may not read v5 objects with 'layers'. Result may be empty.", 
      sversion, oversion
    ))
  }
  
  if (!compareVersion(as.character(sversion), "5.0.0") == -1) {
    # Seurat v5+ uses 'layer'
    mat <- Seurat::GetAssayData(object = cds, assay = assay, layer = "counts")
    if (inherits(mat, "IterableMatrix")) {
      mat <- as(mat, "dgCMatrix")
    }
    return(mat)
  } else if (compareVersion(as.character(sversion), "2.9.9") == 1 && 
             compareVersion(as.character(sversion), "5.0.0") == -1) {
    # Seurat v3/v4 uses 'slot'
    mat <- Seurat::GetAssayData(object = cds, assay = assay, slot = "counts")
    if (inherits(mat, "IterableMatrix")) {
      mat <- as(mat, "dgCMatrix")
    }
    return(mat)
  } else {
    warning("Seurat version is too old for this object. Please upgrade to Seurat v3 or higher.")
    return(NULL)
  }
} 

#' Helper function for summing sparse matrix groups
#' @references Granja, J. M.et al. (2019). Single-cell multiomic analysis identifies regulatory programs in mixed-phenotype 
#' acute leukemia. Nature Biotechnology, 37(12), 1458–1465.
#' @export
#' @keywords internal
sparseRowVariances <- function(m) {
  rM <- Matrix::rowMeans(m)
  rV <- computeSparseRowVariances(m@i + 1, m@x, rM, ncol(m))
  return(rV)
}


# ################################################################################
# # FILE: R/training.R
# # STATUS: Clean
# # ------------------------------------------------------------------------------
# # Functions:
# # [ ] dimension_check           (Not Exported)
# # [x] optimize_backend          (Exported - Internal)
# # [x] viewmastR                 (Exported)
# # [x] setup_training            (Exported - Internal)
# # [ ] seurat_size_factors       (Not Exported)
# # [ ] is_sparse_matrix          (Not Exported)
# # [ ] get_norm_counts           (Not Exported)
# # [ ] get_seurat_version        (Not Exported)
# # [ ] get_counts_seurat         (Not Exported)
# # [x] sparseRowVariances        (Exported - Internal)
# ################################################################################


# setClass("training_item", slots=c(data="numeric", target="numeric"))
# setClass("training_set", slots=c(name="character", items="list", labels="character", features="character"))

# dimension_check <- function (obj){
#   nm <-deparse(substitute(obj))
#   if(is.null(dim(obj))){
#     if(class(obj)=="list"){
#       llen <- sapply(obj, length)
#       if(!is.null(var(llen))){
#         if (var(llen)==0){
#           paste0(nm, "-dim1: ", llen[1], " non-ragged")
#         } else {
#           paste0(nm, "-dim1: ", mean(llen), " (mean) ragged")
#         }
#       } else {
#         message("not numeric")
#       }
#     } else {
#       paste0(nm, "-dim1: ", length(obj))
#     }
#   } else {
#     nd <- length(dim(obj))
#     dv <- paste0("-dim", 1:nd, ": ")
#     paste0(nm, paste0(dv, dim(obj)))
#   }
# }

# #' @keywords internal
# optimize_backend <- function(){
#   if(Sys.info()["machine"]=="arm64" & Sys.info()['sysname'] == "Darwin"){
#     return("nd")
#   }
#   if(Sys.info()["machine"]=="x86_64" & Sys.info()['sysname'] == "Linux"){
#     return("candle")
#   }
#   else {
#     return("wgpu")
#   }
# }

# #' Integrate and Train Models on Reference Dataset and (Optionally) Infer on Query Datasets
# #'
# #' The \code{viewmastR} function preprocesses one or two single-cell datasets (a reference and an optional query), 
# #' splits the reference data into training and test sets, and optionally includes the ability to run inference on a query dataset 
# #' for downstream analysis. It then applies specified modeling functions (e.g., MLR, NN, NB) to train and optionally predict on the 
# #' query data.
# #'
# #' @param query_cds A \code{Seurat} or \code{cell_data_set} object representing the query dataset. If \code{NULL}, 
# #'   the function will operate in "reference-only" mode, using the reference dataset for training and testing only.
# #' @param ref_cds A \code{Seurat} or \code{cell_data_set} object representing the reference dataset. This is required.
# #' @param ref_celldata_col A character string specifying the metadata column in \code{ref_cds} that contains the cell labels.
# #' @param query_celldata_col A character string specifying a metadata column name in \code{query_cds} (or reference in 
# #'   reference-only mode) where predicted labels should be stored. If \code{NULL}, defaults to \code{"viewmastR_pred"}.
# #' @param FUNC A character string specifying the modeling function to apply. One of \code{"mlr"}, \code{"nn"}, or \code{"nb"}.
# #' @param norm_method Character string indicating the normalization method. One of \code{"log"}, \code{"binary"}, 
# #'   \code{"size_only"}, or \code{"none"}.
# #' @param selected_features A character vector specifying genes to subset. If \code{NULL}, uses the set of common features 
# #'   (if query is provided) or selected genes directly (if reference-only).
# #' @param train_frac A numeric value between 0 and 1 specifying the fraction of reference cells to use for training. 
# #'   The remainder are used for testing.
# #' @param tf_idf Logical, whether to apply TF-IDF transformation after normalization.
# #' @param scale Logical, whether to scale the data. If both \code{tf_idf} and \code{scale} are \code{TRUE}, TF-IDF takes precedence.
# #' @param hidden_layers A numeric vector indicating the size of hidden layers (for the NN model). Only 1 or 2 layers are allowed.
# #' @param learning_rate Numeric, learning rate for model training.
# #' @param max_epochs Integer, the maximum number of epochs for model training.
# #' @param LSImethod Integer, specifying the TF-IDF method variant if using TF-IDF.
# #' @param verbose Logical, whether to print progress messages.
# #' @param backend A character string specifying the backend to use. One of \code{"wgpu"}, \code{"nd"}, \code{"candle"}.
# #' @param threshold Currently unused. Can be \code{NULL}.
# #' @param keras_model Currently unused. Can be \code{NULL}.
# #' @param model_dir A character string specifying the directory to store model artifacts.
# #' @param return_probs Logical, whether to return predicted probabilities in the object's metadata.
# #' @param return_type A character string specifying the return type. One of \code{"object"} or \code{"list"}. 
# #'   If \code{"object"}, returns the updated \code{query_cds}. If \code{"list"}, returns a list containing 
# #'   \code{object} and \code{training_output}.
# #' @param debug Logical, whether to print debugging messages and dimension checks.
# #' @param train_only Logical, if \code{TRUE}, only the reference data is processed and no query data is included.
# #'
# #' @details 
# #' The function first calls \code{\link{setup_training}} to preprocess and split the data into training, testing, and 
# #' optionally query subsets. Then, based on the selected \code{FUNC}, it calls one of the model training and prediction 
# #' functions (\code{process_learning_obj_mlr}, \code{process_learning_obj_ann}, \code{process_learning_obj_nb}). 
# #' If \code{train_only = TRUE}, the query portion is skipped and no query predictions are made.
# #'
# #' For \code{"mlr"} and \code{"nn"} functions, predicted log odds are converted to probabilities using the logistic function. 
# #' Predicted cell labels are assigned to the \code{query_cds} (or \code{ref_cds} if query is not provided).
# #'
# #' @return 
# #' Depending on \code{return_type}, returns either:
# #' \itemize{
# #'   \item \code{return_type = "object"}: the input \code{query_cds} (or \code{ref_cds} if \code{query_cds = NULL}) with predicted 
# #'   labels (and optionally probabilities) appended.
# #'   \item \code{return_type = "list"}: a list containing:
# #'   \describe{
# #'     \item{object}{The updated \code{query_cds} (or \code{ref_cds}).}
# #'     \item{training_output}{The output from the model training process, including probabilities if applicable.}
# #'   }
# #' }
# #'
# #' @examples
# #' \dontrun{
# #' # Training and predicting with reference and query data:
# #' res <- viewmastR(
# #'   query_cds = query_seurat_obj,
# #'   ref_cds = ref_seurat_obj,
# #'   ref_celldata_col = "cell_type",
# #'   FUNC = "mlr",
# #'   norm_method = "log",
# #'   train_frac = 0.8,
# #'   backend = "wgpu",
# #'   verbose = TRUE,
# #'   return_type = "object"
# #' )
# #'
# #' # Reference-only scenario:
# #' res_ref <- viewmastR(
# #'   query_cds = NULL,
# #'   ref_cds = ref_cds_obj,
# #'   ref_celldata_col = "cell_type",
# #'   FUNC = "nn",
# #'   norm_method = "none",
# #'   train_frac = 0.7,
# #'   scale = TRUE,
# #'   train_only = TRUE,
# #'   return_type = "list"
# #' )
# #' }
# #'
# #' @export

# viewmastR <- function(query_cds, 
#                       ref_cds, 
#                       ref_celldata_col, 
#                       query_celldata_col = NULL, 
#                       FUNC = c("mlr", "nn", "nb"),
#                       norm_method = c("log", "binary", "size_only", "none"),
#                       selected_features = NULL,
#                       train_frac = 0.8,
#                       tf_idf = FALSE,
#                       scale = FALSE,
#                       hidden_layers = c(as.integer(500),as.integer(100)),
#                       learning_rate = 1e-3,
#                       max_epochs = 5,
#                       LSImethod = 1,
#                       verbose = TRUE,
#                       backend = c("auto", "wgpu", "nd", "candle"),
#                       threshold = NULL,
#                       keras_model = NULL, 
#                       model_dir = "/tmp/sc_local",
#                       return_probs = FALSE,
#                       return_type = c("object", "list"), 
#                       debug = FALSE,
#                       train_only = FALSE
#                       ) {
#   FUNC <- match.arg(FUNC)
#   return_type <- match.arg(return_type)
#   backend <- match.arg(backend)
#   if(backend=="auto"){
#     backend <- optimize_backend()
#   }
#   if(FUNC=="nn"){
#     if(length(hidden_layers)==2){
#       model = "ann2"
#     } else if(length(hidden_layers)==1){
#       model = "ann"
#     } else {
#       stop("Number of hidden layers can only be 1 or 2")
#     }
#   }
#   if(FUNC=="mlr"){
#     model = "mlr"
#   }
#   norm_method <- match.arg(norm_method)

#   ## TRAIN ONLY MODE
#   if (train_only) {
#     if(FUNC == "nb") {
#       message("naive bayes is currently not implemented for training only")
#       return()
#     }
#     # Use setup_training with query_cds = NULL
#     training_list <- setup_training(
#       query_cds = NULL,
#       ref_cds = ref_cds,
#       ref_celldata_col = ref_celldata_col,
#       norm_method = norm_method,
#       selected_features = selected_features,
#       train_frac = train_frac,
#       tf_idf = tf_idf,
#       scale = scale,
#       LSImethod = LSImethod,
#       verbose = verbose,
#       addbias = F,
#       return_type = "sparse_mat",
#       debug = debug
#     )

#     # if (debug) {
#     #   message("Dimension check (Training Only):")
#     #   message(paste0("\t", dimension_check(training_list[["train"]][[1]])))
#     #   message(paste0("\t", dimension_check(training_list[["test"]][[1]])))
#     #   message(paste0("\t", dimension_check(training_list[["labels"]])))
#     # }

#     if(!file.exists(model_dir)) {
#       dir.create(model_dir)
#     }

#     if(is.null(query_celldata_col)) {
#       query_celldata_col <- "viewmastR_pred"
#     }

#     export_list <- call_process_learning_obj_sparse(
#       model_type = model,
#       train_mat = training_list$train_mat,
#       train_labels = training_list$train_labels,
#       train_sf = training_list$train_sf,
#       test_mat = training_list$test_mat,
#       test_labels = training_list$test_labels,
#       test_sf = training_list$test_sf,
#       query_mat = training_list$query_mat,
#       query_sf = training_list$query_sf,
#       labels = training_list$labels,
#       feature_names = training_list$features,
#       hidden_size = as.integer(hidden_layers),
#       learning_rate = learning_rate,
#       num_epochs = max_epochs,
#       directory = model_dir,
#       verbose = verbose,
#       backend = backend
#     )
#     # export_list <- process_learning_obj_sparse(
#     #       model,
#     #       train = training_list[["train"]],
#     #       test = training_list[["test"]],
#     #       query = training_list[["query"]],
#     #       labels = training_list[["labels"]],
#     #       feature_names = training_list[["features"]],
#     #       hidden_size = as.integer(hidden_layers),
#     #       learning_rate = learning_rate,
#     #       num_epochs = max_epochs,
#     #       directory = model_dir,
#     #       verbose = verbose,
#     #       backend = backend
#     #     )
#     return(list(object=NULL, training_output = list(history = export_list$history, duration = export_list$duration, params = export_list$params), model_dir = model_dir))
#   } else {
#     # When train_only = FALSE, we have query_cds provided
#     training_list <- setup_training(
#       query_cds = query_cds,
#       ref_cds = ref_cds,
#       ref_celldata_col = ref_celldata_col,
#       norm_method = norm_method,
#       selected_features = selected_features,
#       train_frac = train_frac,
#       tf_idf = tf_idf,
#       scale = scale,
#       LSImethod = LSImethod,
#       verbose = verbose,
#       addbias = F,
#       return_type = "sparse_mat",
#       debug = debug
#     )

#     # if (debug) {
#     #   message("Dimension check:")
#     #   message(paste0("\t", dimension_check(training_list[["train"]][[1]])))
#     #   message(paste0("\t", dimension_check(training_list[["test"]][[1]])))
#     #   message(paste0("\t", dimension_check(training_list[["query"]][[1]])))
#     #   message(paste0("\t", dimension_check(training_list[["labels"]])))
#     # }

#     if(!file.exists(model_dir)) {
#       dir.create(model_dir)
#     }

#     if(is.null(query_celldata_col)) {
#       query_celldata_col <- "viewmastR_pred"
#     }
#     if(FUNC == "nb") {
#       export_list <- process_learning_obj_nb(
#         train = training_list[["train"]],
#         test = training_list[["test"]],
#         query = training_list[["query"]])
#       if(return_type == "probs") {
#         message("probabilities from multinomial naive bayes not implemented yet")
#       }
#       query_cds[[query_celldata_col]] <- training_list[["labels"]][export_list$predictions[[1]]+1]
#       if (return_type=="object") {
#         return(query_cds)
#       } else {
#         return(list(object=query_cds, training_output = list(probs = prob_mat, history = export_list$history, duration = export_list$duration, params = export_list$params)))
#       }
#     } else {
#       # export_list <- process_learning_obj(
#       #   model,
#       #   train = training_list[["train"]],
#       #   test = training_list[["test"]],
#       #   query = training_list[["query"]],
#       #   hidden_size = as.integer(hidden_layers),
#       #   labels = training_list[["labels"]],
#       #   feature_names = training_list[["features"]],
#       #   learning_rate = learning_rate,
#       #   num_epochs = max_epochs,
#       #   directory = model_dir,
#       #   verbose = verbose,
#       #   backend = backend
#       # )
#       export_list <- call_process_learning_obj_sparse(
#         model_type = model,
#         train_mat = training_list$train_mat,
#         train_labels = training_list$train_labels,
#         train_sf = training_list$train_sf,
#         test_mat = training_list$test_mat,
#         test_labels = training_list$test_labels,
#         test_sf = training_list$test_sf,
#         query_mat = training_list$query_mat,
#         query_sf = training_list$query_sf,
#         labels = training_list$labels,
#         feature_names = training_list$features,
#         hidden_size = as.integer(hidden_layers),
#         learning_rate = learning_rate,
#         num_epochs = max_epochs,
#         directory = model_dir,
#         verbose = verbose,
#         backend = backend
#       )
#     log_odds <- unlist(export_list$probs[[1]])
#     # Check if log_odds has the expected dimensions
#     if(length(log_odds) == dim(query_cds)[2] * length(training_list[["labels"]])){
#       log_odds = matrix(log_odds, ncol = dim(query_cds)[2])
#       log_odds = t(log_odds)
#       colnames(log_odds) <- paste0("prob_", training_list[["labels"]])
#     } else {
#       stop("Error in log odds dimensions of function output")
#     }
    
#     softmax_rows <- function(mat) {
#       shifted <- mat - apply(mat, 1, max)  # stability
#       exp_shifted <- exp(shifted)
#       exp_shifted / rowSums(exp_shifted)
#     }
    
#     prob_mat  <- softmax_rows(log_odds)
#     query_cds[[query_celldata_col]]<-training_list[["labels"]][apply(prob_mat, 1, which.max)]
    
#     if(return_probs){
#       query_cds@meta.data <- cbind(query_cds@meta.data, prob_mat)
#     }
#     if (return_type=="object") {
#       return(query_cds)
#     } else {
#       return(list(object=query_cds, training_output = list(probs = prob_mat, history = export_list$history, duration = export_list$duration, params = export_list$params), model_dir = model_dir))
#     }
#     }
#   }
# }

# #' Set up training, testing, and optional query datasets for model training
# #'
# #' This function prepares normalized and optionally scaled data matrices from reference and (optionally) query datasets. 
# #' If a \code{query_cds} is provided, it identifies common features between the reference and query datasets, normalizes 
# #' them, optionally performs TF-IDF or scaling, and returns both training/test splits from the reference and a query dataset. 
# #' If no \code{query_cds} is provided, the function behaves like a reference-only setup, returning just training and 
# #' testing splits.
# #'
# #' @param query_cds A \code{\link{Seurat}} or \code{\link{cell_data_set}} object representing the query dataset. 
# #'   If \code{NULL}, no query data is processed.
# #' @param ref_cds A \code{\link{Seurat}} or \code{\link{cell_data_set}} object representing the reference dataset.
# #' @param ref_celldata_col A character string indicating the metadata column in \code{ref_cds} to use as labels.
# #' @param norm_method A character string specifying the normalization method to use. One of \code{"log"}, 
# #'   \code{"binary"}, \code{"size_only"}, or \code{"none"}.
# #' @param selected_features A character vector of gene names to subset. If \code{NULL}, uses all common features 
# #'   (if query is provided) or all selected features (if only reference is provided).
# #' @param train_frac A numeric value between 0 and 1 indicating the fraction of reference cells to use for training. 
# #'   The rest are used for testing.
# #' @param tf_idf A logical indicating whether to perform TF-IDF normalization on the count matrices.
# #' @param scale A logical indicating whether to scale the data. If \code{TRUE} and \code{tf_idf = TRUE}, 
# #'   TF-IDF takes precedence and scaling is ignored.
# #' @param LSImethod An integer specifying the TF-IDF method variant to use (passed to \code{\link{tf_idf_transform}}).
# #' @param verbose A logical indicating whether to print progress messages.
# #' @param addbias A logical indicating whether to add a bias row (ones) to the data matrices.
# #' @param return_type A character string specifying the return format. One of \code{"list"}, \code{"matrix"}, or \code{"S4obj"}.
# #' @param debug A logical indicating whether to print debugging messages.
# #'
# #' @details 
# #' This function handles two scenarios:
# #' \itemize{
# #'   \item If \code{query_cds} is provided, it extracts common features between \code{ref_cds} and \code{query_cds}, 
# #'   normalizes both datasets, performs optional TF-IDF or scaling, and returns training/testing splits from the reference 
# #'   along with a query dataset.
# #'   \item If \code{query_cds} is \code{NULL}, it behaves like a reference-only setup, returning just training and 
# #'   testing splits from \code{ref_cds}.
# #' }
# #'
# #' The returned object depends on \code{return_type}:
# #' \itemize{
# #'   \item \code{"matrix"}: A list of matrices containing \code{Xtrain_data}, \code{Xtest_data}, \code{Ytrain_label}, \code{Ytest_label}, 
# #'   optionally \code{query}, as well as \code{label_text} and \code{features}.
# #'   \item \code{"list"}: A list of lists, where each cell is represented as a \code{list} with \code{data} and \code{target} elements.
# #'   \item \code{"S4obj"}: A list of S4 objects \code{training_set}, \code{test_set}, and optionally \code{query_set}, 
# #'   each containing training items and metadata.
# #' }
# #'
# #' @return A list containing either matrices, lists of data items, or S4 objects, depending on \code{return_type}.
# #'
# #' @examples
# #' \dontrun{
# #' # Example with both reference and query data:
# #' result <- setup_training(
# #'   query_cds = query_seurat_obj,
# #'   ref_cds = ref_seurat_obj,
# #'   ref_celldata_col = "cell_type",
# #'   norm_method = "log",
# #'   train_frac = 0.8,
# #'   tf_idf = TRUE,
# #'   scale = FALSE,
# #'   return_type = "list"
# #' )
# #'
# #' # Example with reference only:
# #' result_ref <- setup_training(
# #'   query_cds = NULL,
# #'   ref_cds = ref_cds_obj,
# #'   ref_celldata_col = "cell_type",
# #'   norm_method = "none",
# #'   train_frac = 0.7,
# #'   scale = TRUE,
# #'   return_type = "matrix"
# #' )
# #' }
# #' @keywords internal
# #' @export

# setup_training <- function(query_cds = NULL, 
#                            ref_cds, 
#                            ref_celldata_col, 
#                            norm_method = c("log", "binary", "size_only", "none"),
#                            selected_features = NULL,
#                            train_frac = 0.8,
#                            tf_idf = FALSE,
#                            scale = FALSE,
#                            LSImethod = 1,
#                            verbose = TRUE,
#                            addbias = FALSE,
#                            return_type = c("sparse_mat", "list", "matrix", "S4obj"),
#                            debug = FALSE) {
  
#   if (verbose) {
#     message("Checking arguments and input")
#   }
  
#   norm_method <- match.arg(norm_method)
#   return_type <- match.arg(return_type)
  
#   # If both tf_idf and scale are TRUE, disable scale
#   if (tf_idf & scale) {
#     warning("Both tf_idf and scale selected. Using tf_idf alone.")
#     scale <- FALSE
#   }
  
#   # Determine the software type and extract labels
#   software <- NULL
#   if (class(ref_cds) == "Seurat") {
#     software <- "seurat"
#     labf <- as.factor(ref_cds@meta.data[[ref_celldata_col]])
#   } else if (class(ref_cds) == "cell_data_set") {
#     software <- "monocle3"
#     labf <- as.factor(colData(ref_cds)[[ref_celldata_col]])
#   }
#   if (is.null(software)) { stop("software not found for input objects") }
  
#   # If query_cds is provided, find common features, otherwise use ref only
#   if (!is.null(query_cds)) {
#     if (class(query_cds) != class(ref_cds)) {
#       stop("Input objects (ref_cds, query_cds) must be of the same class")
#     }
#     if (verbose) {
#       message("Finding common features between reference and query")
#     }
#     common_list <- common_features(list(ref_cds, query_cds))
#     names(common_list) <- c("ref", "query")
    
#     # Select genes from common features
#     if (is.null(selected_features)) {
#       # All common features
#       selected_common <- rownames(common_list[["query"]])
#       selected_common <- selected_common[selected_common %in% rownames(common_list[["ref"]])]
#     } else {
#       if (verbose) {
#         message("Subsetting by pre-selected features")
#       }
#       selected_common <- selected_features
#       selected_common <- selected_common[selected_common %in% rownames(common_list[["query"]])]
#       selected_common <- selected_common[selected_common %in% rownames(common_list[["ref"]])]
#     }
    
#     # Normalize counts
#     if (norm_method != "none") {
#       if (verbose) {
#         message("Calculating normalized counts")
#       }
#       query <- get_norm_counts(common_list[["query"]], norm_method = norm_method)[selected_common, ]
#       X <- get_norm_counts(common_list[["ref"]], norm_method = norm_method)[rownames(query), ]
#     } else {
#       query <- get_norm_counts(common_list[["query"]])[selected_common, ]
#       X <- get_norm_counts(common_list[["ref"]])[rownames(query), ]
#     }
#     rm(common_list)
#     gc()
    
#     # Perform scaling methods
#     if (tf_idf) {
#       if (verbose) {
#         message("Performing TF-IDF")
#       }
#       X <- tf_idf_transform(X, LSImethod)
#       query <- tf_idf_transform(query, LSImethod)
#     } else {
#       if (scale) {
#         X <- scale(X)
#         query <- scale(query)
#       }
#     }
    
#     if(return_type != "sparse_mat"){
#       if (verbose) {
#         message("Converting to dense matrix :(")
#       }
#       if (addbias) {
#         if (verbose) {
#           message("Adding bias")
#         }
#         X <- rbind(rep(1, ncol(X)), X)
#         query <- rbind(rep(1, ncol(query)), query)
#       }
#       X <- as.matrix(X)
#       query <- as.matrix(query)
#     }


    
#     Ylab <- as.numeric(labf) - 1
#     labels <- levels(labf)
#     Y <- matrix(model.matrix(~0+labf), ncol = length(labels))
#     colnames(Y) <- NULL
#     features <- rownames(X)
#     rownames(X) <- NULL
#     colnames(X) <- NULL
    
#     # Split into train/test
#     train_idx <- sample(1:dim(X)[2], round(train_frac * dim(X)[2]))
#     test_idx <- which(!1:dim(X)[2] %in% train_idx)
    
#     # Return based on type
#     if (return_type =="matrix") {
#       return(list(
#         Xtrain_data = t(X[, train_idx]), 
#         Xtest_data  = t(X[, test_idx]), 
#         Ytrain_label = Y[train_idx, ], 
#         Ytest_label = Y[test_idx, ],
#         query = t(query),
#         label_text = labels,
#         features = features
#       ))
#     } else if (return_type == "list") {
#       return(list(
#         train = lapply(train_idx, function(idx) {
#           list(data = t(X[, idx])[1, ], target = Ylab[idx])
#         }),
#         test = lapply(test_idx, function(idx) {
#           list(data = t(X[, idx])[1, ], target = Ylab[idx])
#         }),
#         query = lapply(1:dim(query)[2], function(idx) {
#           list(data = t(query[, idx])[1, ])
#         }),
#         labels = labels,
#         features = features
#       ))
#     } else if (return_type == "S4obj") {
#       training_set <- new("training_set", 
#                           name = "train", 
#                           items = lapply(train_idx, function(idx) {
#                             new("training_item", data = t(X[, idx])[1, ], target = Ylab[idx])
#                           }),
#                           labels = labels,
#                           features = features)
#       test_set <- new("training_set", 
#                       name = "test", 
#                       items = lapply(test_idx, function(idx) {
#                         new("training_item", data = t(X[, idx])[1, ], target = Ylab[idx])
#                       }),
#                       labels = labels,
#                       features = features)
#       query_set <- new("training_set", 
#                        name = "query", 
#                        items = lapply(1:dim(query)[2], function(idx) {
#                          new("training_item", data = t(query[, idx])[1, ], target = 0)
#                        }),
#                        labels = "Unknown",
#                        features = features)
#       return(list(training_set, test_set, query_set))
#     ## new return type for sparse matrix handling in rust
#     } else if (return_type =="sparse_mat") {
#         X <- as(X, "dgCMatrix")
#         query <- as(query, "dgCMatrix")
#       return(list(
#         train_mat = X[, train_idx],      # dgCMatrix (genes × cells)
#         train_labels = Ylab[train_idx],  # integer vector
#         test_mat = X[, test_idx],
#         test_labels = Ylab[test_idx],
#         query_mat = query,
#         labels = labels,                  # class names
#         features = features
#         ))
#     } 
    
#   } else {
#     # Behaves like setup_training_ref if query_cds is NULL
#     # Ensure selected_features are present
#     if (norm_method != "none") {
#       if (verbose) {
#         message("Calculating normalized counts")
#       }
#       X <- get_norm_counts(ref_cds[selected_features, ], norm_method = norm_method)
#     } else {
#       X <- get_norm_counts(ref_cds[selected_features, ])
#     }
    
#     gc()
    
#     if (tf_idf) {
#       if (verbose) {
#         message("Performing TF-IDF")
#       }
#       X <- tf_idf_transform(X, LSImethod)
#     } else {
#       if (scale) {
#         X <- scale(X)
#       }
#     }
    
#     if (verbose) {
#       message("Converting to dense matrix :(")
#     }
#     if (addbias) {
#       if (verbose) {
#         message("Adding bias")
#       }
#       # In the ref-only scenario, originally code referred to Xtrain; corrected to X
#       X <- rbind(rep(1, ncol(X)), X)
#     }
#     X <- as.matrix(X)
#     gc()
    
#     Ylab <- as.numeric(labf) - 1
#     labels <- levels(labf)
#     Y <- matrix(model.matrix(~0+labf), ncol = length(labels))
#     colnames(Y) <- NULL
#     features <- rownames(X)
#     rownames(X) <- NULL
#     colnames(X) <- NULL
    
#     train_idx <- sample(1:dim(X)[2], round(train_frac*dim(X)[2]))
#     test_idx <- which(!1:dim(X)[2] %in% train_idx)
    
#     if (return_type == "matrix") {
#       return(list(
#         Xtrain_data = t(X[, train_idx]), 
#         Xtest_data  = t(X[, test_idx]), 
#         Ytrain_label = Y[train_idx, ], 
#         Ytest_label = Y[test_idx, ],
#         query = NULL,
#         label_text = labels,
#         features = features
#       ))
#     } else if (return_type == "list") {
#       return(list(
#         train = lapply(train_idx, function(idx) {
#           list(data = t(X[, idx])[1, ], target = Ylab[idx])
#         }),
#         test = lapply(test_idx, function(idx) {
#           list(data = t(X[, idx])[1, ], target = Ylab[idx])
#         }),
#         labels = labels,
#         features = features
#       ))
#     } else if (return_type == "S4obj") {
#       training_set <- new("training_set",
#                           name = "train",
#                           items = lapply(train_idx, function(idx) {
#                             new("training_item", data = t(X[, idx])[1, ], target = Ylab[idx])
#                           }),
#                           labels = labels,
#                           features = features)
#       test_set <- new("training_set", 
#                       name = "test",
#                       items = lapply(test_idx, function(idx) {
#                         new("training_item", data = t(X[, idx])[1, ], target = Ylab[idx])
#                       }),
#                       labels = labels,
#                       features = features)
#       query_set <- NULL
#       return(list(training_set, test_set, query_set))
#     } else if (return_type =="sparse_mat") {
#         X <- as(X, "dgCMatrix")
#       return(list(
#         train_mat = X[, train_idx],      # dgCMatrix (genes × cells)
#         train_labels = Ylab[train_idx],  # integer vector
#         test_mat = X[, test_idx],
#         test_labels = Ylab[test_idx],
#         query_mat = NULL,
#         labels = labels,                  # class names
#         features = features
#         ))
#     } 
#   }
# }

# #' @keywords internal
# call_process_learning_obj_sparse <- function(
#     model_type,
#     train_mat,
#     train_labels,
#     train_sf,
#     test_mat,
#     test_labels,
#     test_sf,
#     query_mat = NULL,
#     query_sf = NULL,
#     labels,
#     feature_names,
#     hidden_size = NULL,
#     learning_rate = 0.001,
#     num_epochs = 10,
#     directory = "/tmp/sc_local",
#     verbose = TRUE,
#     backend = "nd"
# ) {
#   # Ensure dgCMatrix format
#   train_mat <- as(train_mat, "dgCMatrix")
#   test_mat <- as(test_mat, "dgCMatrix")
  
#   # Handle query (may be NULL)
#   if (!is.null(query_mat)) {
#     query_mat <- as(query_mat, "dgCMatrix")
#     query_x <- query_mat@x
#     query_i <- query_mat@i
#     query_p <- query_mat@p
#     query_dims <- c(nrow(query_mat), ncol(query_mat))
#   } else {
#     query_x <- NULL
#     query_i <- NULL
#     query_p <- NULL
#     query_dims <- NULL
#     query_sf <- NULL
#   }
  
#   # Call Rust function
#   viewmastR:::process_learning_obj_sparse(
#     model_type = model_type,
#     # Train
#     train_x = train_mat@x,
#     train_i = train_mat@i,
#     train_p = train_mat@p,
#     train_dims = c(nrow(train_mat), ncol(train_mat)),
#     train_labels = as.integer(train_labels),
#     train_size_factors = train_sf,
#     # Test
#     test_x = test_mat@x,
#     test_i = test_mat@i,
#     test_p = test_mat@p,
#     test_dims = c(nrow(test_mat), ncol(test_mat)),
#     test_labels = as.integer(test_labels),
#     test_size_factors = test_sf,
#     # Query
#     query_x = query_x,
#     query_i = query_i,
#     query_p = query_p,
#     query_dims = query_dims,
#     query_size_factors = query_sf,
#     # Metadata
#     labels = labels,
#     feature_names = feature_names,
#     # Model params
#     hidden_size = hidden_size,
#     learning_rate = learning_rate,
#     num_epochs = num_epochs,
#     directory = directory,
#     verbose = verbose,
#     backend = backend
#   )
# }


# seurat_size_factors<-function (cds, round_exprs = TRUE, method = c("mean-geometric-mean-total", 
#                                                                    "mean-geometric-mean-log-total")) 
# {
#   method <- match.arg(method)
#   mat<-get_counts_seurat(cds)
#   if (any(Matrix::colSums(mat) == 
#           0)) {
#     warning("Your CDS object contains cells with zero reads. ", 
#             "This causes size factor calculation to fail. Please remove ", 
#             "the zero read cells using ", "cds <- cds[,Matrix::colSums(exprs(cds)) != 0] and then ", 
#             "run cds <- estimate_size_factors(cds)")
#     return(cds)
#   }
#   if (is_sparse_matrix(mat)) {
#     sf <- monocle3:::estimate_sf_sparse(mat, 
#                                         round_exprs = round_exprs, method = method)
#   }
#   else {
#     sf <- monocle3:::estimate_sf_dense(mat, 
#                                        round_exprs = round_exprs, method = method)
#   }
#   return(sf)
# }

# is_sparse_matrix<-function (x) 
# {
#   class(x) %in% c("dgCMatrix", "dgTMatrix", "lgCMatrix")
# }


# get_norm_counts<-function (cds, norm_method = c("log", "binary", "size_only", "none"), 
#                            pseudocount = 1) 
# {
#   software<-NULL
#   norm_method = match.arg(norm_method)
#   if(class(cds)=="Seurat"){software<-"seurat"}
#   if(class(cds)=="cell_data_set"){software<-"monocle3"}
#   if(is.null(software)){stop("software not found for input")}
#   if(software=="monocle3"){
#     norm_mat = SingleCellExperiment::counts(cds)
#     if(norm_method=="none"){
#       return(norm_mat)
#     }
#     sf<-size_factors(cds)
#   }
#   if(software=="seurat"){
#     norm_mat <- get_counts_seurat(cds)
#     if(norm_method=="none"){
#       return(norm_mat)
#     }
#     sf<-seurat_size_factors(cds)
#   }
#   if (norm_method == "binary") {
#     norm_mat = norm_mat > 0
#     if (is_sparse_matrix(norm_mat)) {
#       norm_mat = methods::as(norm_mat, "dgCMatrix")
#     }
#   }
#   else {
#     if (is_sparse_matrix(norm_mat)) {
#       norm_mat@x = norm_mat@x/rep.int(sf, 
#                                       diff(norm_mat@p))
#       if (norm_method == "log") {
#         if (pseudocount == 1) {
#           norm_mat@x = log10(norm_mat@x + pseudocount)
#         }
#         else {
#           stop("Pseudocount must equal 1 with sparse expression matrices")
#         }
#       }
#     }
#     else {
#       norm_mat = Matrix::t(Matrix::t(norm_mat)/sf)
#       if (norm_method == "log") {
#         norm_mat@x <- log10(norm_mat + pseudocount)
#       }
#     }
#   }
#   return(norm_mat)
# }


# get_seurat_version <- function(obj){
#   return(obj@version)
# }

# get_counts_seurat <- function(cds, assay="RNA"){
#   sversion <- packageVersion("Seurat")
#   oversion <- tryCatch(get_seurat_version(cds), error = function(e) NULL)
#   # --- Scenario: User has v4 installed, but Object is v5 ---
#   # Seurat v4 cannot natively read v5 'Assay5' layers.
#   if (!is.null(oversion) && compareVersion(as.character(sversion), "5.0.0")==-1 && !compareVersion(as.character(oversion), "5.0.0")==-1) {
#     warning(sprintf(
#       "Version Mismatch: System has Seurat v%s, but object is v%s. \nSeurat v4 may not read v5 objects with 'layers'. Result may be empty.", 
#       sversion, oversion
#     ))
#   }
#   if (!compareVersion(as.character(sversion), "5.0.0")==-1) {
#     # Seurat v5+ uses 'layer'
#     return(Seurat::GetAssayData(object = cds, assay = assay, layer = "counts"))
#   } else if (compareVersion(as.character(sversion), "2.9.9")==1 & compareVersion(as.character(sversion), "5.0.0")==-1) {
#     # Seurat v3/v4 uses 'slot'
#     return(Seurat::GetAssayData(object = cds, assay = assay, slot = "counts"))
#   } else {
#     warning("Seurat version is too old for this object. Please upgrade to Seurat v3 or higher.")
#     return(NULL)
#   }
# } 

# #' Helper function for summing sparse matrix groups
# #' @references Granja, J. M.et al. (2019). Single-cell multiomic analysis identifies regulatory programs in mixed-phenotype 
# #' acute leukemia. Nature Biotechnology, 37(12), 1458–1465.
# #' @export
# #' @keywords internal
# sparseRowVariances <- function (m){
#   rM <- Matrix::rowMeans(m)
#   rV <- computeSparseRowVariances(m@i + 1, m@x, rM, ncol(m))
#   return(rV)
# }
