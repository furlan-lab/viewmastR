# Define S4 classes for training items and sets
setClass("training_item", slots = c(data = "numeric", target = "numeric"))
setClass("training_set", slots = c(
  name = "character",
  items = "list",
  labels = "character",
  features = "character"
))

# Helper function to check dimensions of an object
dimension_check <- function(obj) {
  nm <- deparse(substitute(obj))
  if (is.null(dim(obj))) {
    if (is.list(obj)) {
      llen <- sapply(obj, length)
      if (length(llen) > 1 && !anyNA(llen)) {
        if (var(llen) == 0) {
          paste0(nm, "-dim1: ", llen[1], " non-ragged")
        } else {
          paste0(nm, "-dim1: ", mean(llen), " (mean) ragged")
        }
      } else {
        message("Not numeric or variable lengths.")
      }
    } else {
      paste0(nm, "-dim1: ", length(obj))
    }
  } else {
    nd <- length(dim(obj))
    dv <- paste0("-dim", 1:nd, ": ")
    paste0(nm, paste0(dv, dim(obj)), collapse = " ")
  }
}

#' @export
#' @title viewmastR
#' @description This function runs viewmastR using the new Rust implementation to generate various forms of training and test data from the given query and reference cell data sets.
#' @param query_cds A query cell data set (cds).
#' @param ref_cds A reference cell data set (cds).
#' @param ref_celldata_col The column in the reference cell data set containing cell data.
#' @param query_celldata_col The column in the query cell data set to store predictions. Defaults to "viewmastR_pred" if not provided.
#' @param FUNC The machine learning model to use ("mlr", "nn", or "nb"). Default is "mlr".
#' @param norm_method The normalization method to use. Options are "log", "binary", "size_only", or "none". Default is "log".
#' @param selected_genes A vector of pre-selected genes for analysis. If NULL, common features between reference and query are used.
#' @param train_frac The fraction of data to use for training. Default is 0.8.
#' @param tf_idf Logical indicating whether to perform TF-IDF transformation. Default is FALSE.
#' @param scale Logical indicating whether to scale the data. Default is FALSE.
#' @param hidden_layers A vector specifying the number of neurons in hidden layers for neural network models. Default is c(500, 100).
#' @param learning_rate The learning rate for training the model. Default is 1e-3.
#' @param max_epochs The maximum number of epochs for training the model. Default is 10.
#' @param LSImethod The method for Latent Semantic Indexing. Default is 1.
#' @param verbose Logical indicating whether to display verbose output. Default is TRUE.
#' @param backend The backend to use ("wgpu", "nd", or "candle"). Default is "wgpu".
#' @param threshold Threshold for predictions. Default is NULL.
#' @param keras_model A pre-trained Keras model to use. Default is NULL.
#' @param dir The directory to save output files. Default is "/tmp/sc_local".
#' @param return_probs Logical indicating whether to return prediction probabilities. Default is FALSE.
#' @param return_type The type of output to return ("object" or "list"). Default is "object".
#' @param debug Logical indicating whether to run in debug mode. Default is FALSE.
#' @param ... Additional arguments.
#' @return Depending on `return_type`, returns the modified query cell data set or a list containing the query cell data set and training output.
viewmastR <- function(
    query_cds,
    ref_cds,
    ref_celldata_col,
    query_celldata_col = NULL,
    FUNC = c("mlr", "nn", "nb"),
    norm_method = c("log", "binary", "size_only", "none"),
    selected_genes = NULL,
    train_frac = 0.8,
    tf_idf = FALSE,
    scale = FALSE,
    hidden_layers = c(500, 100),
    learning_rate = 1e-3,
    max_epochs = 10,
    LSImethod = 1,
    verbose = TRUE,
    backend = c("wgpu", "nd", "candle"),
    threshold = NULL,
    keras_model = NULL,
    dir = "/tmp/sc_local",
    return_probs = FALSE,
    return_type = c("object", "list"),
    debug = FALSE,
    ...
) {
  return_type <- match.arg(return_type)
  backend <- match.arg(backend)
  FUNC <- match.arg(FUNC)
  norm_method <- match.arg(norm_method)
  
  if (return_type == "object" && return_probs == TRUE) {
    stop("Cannot return both probabilities and a single cell object; set return_type to 'list' if probabilities are needed.")
  }
  if (!length(hidden_layers) %in% c(1, 2)) {
    stop("Only 1 or 2 hidden layers are allowed.")
  }
  if (debug) {
    message("Dimension check:")
    message(paste0("\t", dimension_check(query_cds)))
    message(paste0("\t", dimension_check(ref_cds)))
    message(paste0("\t", dimension_check(selected_genes)))
  }
  training_list <- setup_training(
    query_cds = query_cds,
    ref_cds = ref_cds,
    ref_celldata_col = ref_celldata_col,
    norm_method = norm_method,
    selected_genes = selected_genes,
    train_frac = train_frac,
    tf_idf = tf_idf,
    scale = scale,
    LSImethod = LSImethod,
    verbose = verbose,
    return_type = "list",
    debug = debug
  )
  
  if (debug) {
    message("Dimension check:")
    message(paste0("\t", dimension_check(training_list[["train"]][[1]]$data)))
    message(paste0("\t", dimension_check(training_list[["test"]][[1]]$data)))
    message(paste0("\t", dimension_check(training_list[["query"]][[1]]$data)))
    message(paste0("\t", dimension_check(training_list[["labels"]])))
  }
  if (!file.exists(dir)) {
    dir.create(dir, recursive = TRUE)
  }
  
  # Placeholder for the actual model training and prediction functions
  # Replace these with your actual implementations
  if (FUNC == "mlr") {
    export_list <- process_learning_obj_mlr(
      train = training_list[["train"]],
      test = training_list[["test"]],
      query = training_list[["query"]],
      labels = training_list[["labels"]],
      learning_rate = learning_rate,
      num_epochs = max_epochs,
      directory = dir,
      verbose = verbose,
      backend = backend
    )
  }
  if (FUNC == "nn") {
    export_list <- process_learning_obj_ann(
      train = training_list[["train"]],
      test = training_list[["test"]],
      query = training_list[["query"]],
      labels = training_list[["labels"]],
      hidden_size = hidden_layers,
      learning_rate = learning_rate,
      num_epochs = max_epochs,
      directory = dir,
      verbose = verbose,
      backend = backend
    )
  }
  if (FUNC == "nb") {
    export_list <- process_learning_obj_nb(
      train = training_list[["train"]],
      test = training_list[["test"]],
      query = training_list[["query"]]
    )
  }
  
  if (is.null(query_celldata_col)) {
    query_celldata_col <- "viewmastR_pred"
  }
  
  # Assign predictions to the query cell data set
  query_cds[[query_celldata_col]] <- training_list[["labels"]][export_list$predictions + 1]
  
  if (return_type == "object") {
    return(query_cds)
  } else {
    return(list(object = query_cds, training_output = export_list))
  }
}

#' @title Setup Training Datasets
#' @description Prepares training and testing datasets for machine learning models.
#' @param query_cds A query cell data set (cds).
#' @param ref_cds A reference cell data set (cds).
#' @param ref_celldata_col The column in the reference cell data set containing cell data.
#' @param norm_method The normalization method to use. Options are "log", "binary", "size_only", or "none".
#' @param selected_genes A vector of pre-selected genes for analysis.
#' @param train_frac The fraction of data to use for training.
#' @param tf_idf Logical indicating whether to perform TF-IDF transformation.
#' @param scale Logical indicating whether to scale the data.
#' @param LSImethod The method for Latent Semantic Indexing.
#' @param verbose Logical indicating whether to display verbose output.
#' @param addbias Logical indicating whether to add bias.
#' @param return_type The type of output to return ("list", "matrix", or "S4obj").
#' @param debug Logical indicating whether to run in debug mode.
#' @param ... Additional arguments.
#' @return A list containing training and testing datasets.
#' @keywords internal
setup_training <- function(
    query_cds,
    ref_cds,
    ref_celldata_col,
    norm_method = c("log", "binary", "size_only", "none"),
    selected_genes = NULL,
    train_frac = 0.8,
    tf_idf = FALSE,
    scale = FALSE,
    LSImethod = 1,
    verbose = TRUE,
    addbias = FALSE,
    return_type = c("list", "matrix", "S4obj"),
    debug = FALSE,
    ...
) {
  if (verbose) {
    message("Checking arguments and input")
  }
  norm_method <- match.arg(norm_method)
  return_type <- match.arg(return_type)
  
  if (tf_idf && scale) {
    warning("Both tf_idf and scale selected. Using tf_idf alone.")
    scale <- FALSE
  }
  
  if (class(query_cds) != class(ref_cds)) {
    stop("Input objects must be of the same class.")
  }
  software <- NULL
  if (inherits(query_cds, "Seurat")) {
    software <- "seurat"
    labf <- as.factor(ref_cds@meta.data[[ref_celldata_col]])
  } else if (inherits(query_cds, "cell_data_set")) {
    software <- "monocle3"
    labf <- as.factor(colData(ref_cds)[[ref_celldata_col]])
  } else {
    stop("Unsupported object class.")
  }
  
  # Find common features
  if (verbose) {
    message("Finding common features between reference and query")
  }
  common_list <- common_features(list(ref_cds, query_cds))
  names(common_list) <- c("ref", "query")
  rm(ref_cds)
  gc()
  
  if (is.null(selected_genes)) {
    selected_common <- rownames(common_list[["query"]])
    selected_common <- selected_common[selected_common %in% rownames(common_list[["ref"]])]
  } else {
    if (verbose) {
      message("Subsetting by pre-selected features")
    }
    selected_common <- selected_genes
    selected_common <- selected_common[selected_common %in% rownames(common_list[["query"]])]
    selected_common <- selected_common[selected_common %in% rownames(common_list[["ref"]])]
  }
  
  # Normalize counts
  if (verbose) {
    message("Calculating normalized counts")
  }
  if (norm_method != "none") {
    query <- get_norm_counts(common_list[["query"]], norm_method = norm_method)[selected_common, ]
    X <- get_norm_counts(common_list[["ref"]], norm_method = norm_method)[selected_common, ]
  } else {
    query <- get_norm_counts(common_list[["query"]], norm_method = "none")[selected_common, ]
    X <- get_norm_counts(common_list[["ref"]], norm_method = "none")[selected_common, ]
  }
  rm(common_list)
  gc()
  
  # Perform scaling methods
  if (tf_idf) {
    if (verbose) {
      message("Performing TF-IDF")
    }
    X <- tf_idf_transform(X, LSImethod)
    query <- tf_idf_transform(query, LSImethod)
  } else if (scale) {
    if (verbose) {
      message("Scaling data")
    }
    X <- scale(X)
    query <- scale(query)
  }
  
  if (verbose) {
    message("Converting to dense matrix")
  }
  if (addbias) {
    if (verbose) {
      message("Adding bias")
    }
    X <- rbind(rep(1, ncol(X)), X)
    query <- rbind(rep(1, ncol(query)), query)
  }
  X <- as.matrix(X)
  query <- as.matrix(query)
  gc()
  
  # Prepare labels
  labels <- levels(labf)
  Ylab <- as.integer(factor(labf, levels = labels)) - 1
  Y <- model.matrix(~ 0 + labf)
  features <- rownames(X)
  rownames(X) <- NULL
  colnames(X) <- NULL
  
  # Create train/test indices
  set.seed(123)  # For reproducibility
  train_idx <- sample(seq_len(ncol(X)), size = round(train_frac * ncol(X)))
  test_idx <- setdiff(seq_len(ncol(X)), train_idx)
  
  if (return_type == "matrix") {
    return(list(
      Xtrain_data = t(X[, train_idx]),
      Xtest_data = t(X[, test_idx]),
      Ytrain_label = Y[train_idx, ],
      Ytest_label = Y[test_idx, ],
      query = t(query),
      label_text = labels,
      features = features
    ))
  } else if (return_type == "list") {
    train_list <- lapply(train_idx, function(idx) {
      list(data = X[, idx], target = Ylab[idx])
    })
    test_list <- lapply(test_idx, function(idx) {
      list(data = X[, idx], target = Ylab[idx])
    })
    query_list <- lapply(seq_len(ncol(query)), function(idx) {
      list(data = query[, idx])
    })
    return(list(
      train = train_list,
      test = test_list,
      query = query_list,
      labels = labels,
      features = features
    ))
  } else if (return_type == "S4obj") {
    training_set <- new("training_set",
                        name = "train",
                        items = lapply(train_idx, function(idx) {
                          new("training_item", data = X[, idx], target = Ylab[idx])
                        }),
                        labels = labels,
                        features = features
    )
    test_set <- new("training_set",
                    name = "test",
                    items = lapply(test_idx, function(idx) {
                      new("training_item", data = X[, idx], target = Ylab[idx])
                    }),
                    labels = labels,
                    features = features
    )
    query_set <- new("training_set",
                     name = "query",
                     items = lapply(seq_len(ncol(query)), function(idx) {
                       new("training_item", data = query[, idx], target = 0)
                     }),
                     labels = "Unknown",
                     features = features
    )
    return(list(training_set, test_set, query_set))
  }
}

# Function to get normalized counts
get_norm_counts <- function(cds, norm_method = c("log", "binary", "size_only", "none"), pseudocount = 1) {
  norm_method <- match.arg(norm_method)
  software <- NULL
  if (inherits(cds, "Seurat")) {
    software <- "seurat"
  } else if (inherits(cds, "cell_data_set")) {
    software <- "monocle3"
  } else {
    stop("Unsupported object class.")
  }
  
  if (software == "monocle3") {
    norm_mat <- SingleCellExperiment::counts(cds)
    if (norm_method == "none") {
      return(norm_mat)
    }
    sf <- size_factors(cds)
  } else if (software == "seurat") {
    norm_mat <- get_counts_seurat(cds)
    if (norm_method == "none") {
      return(norm_mat)
    }
    sf <- seurat_size_factors(cds)
  }
  
  if (norm_method == "binary") {
    norm_mat <- norm_mat > 0
    if (is_sparse_matrix(norm_mat)) {
      norm_mat <- as(norm_mat, "dgCMatrix")
    }
  } else {
    if (is_sparse_matrix(norm_mat)) {
      norm_mat@x <- norm_mat@x / rep.int(sf, diff(norm_mat@p))
      if (norm_method == "log") {
        if (pseudocount == 1) {
          norm_mat@x <- log10(norm_mat@x + pseudocount)
        } else {
          stop("Pseudocount must equal 1 with sparse expression matrices.")
        }
      }
    } else {
      norm_mat <- t(t(norm_mat) / sf)
      if (norm_method == "log") {
        norm_mat <- log10(norm_mat + pseudocount)
      }
    }
  }
  return(norm_mat)
}

# Function to calculate size factors for Seurat objects
seurat_size_factors <- function(cds, method = c("mean-geometric-mean-total", "mean-geometric-mean-log-total")) {
  method <- match.arg(method)
  mat <- get_counts_seurat(cds)
  if (any(Matrix::colSums(mat) == 0)) {
    warning("Your Seurat object contains cells with zero reads. Please remove zero-read cells before proceeding.")
    return(NULL)
  }
  if (is_sparse_matrix(mat)) {
    sf <- monocle3:::estimate_sf_sparse(mat, method = method)
  } else {
    sf <- monocle3:::estimate_sf_dense(mat, method = method)
  }
  return(sf)
}

# Function to check if a matrix is sparse
is_sparse_matrix <- function(x) {
  inherits(x, c("dgCMatrix", "dgTMatrix", "lgCMatrix"))
}

# Function to get counts from a Seurat object
get_counts_seurat <- function(cds) {
  GetAssayData(object = cds, assay = cds@active.assay, slot = "counts")
}
