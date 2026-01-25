################################################################################
# FILE: R/inference.R
# STATUS: Clean
# ------------------------------------------------------------------------------
# Functions:
# [x] viewmastR_infer           (Exported)
# [x] extract_mpk_shapes        (Exported - Internal)
# [x] check_genes_in_object     (Exported)
################################################################################


#' ViewmastR inference
#' 
#' This function performs cell type inference using a trained model,
#' passing sparse matrices directly to Rust for maximum performance.
#' 
#' @param query_cds Seurat or cell_data_set object
#' @param model_dir Path to the trained model directory
#' @param selected_features Character vector of feature names
#' @param query_celldata_col Name of column to store results (default: "viewmastR_inferred")
#' @param labels Optional character vector of class labels
#' @param verbose Print progress messages
#' @param return_probs If TRUE, add probability columns to metadata
#' @param return_type "object" returns the modified object, "list" returns object and probs
#' @param batch_size Cells per inference batch (default: auto)
#' @param threads Number of parallel threads (default: 1)
#' 
#' @return Modified query_cds with inferred labels
#' @export
viewmastR_infer <- function(query_cds,
                                  model_dir,
                                  selected_features,
                                  query_celldata_col = "viewmastR_inferred",
                                  labels = NULL,
                                  verbose = TRUE,
                                  return_probs = FALSE,
                                  return_type = c("object", "list"),
                                  batch_size = NULL,
                                  threads = 1) {
  
  return_type <- match.arg(return_type)
  
  # Validation
  checked_genes <- check_genes_in_object(query_cds, selected_features, model_dir)
  if (length(checked_genes$genes_missing) > 0) {
    stop("At least one feature in variable_features is not found in the object.")
  }
  
  # Get model info
  model_shapes <- extract_mpk_shapes(file.path(model_dir, "model.mpk"))
  num_classes_model <- as.integer(model_shapes$num_classes)
  
  norm_mat <- get_norm_counts(query_cds[selected_features, ], norm_method = "log")
  # Ensure it's a dgCMatrix
  if (!inherits(norm_mat, "dgCMatrix")) {
    norm_mat <- as(norm_mat, "dgCMatrix")
  }
  
  n_cells <- ncol(norm_mat)
  n_features <- nrow(norm_mat)
  
  if(verbose) message(sprintf("Matrix: %d features × %d cells, %d non-zeros (%.1f%% sparse)",
                               n_features, n_cells, length(norm_mat@x),
                               100 * (1 - length(norm_mat@x) / (n_features * n_cells))))
  
  # Set batch size
  if (is.null(batch_size)) {
    batch_size <- min(ceiling(n_cells / 20), 1024L)
    if (verbose) message(paste0("Batch size: ", batch_size))
  }
  
  if (verbose) message(sprintf("Running inference with %d workers (threads)...", threads
  ))
  
  # Call Rust with sparse matrix components directly
  res <- viewmastR:::infer_sparse(
    x = norm_mat@x,
    i = norm_mat@i,
    p = norm_mat@p,
    dims = c(nrow(norm_mat), ncol(norm_mat)),
    size_factors = NULL,  # Let Rust compute from the matrix
    model_path = file.path(model_dir, "model.mpk"),
    model_type = model_shapes$model_type,
    num_classes = num_classes_model,
    hidden1 = as.integer(model_shapes$hidden_layer1),
    hidden2 = as.integer(model_shapes$hidden_layer2),
    batch_size = as.integer(batch_size),
    num_threads = as.integer(threads),
    verbose = verbose
  )
  
  # Process results
  log_odds <- unlist(res$probs)
  
  if(is.null(labels)) labels <- 1:num_classes_model
  
  log_odds_mat <- matrix(log_odds, ncol = num_classes_model, byrow = TRUE)
  colnames(log_odds_mat) <- paste0("prob_", labels)
  
  # Softmax
  softmax_rows <- function(mat) {
    shifted <- mat - apply(mat, 1, max)
    exp_shifted <- exp(shifted)
    exp_shifted / rowSums(exp_shifted)
  }
  
  prob_mat <- softmax_rows(log_odds_mat)
  
  # Add results to object
  query_cds[[query_celldata_col]] <- labels[apply(prob_mat, 1, which.max)]
  
  if(return_probs){
    if(inherits(query_cds, "Seurat")) {
      query_cds <- Seurat::AddMetaData(query_cds, as.data.frame(prob_mat))
    } else {
      Biobase::pData(query_cds) <- cbind(Biobase::pData(query_cds), prob_mat)
    }
  }
  
  if (return_type == "object") {
    return(query_cds)
  } else {
    return(list(object = query_cds, training_output = list(probs = prob_mat)))
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
#' @importFrom RcppMsgPack msgpackRead
#' @keywords internal
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
  return(list(df = df, num_classes = num_classes, model_type = model_type, hidden_layer1 = hidden_layer1, hidden_layer2 = hidden_layer2))
}

#' @importFrom RcppMsgPack msgpack_read
#' @export
check_genes_in_object <- function(object, genes, model_dir, assay = "RNA", verbose = TRUE, print = FALSE) {
  # Check if the input is a Seurat object
  if (!inherits(object, "Seurat")) {
    stop("Input must be a Seurat object.")
  }

  # Check if the specified assay exists in the Seurat object
  if (!assay %in% names(object@assays)) {
    stop(paste("Assay", assay, "not found in the Seurat object."))
  }

  counts <- get_counts_seurat(object)
  # Get available genes
  available_genes <- rownames(counts)
  object_type <- "Seurat"

  # Check whether all genes are present in the model
  meta <- msgpack_read(file.path(model_dir, "meta.mpk"), simplify = TRUE)
  if(!all(genes %in% meta$feature_names)) {stop(paste0("Not all genes supplied are found in saved meta data file: ", file.path(model_dir, "meta.mpk")))}
  
  # Check which genes are present in the object
  genes_present <- genes[genes %in% available_genes]
  genes_missing <- genes[!(genes %in% available_genes)]


  # Print a summary
  if (verbose) {
    message(paste("Object Type:", object_type))
    message(paste("Total Genes Checked:", length(genes)))
    message(paste("Genes Present:", length(genes_present)))
    message(paste("Genes Missing:", length(genes_missing), "\n"))
  }

  if (print) {
    if (length(genes_present) > 0) {
      message("Genes Present in the Object:")
      print(genes_present)
    } else {
      message("No genes from the input list are present in the object.")
    }

    if (length(genes_missing) > 0) {
      message("\nGenes Missing from the Object:")
      print(genes_missing)
    } else {
      message("\nAll input genes are present in the object.")
    }
  }

  # Return a list containing the results
  return(list(
    genes_present = genes_present,
    genes_missing = genes_missing
  ))
}
