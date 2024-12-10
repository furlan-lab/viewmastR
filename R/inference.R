
#' @importFrom pbmcapply pbmclapply
#' @keywords internal

#' @importFrom pbmcapply pbmclapply
#' @keywords internal
infer_prep <- function(object, variable_features, software, chunks = 1, workers = 1) {
  # If we are not parallelizing (or only one worker), run sequentially
  if (workers <= 1) {
    qcounts <- t(as.matrix(get_norm_counts(object)[variable_features, ]))
    query <- lapply(seq_len(nrow(qcounts)), function(idx) {
      list(data = as.numeric(qcounts[idx, ]))
    })
    return(list(query)) # Return as a single chunk for consistency
  }
  
  # Parallel mode: split cells into chunks
  n_cells <- ncol(get_norm_counts(object))
  chunk_size <- ceiling(n_cells / chunks)
  
  # Create cell index chunks
  all_cells <- seq_len(n_cells)
  cell_chunks <- split(all_cells, ceiling(seq_along(all_cells) / chunk_size))
  
  # Retrieve chunked queries in parallel
  chunked_query <- pbmclapply(cell_chunks, function(cells) {
    mat <- t(get_norm_counts(object)[variable_features, cells, drop = FALSE])
    # Create query list for this chunk
    lapply(seq_len(nrow(mat)), function(idx) {
      list(data = as.numeric(mat[idx, ]))
    })
  }, mc.cores = workers)
  
  # Here, chunked_query is a list of chunks, each chunk is a list of cell entries
  # We do NOT flatten; we keep the chunk structure.
  return(chunked_query)
}


#' Function to infer cell labels using a trained model
#' 
#' @param query_cds Seurat or cell_data_set object
#' @param model_path character path to the trained model file
#' @param vg character vector of features used for inference
#' @param query_celldata_col character column name to store inferred cell labels
#' @param labels character vector of labels (optional)
#' @param verbose logical, show messages
#' @param return_probs logical, return class probabilities
#' @param return_type one of "object" or "list"
#' @param chunks integer number of chunks to split the data into for parallelization
#' @param workers integer number of parallel workers
#' @importFrom RcppMsgPack msgpackRead
#' @importFrom future plan
#' @importFrom future multisession
#' @importFrom future.apply future_lapply
#' @export
viewmastR_infer <- function(query_cds,
                            model_path,
                            vg,
                            query_celldata_col = "viewmastR_inferred",
                            labels = NULL,
                            verbose = TRUE,
                            return_probs = FALSE,
                            return_type = c("object", "list"),
                            chunks = 1,
                            workers = 1) {
  
  return_type <- match.arg(return_type)
  
  software <- NULL
  if (inherits(query_cds, "Seurat")) {
    software <- "seurat"
  } else if (inherits(query_cds, "cell_data_set")) {
    software <- "monocle3"
  }
  if (is.null(software)) {
    stop("Only seurat and monocle3 objects supported")
  }
  
  if (verbose) message("Preparing query")
  # Now infer_prep returns chunked queries if workers > 1
  query_chunks <- infer_prep(query_cds, variable_features = vg, software = software, chunks = chunks, workers = workers)
  # Read the model
  mod <- msgpackRead(model_path, simplify = TRUE)
  num_classes <- mod$item$linear1$weight$param$shape[2]
  # If we have multiple chunks and workers > 1, run inference in parallel on each chunk
  if (workers > 1 && length(query_chunks) > 1) {
    if (verbose) message("Running inference on chunks in parallel")
    # library(future)
    # library(future.apply)
    plan(multisession, workers = workers)  # or plan(cluster, workers=workers) on Windows
    # 
    chunk_results <- future_lapply(query_chunks, function(chunk) {
      viewmastR::infer_from_model(
        model_path = model_path,
        query = chunk,
        num_classes = num_classes,
        num_features = length(vg),
        verbose = FALSE
      )
    })
    # chunk_results <- pbmclapply(query_chunks, function(chunk) {
    #   library(viewmastR)
    #   viewmastR::infer_from_model(
    #     model_path = model_path,
    #     query = chunk,
    #     num_classes = num_classes,
    #     num_features = length(vg),
    #     verbose = FALSE
    #   )
    # }, mc.cores = workers)
    chunk_results <- lapply(query_chunks, function(chunk) {
       viewmastR:::infer_from_model(
         model_path = model_path,
         query = chunk,
         num_classes = num_classes,
         num_features = length(vg),
         verbose = FALSE)
       })
    
    # Combine the probabilities from all chunks
    probs_list <- lapply(chunk_results, `[[`, "probs")
    log_odds <- unlist(probs_list)
    
    # Create a single export_list to mimic original structure
    export_list <- list(probs = log_odds,
                        chunk_results = chunk_results)
    
  } else {
    # Single worker or only one chunk: just run once
    export_list <- infer_from_model(
      model_path = model_path,
      query = query_chunks[[1]],
      num_classes = num_classes,
      num_features = length(vg),
      verbose = verbose
    )
    log_odds <- unlist(export_list$probs)
  }
  
  # Check dimension consistency
  if ((length(log_odds) %% ncol(query_cds)) == 0) {
    log_odds <- matrix(log_odds, ncol = ncol(query_cds))
    log_odds <- t(log_odds)
    if (is.null(labels)) {
      labels <- paste0("prob_celltype_", seq_len(ncol(log_odds)))
    }
    colnames(log_odds) <- paste0("prob_", labels)
  } else {
    stop("Error in log odds dimensions of function output")
  }
  
  # Convert log odds to probabilities
  export_list$probs <- plogis(log_odds)
  
  # Assign inferred labels
  query_cds[[query_celldata_col]] <- labels[apply(log_odds, 1, which.max)]
  
  # Optionally add probabilities
  if (return_probs) {
    if (software == "seurat") {
      query_cds@meta.data <- cbind(query_cds@meta.data, export_list$probs)
    } else if (software == "monocle3") {
      colData(query_cds) <- cbind(colData(query_cds), export_list$probs)
    }
  }
  
  if (return_type == "object") {
    return(query_cds)
  } else {
    return(list(object = query_cds, training_output = export_list))
  }
}

