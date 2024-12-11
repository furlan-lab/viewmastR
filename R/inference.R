#' Infer Cell Labels Using a Trained Model
#'
#' This function prepares query data and runs cell-type label inference in parallel, 
#' using a pre-trained model stored at a specified `model_path`. It can handle both 
#' Seurat and Monocle3 objects, extract relevant feature data, split the work into 
#' parallel chunks, and display a progress bar with total elapsed time.
#'
#' @param query_cds A \code{Seurat} or \code{cell_data_set} object. The dataset for 
#'   which cell labels are to be inferred.
#' @param model_path A character string specifying the path to the trained model file.
#' @param vg A character vector of variable features to be used for inference. These 
#'   features must be present in the \code{query_cds}.
#' @param query_celldata_col A character string specifying the column name in 
#'   \code{query_cds} metadata (for Seurat) or \code{colData}(for Monocle3) 
#'   to store the inferred cell labels. Default is \code{"viewmastR_inferred"}.
#' @param labels A character vector of labels corresponding to the class indices. If 
#'   \code{NULL}, generic labels are generated.
#' @param verbose A logical indicating whether to print informative messages. Default 
#'   is \code{TRUE}.
#' @param return_probs A logical indicating whether to return class probabilities in 
#'   addition to inferred labels. Default is \code{FALSE}.
#' @param return_type A character string, either \code{"object"} or \code{"list"}. If 
#'   \code{"object"}, the updated \code{query_cds} is returned. If \code{"list"}, a 
#'   list containing the updated object and the raw inference results is returned. 
#'   Default is \code{"object"}.
#' @param chunks An integer indicating the number of chunks to split the data into for 
#'   parallelization. Default is 1 (no chunking).
#' @param workers An integer specifying the number of parallel workers to use. Default 
#'   is 1 (no parallelization).
#' @param batch_size An integer specifying the batch size used during inference. If 
#'   \code{NULL}, a heuristic is used to determine a suitable batch size based on the 
#'   data and number of chunks.
#' @param show_progress A logical indicating whether to show a progress bar with total 
#'   elapsed time. Default is \code{TRUE}.
#'
#' @details The function first checks that all variable features specified in \code{vg} 
#' are present in \code{query_cds}, extracts normalized counts, and determines whether 
#' to run sequentially or in parallel. When parallelization is enabled (\code{workers > 1}), 
#' the dataset is split into chunks, and each chunk is processed and run through the 
#' model inference in parallel. A progress bar is displayed showing the number of 
#' completed chunks and total elapsed time.
#'
#' The underlying model is loaded from the specified \code{model_path}, and class 
#' probabilities (log-odds) are computed for each cell. The function then assigns the 
#' most likely label to each cell. Optionally, the probabilities are added to the 
#' object's metadata.
#'
#' @return Depending on \code{return_type}:
#' \itemize{
#' \item \code{"object"}: Returns the updated \code{query_cds} with inferred labels 
#' in \code{query_celldata_col} and optionally probabilities in the metadata.
#' \item \code{"list"}: Returns a list containing:
#'   \item{\code{object}}{The updated \code{query_cds}.}
#'   \item{\code{training_output}}{The raw inference results including probabilities.}
#' }
#'
#' @seealso \code{\link[Seurat]{Seurat}}, \code{\link[monocle3]{cell_data_set}}
#'
#' @examples
#' \dontrun{
#' # Assuming `seu` is a Seurat object and `model_path` is a path to a trained model
#' result <- viewmastR_infer(
#'   query_cds = seu,
#'   model_path = "path/to/model.msgpack",
#'   vg = VariableFeatures(seu)
#' )
#' }
#'
#' @export
#' @importFrom pbmcapply pbmclapply
#' @importFrom future plan multisession multicore
#' @importFrom future.apply future_lapply
#' @importFrom progressr handlers handler_progress progressor with_progress
#' @importFrom Matrix t
#' @importFrom RcppMsgPack msgpackRead

viewmastR_infer <- function(query_cds,
                            model_path,
                            vg,
                            query_celldata_col = "viewmastR_inferred",
                            labels = NULL,
                            verbose = TRUE,
                            return_probs = FALSE,
                            return_type = c("object", "list"),
                            chunks = 1,
                            workers = 1,
                            batch_size = NULL,
                            show_progress = TRUE) {
  return_type <- match.arg(return_type)
  
  # Determine the software type
  if (inherits(query_cds, "Seurat")) {
    software <- "seurat"
  } else if (inherits(query_cds, "cell_data_set")) {
    software <- "monocle3"
  } else {
    stop("Only seurat and monocle3 objects supported")
  }
  
  # Check that all variable_features are present
  checked_genes <- check_genes_in_object(query_cds, vg)
  if (length(checked_genes$genes_missing) > 0) {
    stop("At least one feature in variable_features is not found in the object.")
  }
  
  # Extract normalized counts
  norm_counts <- get_norm_counts(query_cds[vg, ])
  
  # If workers <= 1, run sequentially
  if (workers <= 1) {
    if (verbose) message("Single worker mode: preparing and running inference sequentially")
    qcounts <- Matrix::t(as.matrix(norm_counts))
    nrows <- nrow(qcounts)
    query_list <- vector("list", nrows)
    for (i in seq_len(nrows)) {
      query_list[[i]] <- list(data = qcounts[i, ])
    }
    
    # Load model
    mod <- msgpackRead(model_path, simplify = TRUE)
    num_classes <- mod$item$linear1$weight$param$shape[2]
    
    # Set batch size if null
    if (is.null(batch_size)) {
      batch_size <- round(ceiling(ncol(query_cds)/20), digits = 0)
      if (verbose) message(paste0("Batch size: ", batch_size))
    }
    
    # Run inference once
    export_list <- infer_from_model(
      model_path = model_path,
      query = query_list,
      num_classes = num_classes,
      num_features = length(vg),
      verbose = verbose,
      batch_size = batch_size
    )
    log_odds <- unlist(export_list$probs)
    
  } else {
    if (verbose) message("Parallel mode: preparing and running inference in parallel")
    
    # On Windows use multisession, on UNIX use multicore
    if (tolower(Sys.info()[['sysname']]) == "windows") {
      plan(multisession, workers = workers)
    } else {
      plan(multicore, workers = workers)
    }
    
    # Pre-transpose for efficient chunk indexing
    tnorm_counts <- Matrix::t(norm_counts)
    n_cells <- nrow(tnorm_counts)
    chunk_size <- ceiling(n_cells / chunks)
    cell_chunks <- split(seq_len(n_cells), ceiling(seq_len(n_cells) / chunk_size))
    
    # Load model outside the loop for reuse
    mod <- msgpackRead(model_path, simplify = TRUE)
    num_classes <- mod$item$linear1$weight$param$shape[2]
    
    # Set batch size if null
    if (is.null(batch_size)) {
      if (chunks == 1) {
        batch_size <- round(max(ncol(query_cds)/50, 128), digits =  0)
      } else {
        batch_size <- round(ncol(query_cds)/50, digits = 0)
      }
      if (verbose) message(paste0("Batch size: ", batch_size))
    }
    
    # Setup progress
if (show_progress) {
      progressr::handlers(
        progressr::handler_progress(
          format = "[:bar] :percent Elapsed: :elapsed ETA: :eta",
          clear = FALSE,
          show_after = 0
        )
      )
    } else {
      progressr::handlers("null")
    }
    
    # Run inference on each chunk in parallel
    chunk_results <- with_progress({
      p <- progressor(steps = length(cell_chunks))
      future_lapply(cell_chunks, function(cells) {
        # Convert this chunk to a query list
        mat <- as.matrix(tnorm_counts[cells, , drop = FALSE])
        nrows <- nrow(mat)
        chunk_query <- vector("list", nrows)
        for (i in seq_len(nrows)) {
          chunk_query[[i]] <- list(data = mat[i, ])
        }
        
        # Run inference on this chunk
        res <- infer_from_model(
          model_path = model_path,
          query = chunk_query,
          num_classes = num_classes,
          num_features = length(vg),
          verbose = FALSE,
          batch_size = batch_size
        )
        
        p() # Progress update per chunk
        res
      }, future.seed = TRUE)
    })
    
    # Combine the probabilities from all chunks
    probs_list <- lapply(chunk_results, `[[`, "probs")
    log_odds <- unlist(probs_list)
    
    # Create a single export_list to mimic original structure
    export_list <- list(probs = log_odds,
                        chunk_results = chunk_results)
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


#' @export
check_genes_in_object <- function(object, genes, assay = "RNA", verbose = TRUE, print = FALSE) {
  # Check if the input is a Seurat object
  if (!inherits(object, "Seurat")) {
    stop("Input must be a Seurat object.")
  }

  # Check if the specified assay exists in the Seurat object
  if (!assay %in% names(object@assays)) {
    stop(paste("Assay", assay, "not found in the Seurat object."))
  }

  # Detect Seurat version
  seurat_version <- as.numeric(substr(packageVersion("Seurat"), 1, 1))

  if (seurat_version >= 5) {
    # Seurat v5 and above: Use the 'layer' argument
    counts <- Seurat::GetAssayData(object, assay = assay, layer = "counts")
    if (verbose) message("Seurat v5+ detected: using 'layer' argument.")
  } else {
    # Seurat v4 and below: Use the 'slot' argument
    counts <- Seurat::GetAssayData(object, assay = assay, slot = "counts")
    if (verbose) message("Seurat v4 or below detected: using 'slot' argument.")
  }

  # Get available genes
  available_genes <- rownames(counts)
  object_type <- "Seurat"

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
