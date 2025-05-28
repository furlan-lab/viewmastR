
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
#' @param return_type A character string, either \code{"object"} or \code{"list"}. If 
#'   \code{"object"}, the updated \code{query_cds} is returned. If \code{"list"}, a 
#'   list containing the updated object and the raw inference results is returned. 
#'   Default is \code{"object"}.
#' @param return_probs logical If TRUE, returns the class probabilities. Default is FALSE.
#' @param chunks An integer indicating the number of chunks to split the data into for 
#'   parallelization. Default is 1 (no chunking).
#' @param workers An integer specifying the number of parallel workers to use. Default 
#'   is 1 (no parallelization).
#' @param batch_size An integer specifying the batch size used during inference. If 
#'   \code{NULL}, a heuristic is used to determine a suitable batch size based on the 
#'   data and number of chunks.
#' @param show_progress A logical indicating whether to show a progress bar with total 
#'   elapsed time. Default is \code{TRUE}.
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
#' @importFrom pbmcapply pbmclapply
#' @importFrom future plan multisession multicore
#' @importFrom future.apply future_lapply
#' @importFrom progressr handlers handler_progress progressor with_progress
#' @importFrom Matrix t
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
                            workers = 1,
                            batch_size = NULL,
                            show_progress = TRUE) {
  return_type <- match.arg(arg = NULL, return_type)
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
    model_shapes <- extract_mpk_shapes(model_path)

    # Set batch size if null
    if (is.null(batch_size)) {
      batch_size <- round(ceiling(ncol(query_cds)/20), digits = 0)
      if (verbose) message(paste0("Batch size: ", batch_size))
    }

    # Run inference once
    export_list <- infer_from_model(
      model_path = model_path,
      query = query_list,
      num_classes = as.integer(model_shapes$num_classes),
      num_features = as.integer(length(vg)),
      model_type = model_shapes$model_type,
      hidden1 = as.integer(model_shapes$hidden_layer1),
      hidden2 = as.integer(model_shapes$hidden_layer2),
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

    # Load model
    model_shapes <- extract_mpk_shapes(model_path)

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
          num_classes = as.integer(model_shapes$num_classes),
          num_features = as.integer(length(vg)),
          model_type = model_shapes$model_type,
          hidden1 = as.integer(model_shapes$hidden_layer1),
          hidden2 = as.integer(model_shapes$hidden_layer2),
          verbose = verbose,
          batch_size = batch_size
        )
        p() # Progress update per chunk
        res
      }, future.seed = TRUE)
    })

    # Combine the probabilities from all chunks
    probs_list <- lapply(chunk_results, `[[`, "probs")
    log_odds <- unlist(probs_list)
    #export_list <- list(probs = log_odds)
  }
  
  # Check if log_odds has the expected dimensions
  if(length(log_odds) == dim(query_cds)[2] * model_shapes$num_classes){
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
  
  prob_mat  <- softmax_rows(log_odds)
  query_cds[[query_celldata_col]]<-labels[apply(prob_mat, 1, which.max)]
  if(return_probs){
    query_cds@meta.data <- cbind(query_cds@meta.data, prob_mat)
  }
  if (return_type=="object") {
    return(query_cds)
  } else {
    return(list(object=query_cds, training_output = list(probs = prob_mat)))
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
#' @return A list containing **data frame** with one row per tensor and columns:
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
#'  also parsed elements including: num_classes, model_type, hidden_layer1, hidden_layer2
#'}
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
