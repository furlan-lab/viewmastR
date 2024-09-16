#' @export
#' @title Plot Gene Dispersion
#' @description Plots the gene dispersion against the mean expression for a `cell_data_set` or `Seurat` object.
#' This function visualizes the dispersion estimates and highlights the genes selected for ordering.
#' @param cds A `cell_data_set` or `Seurat` object.
#' @param size Numeric value indicating the size of the points in the plot. Default is 1.
#' @param alpha Numeric value indicating the transparency level of the points. Default is 0.4.
#' @return A `ggplot2` object representing the dispersion plot.
#' @importFrom ggplot2 ggplot aes geom_point geom_line theme_bw scale_color_manual xlab ylab
#' @importFrom methods is
plot_gene_dispersion <- function(cds, size = 1, alpha = 0.4) {
  if (methods::is(cds, "cell_data_set")) {
    dispersion_data <- cds@int_metadata$dispersion
  } else if (methods::is(cds, "Seurat")) {
    dispersion_data <- cds@misc$dispersion
  } else {
    stop("Input object must be a 'cell_data_set' or 'Seurat' object.")
  }
  
  if (is.null(dispersion_data)) {
    stop("Dispersion data not found in the provided object.")
  }
  
  # Prepare data for plotting
  if (!is.null(dispersion_data$disp_func)) {
    # Monocle 2 method
    prd <- dispersion_data$disp_table
    prd$fit <- log(dispersion_data$disp_func(prd$mu))
    prd$log_mean <- log(prd$mu)
    prd$log_dispersion <- log(prd$disp)
    prd$selected_genes <- dispersion_data$use_for_ordering
  } else {
    # Monocle 3 or Seurat method
    prd <- dispersion_data
    prd$selected_genes <- prd$use_for_ordering
  }
  
  # Check for required columns
  required_cols <- c("log_mean", "log_dispersion", "fit", "selected_genes")
  if (!all(required_cols %in% colnames(prd))) {
    stop("Dispersion data does not contain the required columns.")
  }
  
  # Create the plot
  g <- ggplot2::ggplot(prd, ggplot2::aes(x = log_mean, y = log_dispersion)) +
    ggplot2::theme_bw() +
    ggplot2::geom_point(ggplot2::aes(color = selected_genes), alpha = alpha, size = size) +
    ggplot2::geom_line(ggplot2::aes(y = fit)) +
    ggplot2::scale_color_manual(values = c("FALSE" = "lightgray", "TRUE" = "firebrick1")) +
    ggplot2::xlab("Log Mean Expression") +
    ggplot2::ylab("Log Dispersion")
  
  return(g)
}

#' @title Select Genes for Dimensionality Reduction
#' @description Selects genes based on dispersion criteria to be used in downstream dimensionality reduction methods.
#' @param cds A `cell_data_set` or `Seurat` object.
#' @param fit_min Numeric value specifying the minimum dispersion fit multiple. Default is 1.
#' @param fit_max Numeric value specifying the maximum dispersion fit multiple. Default is `Inf`.
#' @param logmean_ul Numeric value specifying the upper limit for log mean expression. Default is `Inf`.
#' @param logmean_ll Numeric value specifying the lower limit for log mean expression. Default is `-Inf`.
#' @param top_n Integer specifying the number of top genes to select based on dispersion. Overrides `fit_min` and `fit_max` if provided.
#' @return An updated `cell_data_set` or `Seurat` object with selected genes marked.
#' @export
select_genes <- function(cds, fit_min = 1, fit_max = Inf, logmean_ul = Inf, logmean_ll = -Inf, top_n = NULL) {
  if (methods::is(cds, "cell_data_set")) {
    dispersion_data <- cds@int_metadata$dispersion
  } else if (methods::is(cds, "Seurat")) {
    dispersion_data <- cds@misc$dispersion
  } else {
    stop("Input object must be a 'cell_data_set' or 'Seurat' object.")
  }
  
  if (is.null(dispersion_data)) {
    stop("Dispersion data not found in the provided object.")
  }
  
  # Prepare data
  if (!is.null(dispersion_data$disp_func)) {
    # Monocle 2 method
    df <- dispersion_data$disp_table
    df$fit <- dispersion_data$disp_func(df$mu)
    df$log_mean <- log(df$mu)
    df$log_dispersion <- log(df$disp)
  } else {
    # Monocle 3 or Seurat method
    df <- dispersion_data
  }
  
  df$ratio <- df$log_dispersion / df$fit
  df$index <- seq_len(nrow(df))
  
  # Select genes
  if (!is.null(top_n)) {
    in_range <- df[df$log_mean > logmean_ll & df$log_mean < logmean_ul, ]
    selected_indices <- in_range[order(-in_range$ratio), ][1:top_n, ]$index
  } else {
    selected_indices <- df$index[df$ratio > fit_min & df$ratio < fit_max & df$log_mean > logmean_ll & df$log_mean < logmean_ul]
  }
  
  # Update dispersion data
  if (methods::is(cds, "cell_data_set")) {
    cds@int_metadata$dispersion$use_for_ordering <- df$index %in% selected_indices
  } else if (methods::is(cds, "Seurat")) {
    cds@misc$dispersion$use_for_ordering <- df$index %in% selected_indices
  }
  
  return(cds)
}


#' @title Get Selected Genes
#' @description Retrieves the list of genes selected for ordering in the `cell_data_set` or `Seurat` object.
#' @param cds A `cell_data_set` or `Seurat` object.
#' @param gene_column The name of the gene identifier column. Default is "id".
#' @return A character vector of selected gene identifiers.
#' @export
get_selected_genes <- function(cds, gene_column = "id") {
  if (methods::is(cds, "cell_data_set")) {
    dispersion_data <- cds@int_metadata$dispersion
  } else if (methods::is(cds, "Seurat")) {
    dispersion_data <- cds@misc$dispersion
  } else {
    stop("Input object must be a 'cell_data_set' or 'Seurat' object.")
  }
  
  if (is.null(dispersion_data)) {
    stop("Dispersion data not found in the provided object.")
  }
  
  if (!is.null(dispersion_data$disp_func)) {
    # Monocle 2 method
    df <- dispersion_data$disp_table
  } else {
    df <- dispersion_data
  }
  
  selected_genes <- df[[gene_column]][dispersion_data$use_for_ordering]
  return(as.character(selected_genes))
}


#' @title Set Selected Genes
#' @description Manually sets the list of genes to be used for ordering in the `cell_data_set` or `Seurat` object.
#' @param cds A `cell_data_set` or `Seurat` object.
#' @param genes A character vector of gene identifiers to set as selected.
#' @param gene_column The name of the gene identifier column in the dispersion data. Default is "id".
#' @param unique_column The name of the unique identifier column. Default is "id".
#' @return An updated `cell_data_set` or `Seurat` object with selected genes marked.
#' @export
set_selected_genes <- function(cds, genes, gene_column = "id", unique_column = "id") {
  if (methods::is(cds, "cell_data_set")) {
    dispersion_data <- cds@int_metadata$dispersion
  } else if (methods::is(cds, "Seurat")) {
    dispersion_data <- cds@misc$dispersion
  } else {
    stop("Input object must be a 'cell_data_set' or 'Seurat' object.")
  }
  
  if (is.null(dispersion_data)) {
    stop("Dispersion data not found in the provided object.")
  }
  
  if (!is.null(dispersion_data$disp_func)) {
    # Monocle 2 method
    df <- dispersion_data$disp_table
  } else {
    df <- dispersion_data
  }
  
  if (gene_column %in% colnames(df)) {
    selected_indices <- df[[gene_column]] %in% genes
  } else {
    stop(sprintf("Gene column '%s' not found in dispersion data.", gene_column))
  }
  
  if (methods::is(cds, "cell_data_set")) {
    cds@int_metadata$dispersion$use_for_ordering <- selected_indices
  } else if (methods::is(cds, "Seurat")) {
    cds@misc$dispersion$use_for_ordering <- selected_indices
  }
  
  if (sum(selected_indices) < 1) {
    warning("No ordering genes found.")
  }
  
  return(cds)
}


#' @title Calculate Gene Dispersion
#' @description Calculates dispersion for genes in a `cell_data_set` or `Seurat` object.
#' This is an optional step before dimensionality reduction to identify highly variable genes.
#' @param cds A `cell_data_set` or `Seurat` object.
#' @param q Integer specifying the polynomial degree for the fit. Default is 3.
#' @param id_tag The name of the feature data column corresponding to the unique gene ID. Default is "id".
#' @param symbol_tag The name of the feature data column corresponding to the gene symbol. Default is "gene_short_name".
#' @param method Character string specifying the method to use ("m3addon" or "m2"). Default is "m3addon".
#' @param remove_outliers Logical indicating whether to remove outliers in the dispersion estimation. Default is TRUE.
#' @return An updated `cell_data_set` or `Seurat` object with dispersion data calculated.
#' @export
#' @importFrom Matrix rowMeans
#' @importFrom stats lm predict
#' @importFrom DelayedMatrixStats rowVars
calculate_gene_dispersion <- function(cds, q = 3, id_tag = "id", symbol_tag = "gene_short_name", method = "m3addon", remove_outliers = TRUE) {
  if (methods::is(cds, "cell_data_set")) {
    software <- "monocle3"
  } else if (methods::is(cds, "Seurat")) {
    software <- "seurat"
  } else {
    stop("Input object must be a 'cell_data_set' or 'Seurat' object.")
  }
  
  if (software == "monocle3") {
    cds@int_metadata$dispersion <- NULL
  } else if (software == "seurat") {
    cds@misc$dispersion <- NULL
  }
  
  if (method == "m3addon") {
    # Monocle 3 method
    if (software == "monocle3") {
      ncounts <- Matrix::t(Matrix::t(exprs(cds)) / monocle3::size_factors(cds))
      fdat <- fData(cds)
    } else if (software == "seurat") {
      ncounts <- get_norm_counts(cds)
    }
    
    m <- Matrix::rowMeans(ncounts)
    sd <- sqrt(DelayedMatrixStats::rowVars(ncounts))
    cv <- sd / m * 100
    df <- data.frame(log_dispersion = log(cv), log_mean = log(m))
    df[[id_tag]] <- if (software == "monocle3") fdat[[id_tag]] else rownames(cds)
    df <- df[is.finite(df$log_dispersion), ]
    
    model <- stats::lm(log_dispersion ~ log_mean + poly(log_mean, degree = q), data = df)
    prd <- data.frame(log_mean = df$log_mean)
    err <- suppressWarnings(stats::predict(model, newdata = prd, se.fit = TRUE))
    prd$fit <- err$fit
    prd$lci <- err$fit - 1.96 * err$se.fit
    prd$uci <- err$fit + 1.96 * err$se.fit
    prd$log_dispersion <- df$log_dispersion
    prd[[id_tag]] <- df[[id_tag]]
    
    if (software == "monocle3") {
      cds@int_metadata$dispersion <- prd
    } else if (software == "seurat") {
      cds@misc$dispersion <- prd
    }
    return(cds)
  } else if (method == "m2" && software == "monocle3") {
    # Monocle 2 method
    df <- calc_dispersion_m2(cds, min_cells_detected = 1, min_exprs = 1, id_tag = id_tag)
    fdat <- fData(cds)
    if (!is.list(df)) stop("Parametric dispersion fitting failed.")
    disp_table <- subset(df, !is.na(mu))
    res <- monocle::parametricDispersionFit(disp_table, verbose = TRUE)
    fit <- res[[1]]
    coefs <- res[[2]]
    if (remove_outliers) {
      CD <- cooks.distance(fit)
      cooks_cutoff <- 4 / nrow(disp_table)
      message(paste("Removing", length(CD[CD > cooks_cutoff]), "outliers"))
      outliers <- union(names(CD[CD > cooks_cutoff]), setdiff(row.names(disp_table), names(CD)))
      res <- monocle::parametricDispersionFit(disp_table[!(row.names(disp_table) %in% outliers), ], verbose = TRUE)
      fit <- res[[1]]
      coefs <- res[[2]]
      names(coefs) <- c("asymptDisp", "extraPois")
      ans <- function(q) coefs[1] + coefs[2] / q
      attr(ans, "coefficients") <- coefs
    }
    res <- list(disp_table = disp_table, disp_func = ans)
    cds@int_metadata$dispersion <- res
    return(cds)
  } else {
    stop("Invalid method or unsupported for the provided object.")
  }
}


#' @title Calculate Dispersion for Monocle 2
#' @description Helper function to calculate dispersion using Monocle 2 method.
#' @param obj A `cell_data_set` object.
#' @param min_cells_detected Integer specifying the minimum number of cells in which a gene must be detected. Default is 1.
#' @param min_exprs Numeric value specifying the minimum expression threshold. Default is 1.
#' @param id_tag The name of the gene identifier column. Default is "id".
#' @return A data frame containing dispersion estimates.
#' @export
#' @importFrom Matrix rowSums
#' @importFrom DelayedMatrixStats rowMeans2 rowVars
calc_dispersion_m2 <- function(obj, min_cells_detected = 1, min_exprs = 1, id_tag = "id") {
  if (!methods::is(obj, "cell_data_set")) {
    stop("Input object must be a 'cell_data_set'.")
  }
  
  rounded <- round(exprs(obj))
  nz_genes <- rowSums(rounded > min_exprs)
  nz_genes <- names(nz_genes[nz_genes > min_cells_detected])
  
  x <- DelayedArray::DelayedArray(t(t(rounded[nz_genes, ]) / pData(obj[nz_genes, ])$Size_Factor))
  xim <- mean(1 / pData(obj[nz_genes, ])$Size_Factor)
  
  f_expression_mean <- DelayedMatrixStats::rowMeans2(x)
  f_expression_var <- DelayedMatrixStats::rowVars(x)
  
  disp_guess <- (f_expression_var - xim * f_expression_mean) / (f_expression_mean^2)
  disp_guess[disp_guess < 0] <- 0
  
  res <- data.frame(
    mu = as.vector(f_expression_mean),
    disp = as.vector(disp_guess),
    stringsAsFactors = FALSE
  )
  res[res$mu == 0, c("mu", "disp")] <- NA
  res[[id_tag]] <- row.names(fData(obj[nz_genes, ]))
  return(res)
}

#' @title Find Common Features Across Single Cell Objects
#' @description Identifies common features (genes) across a list of `cell_data_set` or `Seurat` objects and subsets each object to include only these features.
#' @param cds_list A list of `cell_data_set` or `Seurat` objects.
#' @return A list of `cell_data_set` or `Seurat` objects with common features.
#' @export
common_features <- function(cds_list) {
  if (!is.list(cds_list) || length(cds_list) < 2) {
    stop("Input must be a list of at least two single-cell objects.")
  }
  
  software <- NULL
  feature_lists <- list()
  
  # Identify software and collect feature names
  for (i in seq_along(cds_list)) {
    cds <- cds_list[[i]]
    if (i == 1) {
      if (methods::is(cds, "Seurat")) {
        software <- "seurat"
      } else if (methods::is(cds, "cell_data_set")) {
        software <- "monocle3"
      } else {
        stop("Objects must be either 'Seurat' or 'cell_data_set'.")
      }
    } else {
      if ((software == "seurat" && !methods::is(cds, "Seurat")) ||
          (software == "monocle3" && !methods::is(cds, "cell_data_set"))) {
        stop("All objects must be of the same class.")
      }
    }
    
    if (software == "monocle3") {
      feature_lists[[i]] <- rownames(fData(cds))
    } else if (software == "seurat") {
      feature_lists[[i]] <- rownames(cds)
    }
  }
  
  # Find common features
  common_features <- Reduce(intersect, feature_lists)
  if (length(common_features) == 0) {
    stop("No common features found across the objects.")
  }
  
  # Subset each object
  for (i in seq_along(cds_list)) {
    cds <- cds_list[[i]]
    if (software == "monocle3") {
      cds_list[[i]] <- cds[common_features, ]
    } else if (software == "seurat") {
      counts <- GetAssayData(cds, slot = "counts")
      counts <- counts[common_features, ]
      seurat_obj <- CreateSeuratObject(counts = counts)
      seurat_obj@meta.data <- cds@meta.data
      cds_list[[i]] <- seurat_obj
    }
  }
  
  return(cds_list)
}
