################################################################################
# FILE: R/deconvolute.R
# STATUS: Missing export on visualize_tumor_program
# ------------------------------------------------------------------------------
# Functions:
# [x] deconvolve_bulk           (Exported)
# [x] calculate_fit_metrics     (Exported - Internal)
# [x] plot_deconvolution        (Exported)
# [x] print_metrics_summary     (Exported - Internal)
# [X] extract_tumor_program     (Exported)
# [x] visualize_tumor_program   (In Progress)
# [ ] compare_tumor_consistency (In Progress)
# [ ] export_tumor_genes        (In Progress)
################################################################################



#' Deconvolve bulk RNA-seq data using single-cell signatures
#'
#' @param signatures Matrix of cellular signatures (genes × cell_types). 
#'   Each column should be normalized to sum to 1.
#' @param bulk_counts Matrix of bulk RNA-seq counts (genes × samples).
#'   Gene names must match signatures.
#' @param gene_lengths Numeric vector of gene lengths, same length as nrow(signatures).
#' @param gene_weights Optional numeric vector of gene weights (0-1), same length as 
#'   nrow(signatures). Default is all 1s.
#' @param backend Which compute backend to use: "ndarray" (CPU, default), 
#'   "wgpu" (GPU if compiled with wgpu feature), "cuda" (CUDA if available).
#' @param insert_size Insert size for RNA-seq fragments. Default 500.
#' @param init_log_exp Initial value for log exposures. Default -10.
#' @param learning_rate Learning rate for optimization. Default 0.01.
#' @param l1_lambda L1 regularization parameter. Default 0.0 (no regularization).
#' @param l2_lambda L2 regularization parameter. Default 0.0 (no regularization).
#' @param max_iter Maximum number of iterations. Default 10000.
#' @param poll_interval Check convergence every N iterations. Default 100.
#' @param ll_tolerance Log-likelihood convergence tolerance. Default 1e-6.
#' @param sparsity_tolerance Sparsity convergence tolerance. Default 1e-4.
#' @param verbose emit learning progress stats
#' @param method either default - gd (Poisson regression with gradient descent) or em (Expectation-Maximization algorithm)
#'
#' @return A list with:
#'   \item{exposures}{Matrix of cell type proportions (cell_types+intercept × samples)}
#'   \item{pred_counts}{Matrix of predicted bulk counts (genes × samples)}
#'   \item{proportions}{Normalized cell type proportions excluding intercept}
#'
#' @export
#' @examples
#' \dontrun{
#' # CPU backend (default)
#' result <- deconvolve_bulk(
#'   signatures = signatures,
#'   bulk_counts = bulk_counts,
#'   gene_lengths = gene_lengths
#' )
#' 
#' # GPU backend (if available)
#' result <- deconvolve_bulk(
#'   signatures = signatures,
#'   bulk_counts = bulk_counts,
#'   gene_lengths = gene_lengths,
#'   backend = "wgpu"
#' )
#' }
deconvolve_bulk <- function(
    signatures,
    bulk_counts,
    gene_lengths,
    gene_weights = NULL,
    backend = "wgpu",
    insert_size = 500,
    init_log_exp = -10,
    learning_rate = 0.01,
    l1_lambda = 0.0,
    l2_lambda = 0.0,
    max_iter = 10000,
    poll_interval = 100,
    ll_tolerance = 1e-6,
    sparsity_tolerance = 1e-4,
    verbose = TRUE,
    method = c("gd", "em")
) {
  method <- match.arg(method)
  # Input validation
  if (!is.matrix(signatures)) {
    stop("signatures must be a matrix")
  }
  if (!is.matrix(bulk_counts)) {
    stop("bulk_counts must be a matrix")
  }
  if (!is.numeric(gene_lengths)) {
    stop("gene_lengths must be numeric")
  }
  
  n_genes <- nrow(signatures)
  n_cell_types <- ncol(signatures)
  
  # Check dimensions
  if (nrow(bulk_counts) != n_genes) {
    stop("Number of genes in bulk_counts (", nrow(bulk_counts), 
         ") must match signatures (", n_genes, ")")
  }
  
  if (length(gene_lengths) != n_genes) {
    stop("Length of gene_lengths (", length(gene_lengths), 
         ") must match number of genes (", n_genes, ")")
  }
  
  # Check that signatures are normalized
  col_sums <- colSums(signatures)
  if (any(abs(col_sums - 1.0) > 1e-6)) {
    warning("Some signature columns don't sum to 1. Normalizing...")
    message("  Signature column sum range: [", 
        round(min(col_sums), 6), ", ", 
        round(max(col_sums), 6), "]\n", sep="")
    signatures <- sweep(signatures, 2, colSums(signatures), "/")
    message("  Signatures normalized. New range: [", 
        round(min(colSums(signatures)), 6), ", ", 
        round(max(colSums(signatures)), 6), "]\n", sep="")
  }
  
  
  # Default gene weights
  if (is.null(gene_weights)) {
    gene_weights <- rep(1.0, n_genes)
  }
  
  if (length(gene_weights) != n_genes) {
    stop("Length of gene_weights (", length(gene_weights), 
         ") must match number of genes (", n_genes, ")")
  }
  
  # Ensure gene weights are in [0, 1]
  if (any(gene_weights < 0) || any(gene_weights > 1)) {
    stop("gene_weights must be in [0, 1]")
  }
  
  # Validate backend
  valid_backends <- c("ndarray", "cpu", "wgpu", "gpu")
  if (!backend %in% valid_backends) {
    warning("Invalid backend '", backend, "'. Using 'ndarray' (CPU). ",
            "Valid options: ", paste(valid_backends, collapse = ", "))
    backend <- "ndarray"
  }
  
  # Call Rust function
  message("Running deconvolution...\n")
  if(method=="gd"){
    result <- fit_deconv(
    signatures,
    bulk_counts * 1.0,
    gene_lengths * 1.0,
    gene_weights * 1.0,
    backend,
    insert_size,
    init_log_exp,
    learning_rate,
    l1_lambda,
    l2_lambda,
    max_iter,
    poll_interval,
    ll_tolerance,
    sparsity_tolerance,
    verbose)
  }
  
  if(method=="em"){
    result <- fit_deconvolution_em(
      signatures,
      bulk_counts * 1.0,
      gene_lengths * 1.0,
      gene_weights * 1.0,
      max_iter,
      ll_tolerance,
      l1_lambda,
      verbose)
  }
  # Add row and column names
  n_samples <- ncol(bulk_counts)
  
  # Exposures: (n_cell_types + 1) × n_samples
  rownames(result$exposures) <- c(colnames(signatures), "Intercept")
  colnames(result$exposures) <- colnames(bulk_counts)
  
  # Predicted counts: n_genes × n_samples
  rownames(result$pred_counts) <- rownames(signatures)
  colnames(result$pred_counts) <- colnames(bulk_counts)
  
  # Calculate normalized proportions (excluding intercept)
  exposures_no_intercept <- result$exposures[1:n_cell_types, , drop = FALSE]
  proportions <- sweep(exposures_no_intercept, 2, 
                       colSums(exposures_no_intercept), "/")
  
  # Calculate proportions including intercept (for comparison)
  proportions_with_intercept <- sweep(result$exposures, 2,
                                      colSums(result$exposures), "/")
  
  # Calculate intercept fraction
  intercept_fraction <- proportions_with_intercept[n_cell_types + 1, ]
  
  # Add proportions to result
  result$proportions <- proportions
  result$proportions_with_intercept <- proportions_with_intercept
  result$intercept_fraction <- intercept_fraction
  
  # Calculate goodness of fit metrics
  result$metrics <- calculate_fit_metrics(bulk_counts, result$pred_counts, gene_weights)
  
  message("Done!\n")
  return(result)
}

#' Calculate goodness of fit metrics for deconvolution
#' @param observed Matrix of observed counts (genes × samples)
#' @param predicted Matrix of predicted counts (genes × samples)
#' @export
#' @keywords internal
calculate_fit_metrics <- function(observed, predicted, gene_weights) {
  # FILTER to only weighted genes
  keep_genes <- gene_weights > 0
  observed <- observed[keep_genes, ]
  predicted <- predicted[keep_genes, ]
  
  # Now calculate metrics on filtered data
  obs_total <- colSums(observed)
  pred_total <- colSums(predicted)
  
  # Global correlation (all values for weighted genes only)
  global_correlation <- cor(
    as.vector(observed), 
    as.vector(predicted), 
    method = "pearson"
  )
  
  # Per-sample correlation
  gene_correlations <- sapply(1:ncol(observed), function(i) {
    cor(observed[, i], predicted[, i], method = "pearson")
  })
  
  
  # === Sample-wise correlation (across samples per gene) ===
  # This asks: "Do expression patterns across samples match for each gene?"
  sample_correlations <- sapply(1:nrow(observed), function(i) {
    if (sd(observed[i, ]) < 1e-10) return(NA)  # Skip genes with no variance
    cor(observed[i, ], predicted[i, ], method = "pearson")
  })
  sample_correlations <- sample_correlations[!is.na(sample_correlations)]
  
  # RMSE per sample
  rmse <- sapply(1:ncol(observed), function(i) {
    sqrt(mean((observed[, i] - predicted[, i])^2))
  })
  
  # Normalized RMSE (as % of mean observed count)
  nrmse <- sapply(1:ncol(observed), function(i) {
    sqrt(mean((observed[, i] - predicted[, i])^2)) / mean(observed[, i])
  })
  
  # === R-squared (standard definition) ===
  # Proportion of variance explained
  ss_total <- sum((observed - mean(observed))^2)
  ss_residual <- sum((observed - predicted)^2)
  r_squared <- 1 - (ss_residual / ss_total)
  
  # === Pseudo-R-squared (Poisson deviance-based) ===
  # Better for count data
  null_deviance <- sum(sapply(1:ncol(observed), function(i) {
    obs <- observed[, i]
    null_pred <- rep(mean(obs), length(obs))
    # Poisson deviance
    dev <- 2 * sum(obs * log((obs + 1e-10) / (null_pred + 1e-10)) - (obs - null_pred))
    dev
  }))
  
  full_deviance <- sum(sapply(1:ncol(observed), function(i) {
    obs <- observed[, i]
    pred <- predicted[, i] + 1e-10
    # Poisson deviance
    dev <- 2 * sum(obs * log((obs + 1e-10) / pred) - (obs - pred))
    dev
  }))
  
  pseudo_r2 <- 1 - (full_deviance / null_deviance)
  
  # === Mean Absolute Percentage Error ===
  mape <- mean(abs((observed - predicted) / (observed + 1))) * 100
  
  list(
    n_genes_used = sum(keep_genes),
    n_genes_excluded = sum(!keep_genes),
    # Per-sample metrics
    gene_correlation = gene_correlations,  # Renamed for clarity
    rmse = rmse,
    nrmse = nrmse,
    count_ratio = pred_total / obs_total,
    
    # Global metrics  
    global_correlation = global_correlation,
    r_squared = r_squared,
    pseudo_r2 = pseudo_r2,
    mape = mape,
    
    # Summary statistics
    mean_gene_correlation = mean(gene_correlations),
    median_gene_correlation = median(gene_correlations),
    mean_sample_correlation = mean(sample_correlations, na.rm = TRUE),
    mean_rmse = mean(rmse),
    mean_nrmse = mean(nrmse)
  )
}

#' Plot deconvolution results
#'
#' @param result Output from deconvolve_bulk()
#' @param observed_counts Original bulk counts matrix (for fit plots)
#' @param type Type of plot: "proportions", "fit", "residuals", or "all"
#' @export

plot_deconvolution <- function(result, observed_counts = NULL, type = "all") {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 is required for plotting")
  }
  
  library(ggplot2)
  plots <- list()
  
  # === 1. Cell Type Proportions ===
  if (type %in% c("proportions", "all")) {
    prop_df <- as.data.frame(result$proportions)
    prop_df$CellType <- rownames(prop_df)
    
    # Remove intercept for cleaner visualization
    if ("Intercept" %in% prop_df$CellType) {
      intercept_row <- prop_df[prop_df$CellType == "Intercept", ]
      prop_df <- prop_df[prop_df$CellType != "Intercept", ]
    }
    
    prop_long <- tidyr::pivot_longer(
      prop_df, 
      -CellType, 
      names_to = "Sample", 
      values_to = "Proportion"
    )
    
    p1 <- ggplot(prop_long, aes(x = Sample, y = Proportion, fill = CellType)) +
      geom_bar(stat = "identity") +
      theme_minimal() +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        legend.position = "right"
      ) +
      labs(
        title = "Cell Type Proportions (excluding Intercept)", 
        y = "Proportion", 
        x = "Sample"
      ) 
    
    plots$proportions <- p1
    if (type == "proportions") print(p1)
  }
  
  # === 2. Model Fit Quality ===
  if (type %in% c("fit", "all") && !is.null(observed_counts)) {
    # FIXED: Use actual observed counts, not predicted twice
    obs_flat <- as.vector(observed_counts)
    pred_flat <- as.vector(result$pred_counts)
    
    # Remove zeros for log scale
    keep_idx <- obs_flat > 0 & pred_flat > 0
    
    fit_df <- data.frame(
      Observed = obs_flat[keep_idx],
      Predicted = pred_flat[keep_idx]
    )
    
    # Calculate correlation for title
    fit_cor <- cor(fit_df$Observed, fit_df$Predicted)
    
    p2 <- ggplot(fit_df, aes(x = log1p(Observed), y = log1p(Predicted))) +
      geom_hex(bins = 100) +
      geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", size = 1) +
      theme_minimal() +
      scale_fill_viridis_c(trans = "log10") +
      labs(
        title = sprintf("Model Fit Quality (r = %.3f)", fit_cor),
        x = "log1p(Observed counts)",
        y = "log1p(Predicted counts)"
      ) +
      theme(legend.position = "right")
    
    plots$fit <- p2
    if (type == "fit") print(p2)
  }
  
  # === 3. Per-Sample Correlation ===
  if (type %in% c("correlation", "all") && !is.null(result$metrics)) {
    cor_df <- data.frame(
      Sample = names(result$metrics$count_ratio),
      Correlation = result$metrics$gene_correlation
    )
    
    p3 <- ggplot(cor_df, aes(x = Sample, y = Correlation)) +
      geom_bar(stat = "identity", fill = "steelblue") +
      geom_hline(yintercept = mean(cor_df$Correlation), 
                 color = "red", linetype = "dashed") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8)) +
      labs(
        title = "Per-Sample Gene Correlation",
        subtitle = sprintf("Mean = %.3f", mean(cor_df$Correlation)),
        y = "Pearson Correlation",
        x = "Sample"
      ) +
      ylim(0, 1)
    
    plots$correlation <- p3
    if (type == "correlation") print(p3)
  }
  
  # === 4. Residuals ===
  if (type %in% c("residuals", "all") && !is.null(observed_counts)) {
    residuals <- observed_counts - result$pred_counts
    rel_residuals <- residuals / (observed_counts + 1)  # Relative residuals
    
    # Sample a subset for plotting (too many points otherwise)
    n_points <- min(50000, length(rel_residuals))
    sample_idx <- sample(length(rel_residuals), n_points)
    
    resid_df <- data.frame(
      Predicted = as.vector(result$pred_counts)[sample_idx],
      Residual = as.vector(rel_residuals)[sample_idx]
    )
    
    p4 <- ggplot(resid_df, aes(x = log1p(Predicted), y = Residual)) +
      geom_hex(bins = 100) +
      geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
      theme_minimal() +
      scale_fill_viridis_c(trans = "log10") +
      labs(
        title = "Residual Plot",
        subtitle = "Should be centered around 0 with no patterns",
        x = "log1p(Predicted counts)",
        y = "Relative Residual (obs-pred)/(obs+1)"
      ) +
      ylim(-2, 2)
    
    plots$residuals <- p4
    if (type == "residuals") print(p4)
  }
  
  # === 5. Intercept proportion ===
  if (type %in% c("intercept", "all")) {
    # Show how much signal is captured by intercept vs cell types
    exp_mat <- result$exposures
    if ("Intercept" %in% rownames(exp_mat)) {
      intercept_prop <- exp_mat["Intercept", ] / colSums(exp_mat)
      
      int_df <- data.frame(
        Sample = names(intercept_prop),
        InterceptProportion = intercept_prop
      )
      
      p5 <- ggplot(int_df, aes(x = Sample, y = InterceptProportion)) +
        geom_bar(stat = "identity", fill = "coral") +
        geom_hline(yintercept = 0.5, color = "red", linetype = "dashed") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8)) +
        labs(
          title = "Intercept Contribution",
          subtitle = "High values suggest unexplained signal",
          y = "Intercept / Total Exposure",
          x = "Sample"
        ) +
        ylim(0, 1)
      
      plots$intercept <- p5
      if (type == "intercept") print(p5)
    }
  }
  
  # Print all plots if type = "all"
  if (type == "all") {
    for (p in plots) {
      print(p)
    }
  }
  
  invisible(plots)
}

#' Print summary of deconvolution metrics
#' @export
#' @keywords internal
print_metrics_summary <- function(result) {
  if (is.null(result$metrics)) {
    message("No metrics available. Run calculate_fit_metrics() first.\n")
    return(invisible(NULL))
  }
  
  m <- result$metrics
  
  # cat("=== Deconvolution Fit Metrics ===\n\n")
  # 
  # cat("Global Fit:\n")
  # cat(sprintf("  Overall correlation: %.4f\n", m$global_correlation))
  # cat(sprintf("  R-squared: %.4f\n", m$r_squared))
  # cat(sprintf("  Pseudo R-squared: %.4f\n", m$pseudo_r2))
  # cat(sprintf("  Mean absolute percent error: %.2f\n", m$mape))
  # cat("\n")
  # 
  # cat("Per-Sample Metrics:\n")
  # cat(sprintf("  Mean gene correlation: %.4f (range: %.3f - %.3f)\n", 
  #             m$mean_gene_correlation,
  #             min(m$gene_correlation),
  #             max(m$gene_correlation)))
  # cat(sprintf("  Median gene correlation: %.4f\n", m$median_gene_correlation))
  # cat(sprintf("  Mean RMSE: %.2f\n", m$mean_rmse))
  # cat(sprintf("  Mean normalized RMSE: %.2f%%\n", m$mean_nrmse * 100))
  # cat("\n")
  # 
  # cat("Sample-wise Metrics:\n")
  # cat(sprintf("  Mean sample correlation: %.4f\n", m$mean_sample_correlation))
  # cat("\n")
  # 
  # cat("Count Statistics:\n")
  # cat(sprintf("  Count ratio range: %.3f - %.3f\n", 
  #             min(m$count_ratio), max(m$count_ratio)))
  # cat(sprintf("  Mean count ratio: %.3f (should be ~1.0)\n", 
  #             mean(m$count_ratio)))
  # cat("\n")
  # 
  # # Flag potential issues
  # if (m$global_correlation < 0.7) {
  #   cat("⚠️  WARNING: Low global correlation suggests poor fit\n")
  # }
  # if (m$mean_gene_correlation < 0.3) {
  #   cat("ℹ️  INFO: Low per-sample correlation is normal for deconvolution\n")
  #   cat("   (Gene patterns within samples are partially captured)\n")
  # }
  # if (any(abs(m$count_ratio - 1) > 0.1)) {
  #   cat("⚠️  WARNING: Some samples have >10% total count mismatch\n")
  # }
  # if (!is.null(m$pseudo_r2) && m$pseudo_r2 < 0.5) {
  #   cat("⚠️  WARNING: Low pseudo R-squared suggests model doesn't improve over null\n")
  # }
}



# #' Extract Tumor-Specific Transcriptional Program from Deconvolution
# #'
# #' This function analyzes the intercept term from deconvolution to identify
# #' genes that comprise the tumor-specific program - i.e., genes whose expression
# #' cannot be explained by normal hematopoietic cell types.
# #'
# #' @param deconv_result List returned from fit_deconv_em() containing 'exposures' and 'pred_counts'
# #' @param signatures Matrix of cell type signatures used in deconvolution %\[%\genes x cell_types%\[%\
# #' @param bulk_counts Matrix of observed bulk expression %\[%\genes x samples%\[%\
# #' @param gene_names Optional vector of gene names (default: rownames of signatures)
# #' @param top_n Number of top genes to return (default: 500)
# #' @param return_all If TRUE, return all genes ranked; if FALSE, return top_n (default: FALSE)
# #'
# #' @return A list containing:
# #'   - tumor_upregulated: Genes overexpressed in tumor vs normal hematopoiesis
# #'   - tumor_downregulated: Genes underexpressed in tumor vs normal hematopoiesis
# #'   - intercept_values: Per-sample intercept (noise) levels
# #'   - residuals: Full residual matrix %\[%\genes x samples%\[%\
# #'   - mean_residual: Mean residual per gene across samples
# #'   - intercept_fraction: Fraction of total signal from intercept per sample
# #'
# #' @examples
# #' result <- fit_deconv_em(sigs, bulk, gene_lengths, w_vec, max_iter, tol, l1, verbose)
# #' tumor_program <- extract_tumor_program(result, sigs, bulk)
# #' 
# #' # Top tumor-upregulated genes
# #' head(tumor_program$tumor_upregulated)
# #' 
# #' # Visualize
# #' plot(tumor_program$mean_residual, 
# #'      main = "Tumor-Specific Expression Profile",
# #'      xlab = "Gene Index", ylab = "Mean Residual")
# extract_tumor_program <- function(deconv_result, 
#                                   signatures, 
#                                   bulk_counts,
#                                   gene_names = NULL,
#                                   top_n = 500,
#                                   return_all = FALSE) {
  
#   # Extract components
#   exposures <- deconv_result$exposures
#   n_celltypes <- nrow(exposures) - 1
#   n_samples <- ncol(exposures)
#   n_genes <- nrow(signatures)
  
#   # Get gene names
#   if (is.null(gene_names)) {
#     gene_names <- rownames(signatures)
#     if (is.null(gene_names)) {
#       gene_names <- paste0("Gene_", 1:n_genes)
#     }
#   }
  
#   # Separate normal cell type exposures from intercept
#   normal_exposures <- exposures[1:n_celltypes, , drop = FALSE]
#   intercept_row <- exposures[nrow(exposures), , drop = FALSE]
  
#   # Compute predicted counts from NORMAL cell types only
#   normal_predicted <- signatures %*% normal_exposures
  
#   # Compute RESIDUAL = Observed - Normal Predicted
#   # This is what the intercept is capturing (tumor-specific signal)
#   residual <- bulk_counts - normal_predicted
  
#   # Mean residual per gene across samples
#   mean_residual <- rowMeans(residual)
#   names(mean_residual) <- gene_names
  
#   # Standard deviation of residual (consistency across samples)
#   sd_residual <- apply(residual, 1, sd)
#   names(sd_residual) <- gene_names
  
#   # Rank genes by residual
#   ranked_indices <- order(mean_residual, decreasing = TRUE)
  
#   # Identify tumor-upregulated genes (positive residual)
#   tumor_up_indices <- ranked_indices[mean_residual[ranked_indices] > 0]
#   tumor_up_genes <- if (return_all || length(tumor_up_indices) <= top_n) {
#     tumor_up_indices
#   } else {
#     tumor_up_indices[1:min(top_n, length(tumor_up_indices))]
#   }
  
#   # Identify tumor-downregulated genes (negative residual)
#   tumor_down_indices <- ranked_indices[mean_residual[ranked_indices] < 0]
#   tumor_down_indices <- rev(tumor_down_indices)  # Most negative first
#   tumor_down_genes <- if (return_all || length(tumor_down_indices) <= top_n) {
#     tumor_down_indices
#   } else {
#     tumor_down_indices[1:min(top_n, length(tumor_down_indices))]
#   }
  
#   # Compute intercept statistics
#   intercept_mean <- mean(intercept_row)
#   intercept_per_sample <- as.vector(intercept_row)
  
#   # Total signal per sample (from normal cells)
#   total_normal_signal <- colSums(normal_exposures)
  
#   # Fraction of signal from intercept (noise/tumor)
#   intercept_fraction <- intercept_per_sample / (total_normal_signal + intercept_per_sample)
  
#   # Create output data frames
#   tumor_upregulated_df <- data.frame(
#     gene_index = tumor_up_genes,
#     gene_name = gene_names[tumor_up_genes],
#     mean_residual = mean_residual[tumor_up_genes],
#     sd_residual = sd_residual[tumor_up_genes],
#     rank = 1:length(tumor_up_genes),
#     stringsAsFactors = FALSE
#   )
  
#   tumor_downregulated_df <- data.frame(
#     gene_index = tumor_down_genes,
#     gene_name = gene_names[tumor_down_genes],
#     mean_residual = mean_residual[tumor_down_genes],
#     sd_residual = sd_residual[tumor_down_genes],
#     rank = 1:length(tumor_down_genes),
#     stringsAsFactors = FALSE
#   )
  
#   # Return results
#   list(
#     tumor_upregulated = tumor_upregulated_df,
#     tumor_downregulated = tumor_downregulated_df,
#     intercept_values = intercept_per_sample,
#     intercept_mean = intercept_mean,
#     intercept_fraction = intercept_fraction,
#     residuals = residual,
#     mean_residual = mean_residual,
#     sd_residual = sd_residual,
#     normal_predicted = normal_predicted
#   )
# }


# #' Visualize Tumor Program
# #'
# #' Create diagnostic plots for tumor-specific program analysis
# #'
# #' @param tumor_program Output from extract_tumor_program()
# #' @param bulk_counts Original bulk expression matrix
# #' @param main_title Main title for plots
# visualize_tumor_program <- function(tumor_program, 
#                                     bulk_counts = NULL,
#                                     main_title = "Tumor-Specific Program Analysis") {
  
#   par(mfrow = c(2, 2))
  
#   # 1. Distribution of residuals
#   hist(tumor_program$mean_residual, 
#        breaks = 50,
#        main = "Distribution of Gene Residuals",
#        xlab = "Mean Residual (Tumor - Normal)",
#        col = "lightblue",
#        border = "darkblue")
#   abline(v = 0, col = "red", lwd = 2, lty = 2)
  
#   # 2. Intercept fraction per sample
#   plot(tumor_program$intercept_fraction, 
#        type = "b",
#        main = "Intercept Contribution per Sample",
#        xlab = "Sample Index",
#        ylab = "Fraction of Signal from Intercept",
#        pch = 19,
#        col = "darkred")
#   abline(h = mean(tumor_program$intercept_fraction), 
#          col = "blue", lwd = 2, lty = 2)
  
#   # 3. Top tumor genes - mean vs SD
#   top_genes <- rbind(
#     head(tumor_program$tumor_upregulated, 100),
#     head(tumor_program$tumor_downregulated, 100)
#   )
  
#   plot(top_genes$mean_residual,
#        top_genes$sd_residual,
#        pch = 19,
#        col = ifelse(top_genes$mean_residual > 0, "red", "blue"),
#        main = "Tumor Gene Consistency",
#        xlab = "Mean Residual",
#        ylab = "SD Residual")
#   legend("topright", 
#          legend = c("Upregulated", "Downregulated"),
#          col = c("red", "blue"),
#          pch = 19)
  
#   # 4. Residual heatmap preview (if bulk counts provided)
#   if (!is.null(bulk_counts)) {
#     top_up <- head(tumor_program$tumor_upregulated$gene_index, 25)
#     top_down <- head(tumor_program$tumor_downregulated$gene_index, 25)
#     top_50 <- c(top_up, top_down)
    
#     residual_subset <- tumor_program$residuals[top_50, ]
    
#     # Simple heatmap
#     image(t(residual_subset),
#           main = "Top 50 Tumor Genes (Residuals)",
#           xlab = "Sample",
#           ylab = "Gene",
#           col = colorRampPalette(c("blue", "white", "red"))(50),
#           axes = FALSE)
#     axis(1, at = seq(0, 1, length.out = ncol(residual_subset)), 
#          labels = 1:ncol(residual_subset))
#   }
  
#   par(mfrow = c(1, 1))
# }


# #' Compare Tumor Program Across Samples
# #'
# #' Identify genes that are consistently tumor-specific across all samples
# #' vs those that vary between samples
# #'
# #' @param tumor_program Output from extract_tumor_program()
# #' @param consistency_threshold Genes with CV < this are "consistent" (default: 0.5)
# #'
# #' @return List with consistent and variable tumor genes
# compare_tumor_consistency <- function(tumor_program, 
#                                       consistency_threshold = 0.5) {
  
#   residuals <- tumor_program$residuals
#   mean_res <- tumor_program$mean_residual
#   sd_res <- tumor_program$sd_residual
  
#   # Coefficient of variation
#   cv <- abs(sd_res / (mean_res + 1e-10))
  
#   # Consistent genes (low CV)
#   consistent_up <- tumor_program$tumor_upregulated$gene_index[
#     cv[tumor_program$tumor_upregulated$gene_index] < consistency_threshold
#   ]
  
#   consistent_down <- tumor_program$tumor_downregulated$gene_index[
#     cv[tumor_program$tumor_downregulated$gene_index] < consistency_threshold
#   ]
  
#   # Variable genes (high CV)
#   variable_up <- tumor_program$tumor_upregulated$gene_index[
#     cv[tumor_program$tumor_upregulated$gene_index] >= consistency_threshold
#   ]
  
#   variable_down <- tumor_program$tumor_downregulated$gene_index[
#     cv[tumor_program$tumor_downregulated$gene_index] >= consistency_threshold
#   ]
  
#   list(
#     consistent_upregulated = consistent_up,
#     consistent_downregulated = consistent_down,
#     variable_upregulated = variable_up,
#     variable_downregulated = variable_down,
#     cv = cv
#   )
# }


# #' Export Tumor Gene Lists
# #'
# #' Export tumor gene lists for downstream analysis (e.g., pathway enrichment)
# #'
# #' @param tumor_program Output from extract_tumor_program()
# #' @param output_prefix File prefix for output files
# export_tumor_genes <- function(tumor_program, output_prefix = "tumor_genes") {
  
#   # Write upregulated genes
#   write.table(tumor_program$tumor_upregulated,
#               file = paste0(output_prefix, "_upregulated.txt"),
#               sep = "\t",
#               quote = FALSE,
#               row.names = FALSE)
  
#   # Write downregulated genes
#   write.table(tumor_program$tumor_downregulated,
#               file = paste0(output_prefix, "_downregulated.txt"),
#               sep = "\t",
#               quote = FALSE,
#               row.names = FALSE)
  
#   # Write summary statistics
#   summary_stats <- data.frame(
#     metric = c("Mean_Intercept", 
#                "Mean_Intercept_Fraction",
#                "N_Upregulated", 
#                "N_Downregulated"),
#     value = c(tumor_program$intercept_mean,
#               mean(tumor_program$intercept_fraction),
#               nrow(tumor_program$tumor_upregulated),
#               nrow(tumor_program$tumor_downregulated))
#   )
  
#   write.table(summary_stats,
#               file = paste0(output_prefix, "_summary.txt"),
#               sep = "\t",
#               quote = FALSE,
#               row.names = FALSE)
  
#   message("Exported tumor gene lists to:")
#   message("  - ", paste0(output_prefix, "_upregulated.txt"))
#   message("  - ", paste0(output_prefix, "_downregulated.txt"))
#   message("  - ", paste0(output_prefix, "_summary.txt"))
# }


# # Example usage:
# # --------------
# # 
# # # Run deconvolution
# # result <- fit_deconv_em(sigs, bulk, gene_lengths, w_vec, 
# #                         max_iter = 1000, tolerance = 1e-6, 
# #                         l1_lambda = 0.0, verbose = TRUE)
# # 
# # # Extract tumor program
# # tumor_prog <- extract_tumor_program(result, sigs, bulk, 
# #                                      gene_names = rownames(sigs),
# #                                      top_n = 500)
# # 
# # # View top tumor genes
# # head(tumor_prog$tumor_upregulated, 20)
# # head(tumor_prog$tumor_downregulated, 20)
# # 
# # # Visualize
# # visualize_tumor_program(tumor_prog, bulk)
# # 
# # # Check consistency
# # consistency <- compare_tumor_consistency(tumor_prog)
# # length(consistency$consistent_upregulated)  # Core tumor program
# # 
# # # Export for pathway analysis
# # export_tumor_genes(tumor_prog, "AML_tumor_program")
