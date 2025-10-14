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
