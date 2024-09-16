#' @title Find Common Features Across Single-Cell Objects
#' @description Identifies common features (genes) across a list of `Seurat` or `cell_data_set` objects and subsets each object to include only these features.
#' @param cds_list A list of `Seurat` or `cell_data_set` objects.
#' @return A list of the same objects, each subset to include only the common features.
#' @export
common_features <- function(cds_list) {
  if (!is.list(cds_list) || length(cds_list) < 2) {
    stop("Input must be a list containing at least two single-cell objects.")
  }
  
  # Determine the class of the first object and ensure all objects are of the same class
  first_class <- class(cds_list[[1]])
  if (!first_class %in% c("Seurat", "cell_data_set")) {
    stop("Objects must be either 'Seurat' or 'cell_data_set' class.")
  }
  
  for (i in seq_along(cds_list)) {
    if (class(cds_list[[i]]) != first_class) {
      stop("All objects in the list must be of the same class.")
    }
  }
  
  # Get the initial set of features from the first object
  if (inherits(cds_list[[1]], "Seurat")) {
    common_features <- rownames(cds_list[[1]])
  } else {
    common_features <- rownames(fData(cds_list[[1]]))
  }
  
  # Find common features across all objects
  for (i in seq_along(cds_list)[-1]) {
    if (inherits(cds_list[[i]], "Seurat")) {
      features_i <- rownames(cds_list[[i]])
    } else {
      features_i <- rownames(fData(cds_list[[i]]))
    }
    common_features <- intersect(common_features, features_i)
    if (length(common_features) == 0) {
      stop("No common features found among the provided datasets.")
    }
  }
  
  # Subset each object to include only the common features
  for (i in seq_along(cds_list)) {
    if (inherits(cds_list[[i]], "Seurat")) {
      cds_list[[i]] <- subset(cds_list[[i]], features = common_features)
    } else {
      cds_list[[i]] <- cds_list[[i]][common_features, ]
    }
  }
  
  return(cds_list)
}



#' @title Perform TF-IDF Transformation on Single-Cell Data
#' @description Applies Term Frequency-Inverse Document Frequency (TF-IDF) transformation to a count matrix or a `cell_data_set` object.
#' @param input A count matrix (genes x cells) or a `cell_data_set` object.
#' @param method Integer specifying the method to use (1, 2, or 3). Default is 1.
#' @param scale_to Numeric value to scale the data (used in method 2). Default is 1e4.
#' @param verbose Logical indicating whether to display progress messages. Default is TRUE.
#' @return A transformed matrix or updated `cell_data_set` object.
#' @importFrom Matrix rowSums colSums Diagonal t
#' @export
tf_idf_transform <- function(input, method = 1, scale_to = 1e4, verbose = TRUE) {
  if (inherits(input, "cell_data_set")) {
    mat <- SingleCellExperiment::counts(input)
  } else if (is.matrix(input) || inherits(input, "dgCMatrix")) {
    mat <- input
  } else {
    stop("Input must be a count matrix or a 'cell_data_set' object.")
  }
  rn <- rownames(mat)
  cn <- colnames(mat)
  
  # Remove genes with zero total counts
  gene_totals <- rowSums(mat)
  nonzero_genes <- gene_totals > 0
  mat <- mat[nonzero_genes, ]
  gene_totals <- gene_totals[nonzero_genes]
  rn <- rn[nonzero_genes]
  
  # Column normalize (Term Frequency)
  col_sums <- colSums(mat)
  mat <- t(t(mat) / col_sums)
  
  # Compute Inverse Document Frequency
  if (verbose) message("Computing Inverse Document Frequency")
  if (method %in% c(1, 3)) {
    idf <- log(1 + ncol(mat) / gene_totals)
  } else if (method == 2) {
    idf <- ncol(mat) / gene_totals
  } else {
    stop("Invalid method selected. Choose 1, 2, or 3.")
  }
  
  # Apply TF-IDF transformation
  if (verbose) message("Computing TF-IDF Matrix")
  mat <- Diagonal(x = as.vector(idf)) %*% mat
  
  # Additional scaling for method 2
  if (method == 2) {
    mat@x <- log(mat@x * scale_to + 1)
  } else if (method == 3) {
    mat@x <- log(mat@x + 1)
  }
  
  # Restore row and column names
  rownames(mat) <- rn
  colnames(mat) <- cn
  
  if (inherits(input, "cell_data_set")) {
    SingleCellExperiment::counts(input) <- mat
    return(input)
  } else {
    return(mat)
  }
}


#' @title Add Random Noise to Data
#' @description Adds uniform random noise to a vector or matrix.
#' @param data A numeric vector or matrix.
#' @param amount Numeric value indicating the maximum amplitude of the noise. Default is 1e-4.
#' @return The input data with added noise.
#' @export
Noisify <- function(data, amount = 1e-4) {
  if (!is.numeric(data)) {
    stop("Data must be numeric.")
  }
  if (is.vector(data)) {
    noise <- runif(length(data), -amount, amount)
    noisified <- data + noise
  } else if (is.matrix(data)) {
    noise <- matrix(runif(length(data), -amount, amount), nrow = nrow(data))
    noisified <- data + noise
  } else {
    stop("Data must be a vector or matrix.")
  }
  return(noisified)
}


#' @title Calculate Bimodality Coefficient
#' @description Computes the bimodality coefficient of a numeric vector.
#' @param x A numeric vector.
#' @param finite Logical indicating whether to apply finite sample size correction. Default is TRUE.
#' @return The bimodality coefficient.
#' @export
bimodality_coefficient <- function(x, finite = TRUE) {
  n <- length(x)
  G <- skewness(x, finite)
  K <- kurtosis(x, finite)
  if (finite) {
    B <- ((G^2) + 1) / (K + ((3 * (n - 1)^2) / ((n - 2) * (n - 3))))
  } else {
    B <- ((G^2) + 1) / (K + 3)
  }
  return(B)
}

#' @title Calculate Skewness
#' @description Computes the skewness of a numeric vector.
#' @param x A numeric vector.
#' @param finite Logical indicating whether to apply finite sample size correction. Default is TRUE.
#' @return The skewness of the data.
#' @export
skewness <- function(x, finite = TRUE) {
  n <- length(x)
  m2 <- mean((x - mean(x))^2)
  m3 <- mean((x - mean(x))^3)
  S <- m3 / (m2^(3 / 2))
  if (finite && n > 2) {
    S <- S * sqrt(n * (n - 1)) / (n - 2)
  }
  return(S)
}

#' @title Calculate Kurtosis
#' @description Computes the kurtosis of a numeric vector.
#' @param x A numeric vector.
#' @param finite Logical indicating whether to apply finite sample size correction. Default is TRUE.
#' @return The excess kurtosis of the data.
#' @export
kurtosis <- function(x, finite = TRUE) {
  n <- length(x)
  m2 <- mean((x - mean(x))^2)
  m4 <- mean((x - mean(x))^4)
  K <- m4 / (m2^2) - 3
  if (finite && n > 3) {
    K <- ((n - 1) * ((n + 1) * K + 6)) / ((n - 2) * (n - 3))
  }
  return(K)
}