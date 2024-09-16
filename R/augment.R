#' @title Augment Underrepresented Cell Groups in a Seurat Object
#' @description This function augments a Seurat object by simulating additional cells for underrepresented cell groups, aiming to balance the dataset such that each group has at least `norm_number` of cells. Optionally, overrepresented groups can be downsampled to `norm_number`.
#' @param obj A Seurat object.
#' @param column The name of the metadata column that designates cell groups (e.g., cell type).
#' @param norm_number The target number of cells for each group. Default is 2000.
#' @param assay The assay to use. Default is "RNA".
#' @param prune Logical indicating whether to downsample groups with more than `norm_number` cells to `norm_number`. Default is `FALSE`.
#' @return A Seurat object augmented with simulated cells such that all cell groups have at least `norm_number` cells.
#' @importFrom pbmcapply pbmclapply
#' @importFrom parallel detectCores
#' @importFrom Seurat CreateSeuratObject GetAssayData
#' @export
augment_data <- function(obj, column, norm_number = 2000, assay = "RNA", prune = FALSE) {
  # Validate inputs
  if (!inherits(obj, "Seurat")) {
    stop("The 'obj' parameter must be a Seurat object.")
  }
  if (!column %in% colnames(obj@meta.data)) {
    stop("The specified 'column' does not exist in the metadata of the Seurat object.")
  }
  if (!assay %in% names(obj@assays)) {
    stop("The specified 'assay' does not exist in the Seurat object.")
  }
  
  message("Extracting data for the selected assay")
  # Extract counts and metadata
  counts <- GetAssayData(obj, assay = assay, slot = "counts")
  meta_data <- obj@meta.data
  
  # Ensure counts and metadata are aligned
  if (!all(colnames(counts) == rownames(meta_data))) {
    stop("Column names of counts and row names of metadata do not match.")
  }
  
  # Get group labels
  group_labels <- as.character(meta_data[[column]])
  group_levels <- unique(group_labels)
  
  # Compute balance (cells needed or to be removed per group)
  group_counts <- table(group_labels)
  balance <- norm_number - group_counts
  names(balance) <- names(group_counts)
  
  # Prune overrepresented groups if requested
  if (prune) {
    message("Pruning overrepresented cell groups")
    overabundant_groups <- names(balance)[balance < 0]
    for (group in overabundant_groups) {
      group_cells <- which(group_labels == group)
      num_to_remove <- length(group_cells) - norm_number
      set.seed(123)  # For reproducibility
      cells_to_remove <- sample(group_cells, num_to_remove)
      counts <- counts[, -cells_to_remove]
      meta_data <- meta_data[-cells_to_remove, , drop = FALSE]
      group_labels <- group_labels[-cells_to_remove]
    }
  }
  
  # Update balance after pruning
  group_counts <- table(group_labels)
  balance <- norm_number - group_counts
  names(balance) <- names(group_counts)
  
  # Identify groups that need augmentation
  groups_to_augment <- names(balance)[balance > 0]
  
  if (length(groups_to_augment) == 0) {
    message("No groups need augmentation. Returning the original Seurat object.")
    obj@assays[[assay]]@counts <- counts
    obj@meta.data <- meta_data
    return(obj)
  }
  
  message("Simulating cells for underrepresented groups")
  # Prepare for simulation
  universe_genes <- rownames(counts)
  
  # Simulate cells for each underrepresented group
  sim_data_list <- pbmcapply::pbmclapply(groups_to_augment, function(group) {
    N <- balance[group]  # Number of cells to simulate
    group_cells <- which(group_labels == group)
    if (length(group_cells) < 2) {
      stop(paste("Not enough cells to simulate for group:", group))
    }
    # Get counts for the group
    group_counts_mat <- counts[, group_cells, drop = FALSE]
    # Get total counts per gene (row sums)
    gene_totals <- rowSums(group_counts_mat)
    # Get cell sizes (column sums)
    cell_sizes <- colSums(group_counts_mat)
    # Estimate density of cell sizes
    den <- density(cell_sizes)
    # Sample new cell sizes
    newsizes <- sample(cell_sizes, N, replace = TRUE) + rnorm(N, 0, den$bw)
    newsizes <- round(newsizes)
    # Ensure newsizes are within reasonable bounds
    newsizes <- newsizes[newsizes > min(cell_sizes) & newsizes < max(cell_sizes)]
    if (length(newsizes) < N) {
      newsizes <- rep(round(mean(cell_sizes)), N)
    } else {
      newsizes <- sample(newsizes, N, replace = TRUE)
    }
    # Create splat vector (gene names replicated by total counts)
    splat <- rep(universe_genes, times = gene_totals)
    # Simulate cells
    sim_cells <- lapply(newsizes, function(size) {
      sampled_genes <- sample(splat, size, replace = TRUE)
      gene_counts <- table(sampled_genes)
      counts_vector <- numeric(length(universe_genes))
      names(counts_vector) <- universe_genes
      counts_vector[names(gene_counts)] <- as.numeric(gene_counts)
      counts_vector
    })
    sim_mat <- do.call(cbind, sim_cells)
    colnames(sim_mat) <- paste0("simcell_", group, "_", seq_len(N))
    return(sim_mat)
  }, mc.cores = min(parallel::detectCores(), length(groups_to_augment)))
  
  # Combine simulated data
  sim_counts <- do.call(cbind, sim_data_list)
  sim_meta_data <- data.frame(row.names = colnames(sim_counts))
  sim_meta_data[[column]] <- sub("^simcell_([^_]+)_.*$", "\\1", colnames(sim_counts))
  
  # Merge simulated data with original data
  all_counts <- cbind(counts, sim_counts)
  all_meta_data <- rbind(meta_data, sim_meta_data)
  
  # Create new Seurat object
  augmented_obj <- CreateSeuratObject(counts = all_counts, meta.data = all_meta_data)
  
  return(augmented_obj)
}
