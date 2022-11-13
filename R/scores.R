#' Calculates geneset scores using totals
#'
#' @description Geneset scores are a score calculated for each single cell derived from more than one gene.  
#' In this implementation, the sum of the size-factor corrected, log-normalized gene expression for a give set of genes
#' is performed
#' 

#' @param cds Input cell_data_set object.
#' @param marker_set1 Vector of genes in the gene_metadata DataFrame (fData) that can be found in the column 'fData_col'
#' @param fData_col Character string denoting the gene_metadata DataFrame (fData) column that contains the genes in marker_set1.  Default = 'gene_short_name'
#' @return Vector of single cell scores that are derived from the sum of all gene expression values in the geneset
#' that were size factor corrected and log-normalized.
#' @importFrom Matrix colSums
#' @importFrom Matrix t
#' @export
#' 

estimate_score <- function(cds, marker_set1, fData_col="gene_short_name"){
  cds_marker_set1 = cds[fData(cds)[[fData_col]] %in% marker_set1,] 
  aggregate_marker_set1_expression = exprs(cds_marker_set1)
  aggregate_marker_set1_expression = t(t(aggregate_marker_set1_expression) / pData(cds_marker_set1)$Size_Factor)
  aggregate_marker_set1_expression = Matrix::colSums(aggregate_marker_set1_expression)
  aggregate_marker_set1_score = log(aggregate_marker_set1_expression + 1)
}

#' Calculates geneset scores (corrected)
#'
#' @description 
#' This function was implemented by Scott Furlan in the spirit of the text below.
#' 
#' The following text is taken from: Puram, S. V. et al. Single-Cell Transcriptomic Analysis of Primary and 
#' Metastatic Tumor Ecosystems in Head and Neck Cancer. Cell 171, 1611.e1–1611.e24 (2017).
#' 
#' Cell scores (can be calculated) in order to evaluate the degree to which individual cells 
#' express a certain pre-defined expression program. These are initially based on the average 
#' expression of the genes from the pre-defined program in the respective cell: Given an input 
#' set of genes (Gj), we define a score, SCj(i), for each cell i, as the average relative 
#' expression (Er) of the genes in Gj. However, such initial scores may be confounded by 
#' cell complexity, as cells with higher complexity have more genes detected (i.e., less zeros) 
#' and consequently would be expected to have higher cell scores for any gene-set. To control for
#' this effect we also add a control gene-set (Gjcont); we calculate a similar cell score with the 
#' control gene-set and subtract it from the initial cell scores: 
#' 
#' SCj(i) = average[Er(Gj,i)] – average[Er(Gjcont,i)]. 
#' 
#' The control gene-set is selected in a way that ensures similar properties (distribution of expression levels) 
#' to that of the input gene-set to properly control for the effect of complexity. First, all analyzed genes 
#' are binned into 25 bins of equal size based on their aggregate expression levels (Ea). 
#' Next, for each gene in the given gene-set, we randomly select 100 genes from the same 
#' expression bin. In this way, the control gene-set has a comparable distribution of expression 
#' levels to that of the considered gene-set, and is 100-fold larger, such that its average 
#' expression is analogous to averaging over 100 randomly-selected gene-sets of the same size 
#' as the considered gene-set.

#' @param cds Input cell_data_set object.
#' @param marker_set1 Vector of genes in the gene_metadata DataFrame (fData) that can be found in the column 'fData_col'
#' @param fData_col Character string denoting the gene_metadata DataFrame (fData) column that contains the genes in marker_set1.  Default = 'gene_short_name'
#' @return Single cell scores for a give gene set that have been "corrected" using 100X genes with similar
#' expression levels
#' @importFrom Matrix colSums
#' @export
#' 
estimate_corrected_score <- function(cds, marker_set1, fData_col="gene_short_name"){
  set.seed(2019)
  if(!"exprs_bin" %in% colnames(fData(cds))) stop ("No expression binning performed, run m3addon implementation of detect_genes using default settings")
  cds_marker_set1 = cds[fData(cds)[[fData_col]] %in% marker_set1,] 
  aggregate_marker_set1_expression = normalized_counts(cds_marker_set1)
  aggregate_marker_set1_score = colSums(aggregate_marker_set1_expression)
  fdat_notmarker_set1 = data.table::as.data.table(fData(cds)[!fData(cds)[[fData_col]] %in% marker_set1,])
  control_list = split(as.character(fdat_notmarker_set1$gene_short_name), fdat_notmarker_set1$exprs_bin)
  n = as.list(table(fData(cds_marker_set1)$exprs_bin)*100)
  n = n[n > 0]
  control_set1 = unlist(lapply(1:length(n), function(x) sample(control_list[[names(n)[x]]], n[[x]], replace = T)))
  cds_control_set1 = cds[fData(cds)[[fData_col]] %in% control_set1,] 
  aggregate_control_set1_expression = normalized_counts(cds_control_set1)
  aggregate_control_set1_score = colSums(aggregate_control_set1_expression)
  aggregate_marker_set1_score/nrow(cds_marker_set1) - aggregate_control_set1_score/nrow(cds_control_set1)
}

