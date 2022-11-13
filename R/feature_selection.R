#' Find Cluster specific genes quickly
#' @param cds cell_data_set object with a reduced dimension matrix (currently LSI supported) as specified in
#' reduced_dim argument and a model used to create a low dimensional embedding
#' @param column a SummarizedExperiment type object (cell_data_set currently supported) to be projected using 
#' the models contained in the projector
#' @param verbose A boolean value indicating whether to use verbose output during execution of this function. Can be set to FALSE for a cleaner output.
#' @param threads The number of threads used for parallel execution
#' @import monocle3
#' @importFrom magrittr "%>%"
#' @export
cluster_specific <- function(cds,
                             column = "cluster",
                             verbose = T, 
                             threads = 1, 
                             run_glm=F, 
                             min_expression = 0.2, 
                             top_n=500,
                             symbol_id = "gene_id"){
  if (column=="cluster"){
    cluster<-clusters(cds)
  }else{
    cluster<-factor(colData(cds)[[column]])
  }
  clist<-as.list(levels(cluster))
  tm<-top_markers(cds, genes_to_test_per_group = dim(cds)[1], marker_sig_test = F,  group_cells_by = column)
  toplist<-lapply(clist, function(group){
    tm %>% dplyr::arrange(-specificity) %>% dplyr::filter(cell_group %in% group) %>% dplyr::filter(mean_expression>min_expression)
  })
  toplist<-unique(unlist(lapply(toplist, function(df) df[1:top_n,colnames(df) %in% symbol_id][[symbol_id]])))
}
