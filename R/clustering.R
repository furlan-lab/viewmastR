#' Finds partitions akin to Monocle3 FindPartition Function but for a Seurat object
#'
#' @description Stated differently, low cluster resolution
#' @param obj Input seurat object.
#' @param method Louvain or Leiden
#' @param k nearest number k
#' @param reduction reduction to find partitions of
#' @param dims number of dimensions
#' @param num_iter Number of iterations (leiden only); Default is 1.
#' @param random_seed seed default 2020
#' @param resolution_parameter leiden only
#' @param verbose Default True
#' @param partition_q_value Louvain only - sets resolution
#' @export
#' 

find_partitions<-function(obj, method = "louvain", k = 20, reduction="umap", dims = NULL, weight = F,  num_iter = 1, resolution_parameter = NULL, random_seed = 2020, 
                         verbose = T, partition_q_value = 0.05){
  {
    if(class(obj)!="Seurat"){stop("This does not appear to be a seurat object")}
    if(is.null(obj@reductions[[reduction]]@cell.embeddings)){stop("No reduction found")}
    reduction<-obj@reductions[[reduction]]@cell.embeddings
    if (is.null(dims)) {
      dims <- 1:dim(reduction)[2]
    }
    if (method == "leiden") {
      cluster_result <- monocle3:::leiden_clustering(data = as.matrix(reduction)[, 
                                                                                 dims], pd = obj@meta.data, k = k, weight = weight, num_iter = num_iter, 
                                                     resolution_parameter = resolution_parameter, random_seed = random_seed, 
                                                     verbose = verbose, nn_control = list(method = "nn2"))
    }
    else {
      cluster_result <- monocle3:::louvain_clustering(data = as.matrix(reduction)[, dims], pd = obj@meta.data, k = k, weight = weight, random_seed = random_seed, verbose = verbose, nn_control = list(method ="nn2"))
    }
    cluster_graph_res <- monocle3:::compute_partitions(cluster_result$g, 
                                                       cluster_result$optim_res, partition_q_value, verbose = t)
    ps<-as.factor(igraph::components(cluster_graph_res$cluster_g)$membership[cluster_result$optim_res$membership])
    message(paste0("Found ", length(levels(ps)), " partitions!"))
    obj@meta.data$partitions<-ps
    obj
    # clusters <- factor(igraph::membership(cluster_result$optim_res))
    # cds@clusters[["UMAP"]] <- list(cluster_result = cluster_result, 
    #     partitions = partitions, clusters = clusters)
    # cds
  }
}

