
#' @title Scrublet
#' @description See preprint: Scrublet: computational identification of cell doublets in single-cell transcriptomic data
#' Samuel L Wolock, Romain Lopez, Allon M Klein.  bioRxiv 357368; doi: https://doi.org/10.1101/357368
#' @param object the object upon which to perform Scrublet (monocle3 objects and seurat supported)
#' @param split_by the column in the meta data to split the object by before running scrublet
#' @param python_home The python home directory where Scrublet is installed
#' @param cores Number of cores (only helps withwhen splitting and object)
#' @param return_results_only bool (optional, default False)
#' @param min_counts, int (optional, default=2), See scrublet reference 
#' @param min_cells, int (optional, default=3), See scrublet reference  
#' @param expected_doublet_rate, float (optional, default=0.06), See scrublet reference - expected_doublet_rate: the 
#' fraction of transcriptomes that are doublets, typically 0.05-0.1. Results are not particularly sensitive to this parameter. For this example, the expected doublet rate comes from the Chromium User Guide: https://support.10xgenomics.com/permalink/3vzDu3zQjY0o2AqkkkI4CC
#' @param min_gene_variability_pctl, int (optional, default=85), See scrublet reference 
#' @param n_prin_comps, int (optional, default=50), See scrublet reference  (Number of principal components to use)
#' @param sim_doublet_ratio, int (optional, default=2),  the number of doublets to simulate, relative to the number of observed transcriptomes. This should be high enough that all doublet states are well-represented by simulated doublets. Setting it too high is computationally expensive. The default value is 2, though values as low as 0.5 give very similar results for the datasets that have been tested.
#' @param n_neighbors, int (optional) n_neighbors: Number of neighbors used to construct the KNN classifier of observed transcriptomes and simulated doublets. The default value of round(0.5*sqrt(n_cells)) generally works well.
#' Return only a list containing scrublet output
#' @return The input CellDataSet with an additional column added to pData with both the doublet_score output from scrublet, 
#' and 
#' @importFrom reticulate use_python 
#' @importFrom pbmcapply pbmclapply
#' @importFrom reticulate source_python
#' @export
scrublet<-function (object, split_by=NULL, python_home = system("which python", intern = TRUE), 
                    return_results_only = FALSE, min_counts = 2, min_cells = 3, 
                    expected_doublet_rate = 0.06, min_gene_variability_pctl = 85, 
                    n_prin_comps = 50, sim_doublet_ratio = 2, n_neighbors = NULL, seurat_assay="RNA", cores=1) 
{
  
  
  if (!py_available("scrublet")) 
    stop("python module scrublet does not seem to be installed; - try running 'py_config()'")
  reticulate::source_python(paste(system.file(package = "m3addon"), 
                                  "scrublet.py", sep = "/"))
  
  if(class(object)=="Seurat"){
    meta <- object@meta.data
  }else{
    meta <- colData(cds)
  }
  if(is.null(split_by)){
    n=1
    splitvec<-factor(rep("nosplit", nrow(meta)))
  }else{
    n = length(levels(factor(meta[[split_by]])))
    splitvec<-factor(meta[[split_by]])
  }
  
  message("Splitting object into ", n, " smaller objects prior to performing scrublet")
  indices<-vector()
  if(class(object)=="Seurat"){
    dat<-lapply(levels(splitvec), function(split) {
      splitind<-which(splitvec %in% split)
      Xsub <- as(t(object@assays[["RNA"]]@counts[,splitind]), "TsparseMatrix")
      list(ind=splitind, X=Xsub)
    })
    #
  }else{
    dat<-lapply(levels(splitvec), function(split) {
      splitind<-which( splitvec %in% split)
      Xsub <- as(t(exprs(cds)[,splitind]), "TsparseMatrix")
      list(ind=splitind, X=Xsub)
    })
  }
  
  fdata<-pbmclapply(dat, FUN = function(data){
    X<-data$X
    i <- as.integer(X@i)
    j <- as.integer(X@j)
    val <- X@x
    dim <- as.integer(X@Dim)
    if (is.null(n_neighbors)) {
      n_neighbors <- round(0.5 * sqrt(nrow(X)))
    }
    scrublet_py_args <- c(list(i = i, j = j, val = val, dim = dim, 
                               expected_doublet_rate = expected_doublet_rate, min_counts = min_counts, 
                               min_cells = min_cells, min_gene_variability_pctl = min_gene_variability_pctl, 
                               n_prin_comps = n_prin_comps, sim_doublet_ratio = sim_doublet_ratio, 
                               n_neighbors = n_neighbors))
    scrublet_res <- do.call(scrublet_py, scrublet_py_args)
    names(scrublet_res) <- c("doublet_scores", "predicted_doublets")
    list(res=scrublet_res, data$ind)
  }, mc.cores=cores)
  res<-lapply(fdata, "[[", 1)
  ind<-unlist(sapply(fdata, "[[", 2))
  
  ds<-unlist(sapply(lapply(fdata, "[[", 1), "[[", "doublet_scores"))[order(ind)]
  pd<-unlist(sapply(lapply(fdata, "[[", 1), "[[", "predicted_doublets"))[order(ind)]
  
  final_res<-data.frame(doublet_scores=ds, predicted_doublets=pd)
  
  if (return_results_only) {
    return(final_res)
  }
  else {
    if(class(object)=="Seurat"){
      object@meta.data[["doublet_scores"]] <- final_res$doublet_scores
      object@meta.data[["predicted_doublets"]] <- final_res$predicted_doublets
      object
    }else{
      pData(object)[["doublet_scores"]] <- final_res$doublet_scores
      pData(object)[["predicted_doublets"]] <- final_res$predicted_doublets
      object
    }
  }
}
