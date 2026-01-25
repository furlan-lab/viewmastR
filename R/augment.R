################################################################################
# FILE: R/augment.R
# STATUS: Clean
# ------------------------------------------------------------------------------
# Functions:
# [x] augment_data              (Exported)
################################################################################


#' Augment data
#' @description This function takes a seurat object and finds cells that are not sufficiently abundant when grouped by the
#' column parameter, then simulates data to augment cell number to a level of the parameter - norm_number
#' @param column column from the metadata that designates cell group (i.e. celltype)
#' @param norm_number cell number to augment data to for cells that are not sufficiently abundant in the object
#' @param prune downsample cells present at number higher than norm_number to the level of norm_number (default = F)
#' @return a seurat object augmented with simulated cells such that all cell groups are present at a level of norm_number of cells
#' @importFrom pbmcapply pbmclapply
#' @importFrom parallel detectCores
#' @export
#' 

augment_data<-function(obj, column, norm_number=2000, assay="RNA", prune = F){
  message("Getting data for selected assay")
  obj<-CreateSeuratObject(counts = GetAssayData(obj, assay = assay), meta.data = obj@meta.data)
  splitparam<-as.character(obj[[column]][,1])
  balance<-table(splitparam) - setNames(rep(norm_number, length(levels(factor(splitparam)))), levels(factor(splitparam)))
  if (prune){
    message("Pruning over abundant celltypes")
    to_downsample<-balance[balance>0]
    tdl<-lapply(1:length(to_downsample), function(n) setNames(as.numeric(to_downsample[n]), names(to_downsample[n])))
    ds_indices_list<-lapply(tdl, function(cellt) {
      found<-which(obj[[column]][,1] %in% names(cellt))
      sample(found, (length(found)-norm_number))
    })
    ds_indices<-unlist(ds_indices_list)[order(unlist(ds_indices_list))]
    obj<-obj[,-ds_indices]
    splitparam<-as.character(obj[[column]][,1])
  }
  message("Extracting less abundant celltypes")
  to_synthesize<-balance[balance<0]*-1
  spmat<-get_counts_seurat(obj)
  universe<-rownames(spmat)
  tsl<-lapply(1:length(to_synthesize), function(n) setNames(as.numeric(to_synthesize[n]), names(to_synthesize[n])))
  message("Simulating cells")
  dl<-pbmclapply(tsl, function(type){
    N<-as.numeric(type)
    type<-names(type)
    rsums<-rowSums(spmat[,splitparam %in% type])
    sizes<-colSums(spmat[,splitparam %in% type])
    den<-density(sizes)
    newsizes <- sample(sizes, N, replace=TRUE) + rnorm(N*10, 0, den$bw)
    trimmed_newdata<-round(newsizes[newsizes>min(sizes) & max(sizes)], 0)
    final_newsizes<-sample(trimmed_newdata, N)
    splat <- names(rsums)[rep(seq_along(rsums), rsums)]
    dl<-lapply(final_newsizes, function(i){
      tab<-table(sample(splat, newsizes[1]))
      nf<-universe[!universe %in% names(tab)]
      all<-c(tab, setNames(rep(0, length(nf)), nf))
      all[match(universe, names(all))]
    })
    as.sparse(do.call(cbind, dl))
  }, mc.cores = min(c(detectCores(), length(to_synthesize))))
  message("Merging simulated cells")
  sm<-do.call(cbind, dl)
  nvl<-lapply(tsl, function(type){
    N<-as.numeric(type)
    type<-names(type)
    rep(type, N)})
  colnames(sm)<-make.unique(paste0("stimcell_", do.call(c, nvl)))
  seuS<-CreateSeuratObject(counts=sm)
  seuS[[column]]<-do.call(c, nvl)
  stimulated_column_name<-do.call(c, nvl)
  message("Returning merged objects")
  meta<-dplyr::bind_rows(obj@meta.data, seuS@meta.data)
  rownames(meta)<-make.unique(rownames(meta))
  new_counts<-cbind(get_counts_seurat(obj), get_counts_seurat(seuS))
  colnames(new_counts)<-rownames(meta)
  CreateSeuratObject(counts = new_counts, meta.data = meta)
}
