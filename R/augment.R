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
  #type<-to_synthesize[2]
  tsl<-lapply(1:length(to_synthesize), function(n) setNames(as.numeric(to_synthesize[n]), names(to_synthesize[n])))
  message("Simulating cells")
  dl<-pbmclapply(tsl, function(type){
    N<-as.numeric(type)
    type<-names(type)
    #namevec<-c(namevec, rep(type, N))
    rsums<-rowSums(spmat[,splitparam %in% type])
    sizes<-colSums(spmat[,splitparam %in% type])
    # ggplot(data.frame(x=sizes), aes(x=x))+geom_histogram(aes(y = ..density..),
    #                colour = 1, fill = "white")+geom_density()
    den<-density(sizes)
    newsizes <- sample(sizes, N, replace=TRUE) + rnorm(N*10, 0, den$bw)
    # ggplot(data.frame(x=newsizes), aes(x=x))+geom_histogram(aes(y = ..density..),
    #                colour = 1, fill = "white")+geom_density()
    trimmed_newdata<-round(newsizes[newsizes>min(sizes) & max(sizes)], 0)
    final_newsizes<-sample(trimmed_newdata, N)
    # ggplot(data.frame(x=final_newsizes), aes(x=x))+geom_histogram(aes(y = ..density..),
    #                colour = 1, fill = "white")+geom_density()
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

# 
# 
# 
# #get densities for a given scrna experiment
# if(grepl("^gizmo", strsplit(Sys.info()[4], "\\.")[[1]])){
#   f<-"/fh/fast/furlan_s/grp/data/ddata/BM_data/"
# }else{
#   f<-"~/Library/CloudStorage/OneDrive-SharedLibraries-FredHutchinsonCancerCenter/Furlan_Lab - General/datasets/Healthy_BM_greenleaf"
# }
# hm<-readRDS(file.path(f, "230329_rnaAugmented_seurat.RDS"))
# sizes_sc <- colSums(hm@assays$RNA@counts)
# den_sc <- density(sizes_sc)
# 
# table(seu$subtype)
# 
# 
# newdata<-pbmcapply::pbmclapply(1:(dim(obj)[2]), function(n){
#   newsizes <- sample(sizes_sc, N, replace = TRUE) + rnorm(N * 10, 0, den_sc$bw)
#   trimmed_newdata <- round(newsizes[newsizes > min(sizes_sc) & 
#                                       max(sizes_sc)], 0)
#   final_newsizes <- sample(trimmed_newdata, N)
#   # newdata<-pbmcapply::pbmclapply(1:3, function(n){
#   rsums <- seu@assays$RNA$counts[,n]
#   if(is.null(names(rsums))){
#     names(rsums)<-rownames(seu@assays$RNA$counts)
#   }
#   splat <- names(rsums)[rep(seq_along(rsums), rsums)]
#   dl <- lapply(final_newsizes, function(i) {
#     tab <- table(sample(splat, i))
#     nf <- universe[!universe %in% names(tab)]
#     all <- c(tab, setNames(rep(0, length(nf)), nf))
#     all[match(universe, names(all))]
#   })
#   return(as.sparse(do.call(cbind, dl)))
# }, mc.cores = parallel::detectCores())
# 
# output<-lapply(newdata, function(x) class(x)[1])
# good_ind<-lapply(output, function(x) x=="dgCMatrix")
# good_ind<-which(unlist(good_ind))
# 
# 
# metai<-lapply(good_ind, function(n) rep(n, N)) %>% do.call("c", .)
# newmeta<-meta[metai,]
# gooddata<-newdata[good_ind]
# tmpdata<-Reduce(cbind, gooddata[-1], gooddata[[1]])
# dim(tmpdata)
# dim(newmeta)
# rownames(newmeta)<-make.unique(as.character(metai))
# colnames(tmpdata)<-rownames(newmeta)
# 
# seuF<-CreateSeuratObject(tmpdata, meta.data = as.data.frame(newmeta))
# rm(den_sc, ds, fig, gooddata, mat, meta, newdata, newmeta, obj, output, plot.data, tmpdata)
# gc()
# 
# seuF<-merge(seuF, hm)
# rm(hm)
# gc()
# seuF <- FindVariableFeatures(seuF, selection.method = "vst", nfeatures = 2000)
# seuF <- ScaleData(seuF) %>% RunPCA(features = VariableFeatures(object = seuF), npcs = 50)
# ElbowPlot(seuF, 50) 
# seuF<- FindNeighbors(seuF, dims = 1:40)
# seuF <- FindClusters(seuF, resolution = 0.5)
# seuF <- RunUMAP(seuF, dims = 1:40)
# 
# seuF$celltype<-seuF$category1
# seuF$celltype[match(colnames(hm), colnames(seuF))]<-hm$SFClassification
# cvg<-common_variant_genes(seuF, sc, top_n = 10000)
# 
# seuF$celltype[is.na(seuF$celltype)]<-"NotFound"
# 
# DimPlot(seuF, group.by = "celltype")
# DimPlot(seuF, group.by = "category0")
# DimPlot(seuF, group.by = "category1")+NoLegend()
# DimPlot(seuF, group.by = "category2")
# 
# sc<-viewmastR2::viewmastR(sc, seuF, ref_celldata_col = "celltype", query_celldata_col = "learned_label", selected_genes = cvg, verbose = T, FUNC = "softmax_regression", sparse=T) 

