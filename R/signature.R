#this code in process

# #BiocManager::install("karyoploteR")
# #BiocManager::install("ballgown")
# library(karyoploteR)
# library(ballgown)
# # library(BSgenome.Hsapiens.UCSC.hg38)
# # library(TxDb.Hsapiens.UCSC.hg38.knownGene)
# # txdb <- TxDb.Hsapiens.UCSC.hg38.knownGene
# # ls(txdb)
# # transcripts(txdb)
# f<-"/Users/sfurlan/Library/CloudStorage/OneDrive-SharedLibraries-FredHutchinsonCancerCenter/Furlan_Lab - General/resources/refs/cellranger_hg38.gtf"
# gr<-gffReadGR(f)
# gr<-gr[gr$type=="gene",]
# 
# regions <- createRandomRegions(nregions=100, length.mean = 1e3, mask=NA, non.overlapping=FALSE, genome = "hg38")
# 
# 
# gri<-gr[gr$gene_name %in% vg,]
# 
# kp <- plotKaryotype()
# kpPlotCoverage(kp, data=gri)
# 
# training_list$labels
# 
# plot_signature<-function(training_list, granges_obj, celltype, flatten=F, log=F){
#   tl<-training_list$test[sapply(training_list$test, "[[", 2)==celltype]
#   d<-lapply(tl, "[[", 1)
#   m<-do.call(cbind, d)
#   rownames(m)<-training_list$features
#   if(log){
#     coverage<-log((rowSums(m)*10)+1)
#   }else{
#     coverage<-rowSums(m)
#   }
#   coverage<-ceiling(coverage)
#   if(flatten){
#     cv<-names(coverage)[as.numeric(coverage)>0]
#   }else{
#     cv<-rep(names(coverage), as.numeric(coverage))
#   }
#   grcov<-granges_obj[match(cv, granges_obj$gene_name),]
#   kp <- plotKaryotype()
#   kpPlotCoverage(kp, data=grcov, main=celltype)
# }
# 
# plot_sc_signature<-function(training_list, granges_obj, celltype, flatten=F){
#   tl<-training_list$test[sapply(training_list$test, "[[", 2)==celltype]
#   d<-lapply(tl, "[[", 1)
#   m<-d[[sample(1:length(d), 1)]]
#   # m<-do.call(cbind, d)
#   names(m)<-training_list$features
#   coverage<-log((m*10)+1)
#   coverage<-ceiling(coverage)
#   if(flatten){
#     cv<-names(coverage)[as.numeric(coverage)>0]
#   }else{
#     cv<-rep(names(coverage), as.numeric(coverage))
#   }
#   grcov<-granges_obj[match(cv, granges_obj$gene_name),]
#   kp <- plotKaryotype()
#   kpPlotCoverage(kp, data=grcov, main=celltype)
# }
# plot_sc_signature(training_list, granges_obj=gri, celltype=0, flatten = T)
# plot_sc_signature(training_list, granges_obj=gri, celltype=0, flatten = F)
# debug(plot_sc_signature)
# plot_signature(training_list, granges_obj=gri, celltype=0, flatten = T)
# plot_signature(training_list, granges_obj=gri, celltype=0, flatten = F)
# plot_signature(training_list, granges_obj=gri, celltype=6, flatten = T)
# plot_signature(training_list, granges_obj=gri, celltype=6, flatten = F)
# plot_signature(training_list, granges_obj=gri, celltype=16, flatten = T)
# plot_signature(training_list, granges_obj=gri, celltype=20, flatten = F)
# plot_signature(training_list, granges_obj=gri, celltype=14, flatten = T)
# plot_signature(training_list, granges_obj=gri, celltype=20, flatten = T)
# 
# library(rbenchmark)
# benchmark(
#   "cpp" = {
#     computeSparseRowVariancesCPP(mat@i + 1, mat@x, rM, ncol(mat))
#   },
#   "rust" = {
#     computeSparseRowVariances(mat@i + 1, mat@x, rM, ncol(mat))
#   },
#   replications = 10,
#   columns = c("test", "replications", "elapsed",
#               "relative", "user.self", "sys.self"))
# 
# 
# 
# system.time({ computeSparseRowVariancesCPP(mat@i + 1, mat@x, rM, ncol(mat)) })
# system.time({ computeSparseRowVariances(mat@i + 1, mat@x, rM, ncol(mat)) })
# 
# lp<-installed.packages()[rownames(installed.packages())=="viewmastR"][2]
# lp
# list.files(file.path(lp, "viewmastR", "extdata", "mnist"))
# 
#            