library(tidyverse)
library(htmltools)
library(viridisLite)
library(viewmastRust)
library(Seurat)
library(ggplot2)

if(grepl("^gizmo", Sys.info()["nodename"])){
  ROOT_DIR1<-"/fh/fast/furlan_s/experiments/MB_10X_5p/cds"
  ROOT_DIR2<-"/fh/fast/furlan_s/grp/data/ddata/BM_data"
} else {
  ROOT_DIR1<-"/Users/sfurlan/Library/CloudStorage/OneDrive-SharedLibraries-FredHutchinsonCancerCenter/Furlan_Lab - General/experiments/MB_10X_5p/cds"
  ROOT_DIR2<-"/Users/sfurlan/Library/CloudStorage/OneDrive-SharedLibraries-FredHutchinsonCancerCenter/Furlan_Lab - General/datasets/Healthy_BM_greenleaf"
}

#query dataset
seu<-readRDS(file.path(ROOT_DIR1, "220302_final_object.RDS"))

seu$seurat_clusters
DimPlot(seu, group.by = "seurat_clusters", cols = as.character(pals::alphabet2()))
DimPlot(seu, group.by = "celltype")
seu$CellType<-factor(seu$seurat_clusters)
labels<-levels(factor(seu$celltype))
labels<-c(labels,  "CD8 Effector")

levels(seu$CellType)<-c(labels[4], #0
                            labels[6], #1
                            labels[4], #2
                            labels[7], #3
                            labels[8], #4
                            labels[12], #5
                            labels[2], #6
                            labels[5], #7
                            labels[9], #8,
                            labels[14], #9,
                            labels[1], #10,
                            labels[15], #11,
                            labels[10], #12,
                            labels[4], #13,
                            labels[4]) #14



### example 1 learning
indices<-1:dim(seu)[2] %in% sample(1:dim(seu)[2], 8000)

seuRef<-CreateSeuratObject(counts = viewmastRust:::get_counts_seurat(seu)[,indices], meta.data = seu@meta.data[indices,])
seuRef@reductions$umap = CreateDimReducObject(embeddings = seu@reductions$umap@cell.embeddings[indices,], assay = "RNA")

seuQuery<-CreateSeuratObject(counts = viewmastRust:::get_counts_seurat(seu)[,!indices], meta.data = seu@meta.data[!indices,])
seuQuery@reductions$umap = CreateDimReducObject(embeddings = seu@reductions$umap@cell.embeddings[!indices,], assay = "RNA")

seuRef$CellType<-as.character(seuRef$CellType)
colors<-as.character(pals::polychrome(12))
names(colors)<-levels(factor(seuRef$CellType))
p<-DimPlot(seuRef, group.by = "CellType", cols=colors)+ 
  theme(axis.text.x=element_blank(), 
        axis.ticks.x=element_blank(), 
        axis.text.y=element_blank(), 
        axis.ticks.y=element_blank())+xlab("UMAP 1")+ylab("UMAP 2")+ggtitle("Ground Truth Celltype on Training Set")
P<-plotly::ggplotly(p)


seuQuery<-viewmastR(query_cds = seuQuery, ref_cds = seuRef, ref_celldata_col = "CellType", selected_genes = VariableFeatures(seu), max_epochs = 20)

seuQuery$CellType<-as.character(seuQuery$CellType)
p2<-DimPlot(seuQuery, group.by = "viewmastRust_smr", cols = colors)+ 
  theme(axis.text.x=element_blank(), 
        axis.ticks.x=element_blank(), 
        axis.text.y=element_blank(), 
        axis.ticks.y=element_blank())+xlab("UMAP 1")+ylab("UMAP 2")+ggtitle("viewmastR Prediction on Validation Set")
P2<-plotly::ggplotly(p2)


confusion_matrix(pred = factor(seuQuery$viewmastRust_smr), gt = factor(seuQuery$CellType), cols = colors)

pred<-factor(seuQuery$viewmastRust_smr)
gt<-factor(seuQuery$CellType)
mat<-table(pred, gt)
labels = union(colnames(mat), rownames(mat))
levels(gt)<-c(levels(gt), levels(pred)[!levels(pred) %in% levels(gt)])
mat_full<-table( pred, gt)
#deal with null colors
if(is.null(cols)){
  cols = sample(colors()[grep('gr(a|e)y', colors(), invert = T)], length(labels))
  names(cols)<-labels
}
# } else {
#   if(length(cols)!=length(labels)) stop("length of color vector provided is incorrect")
# }
mat_full<-mat_full[,match(rownames(mat_full), colnames(mat_full))]
data<-caret::confusionMatrix(mat_full)
pmat<-sweep(mat, MARGIN = 2, colSums(mat), "/")*100

long<-reshape::melt(pmat)
colnames(long) = c("Prediction", "GroundTruth", "Accuracy")
fig <- plotly::plot_ly(data = long, x = ~Prediction, y = ~GroundTruth, z = ~Accuracy, type = "heatmap") 
fig


figALL <- plotly::subplot(P, P2, nrows = 1) %>% 
  plotly::layout(title = list(text = 'Ground Truth Celltype (Left) | viewmastR Prediction (Right)', font = list(size = 18))) %>% plotly::layout(showlegend = FALSE)
figALL

try(dir.create("pkgdown/assets/"))

htmltools::save_html(
  html   = figALL, 
  file   = "pkgdown/assets/index-examples.html"
)

htmltools::save_html(
  html   = fig, 
  file   = "pkgdown/assets/index-examples2.html"
)
