
##cluster
rm(list=ls())
sFH2
ml ArrayFire/3.8.1-foss-2019b-CUDA-10.2.89
ml R/4.0.0-foss-2019b-fh1
export AF_PATH=/app/software/ArrayFire/3.8.1-foss-2019b-CUDA-10.2.89
R
if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
ps<-.libPaths()
.libPaths(ps[c(2,1,3)])
devtools::install_github("daqana/rcpparrayfire")
devtools::install_github("furlan-lab/viewmastR")
install.packages("keras")



sFH2
sinfo
srun --pty -c 22 --mem=240G -p campus-new -t "7-0" --gres=gpu --nodelist=gizmok138 /bin/bash -il
ml ArrayFire/3.8.1-foss-2019b-CUDA-10.2.89
ml R/4.0.0-foss-2019b-fh1
R
library(viewmastR)
library("Seurat")
library(monocle3)
#devtools::install_github('cole-trapnell-lab/monocle3')
library(ggplot2)
setwd("~/Analysis/viewmastR")

seu<-readRDS("query.RDS")
DimPlot(seu)
refm<-seurat_to_monocle3(seu)
# refm<-monocle3::preprocess_cds(refm)
# refm<-reduce_dimension(refm)
# plot_cells(refm, color_cells_by = "seurat_clusters")
# plot_cells(refm, color_cells_by = "celltype")

undebug(iterative_LSI)
undebug(monocle3:::leiden_clustering)
cluster_cells
refm<-viewmastR::iterative_LSI(refm)
refm<-reduce_dimension(refm, preprocess_method = "LSI")
plot_cells(refm, color_cells_by = "celltype")
ref<-readRDS("refCDS.RDS")
  vg<-common_variant_genes(query, ref, top_n = 5000)

  plot_cells(ref, color_cells_by="BioClassification")
  query<-viewmastR(query, ref, 
    ref_celldata_col = "BioClassification", 
    selected_genes = vg, 
    verbose=T, learning_rate=0.5,
    FUNC = "neural_network", 
    max_epochs= 500,
    hidden_layers = c(500),
    tf_idf = F, device=1)

bmcols=sfc(16)[c(1:3, 5:16)]
  df<-data.frame(old=levels(factor(ref$BioClassification)), new=c("01_HSC", "02_Early_Erythroid", "03_Late_Erythroid", "04_Myeloid_Progenitor", "04_Myeloid_Progenitor", "05_Lymphoid_Progenitor", "04_Myeloid_Progenitor", "04_Myeloid_Progenitor", "06_pDC", "07_cDC", "08_CD14_Monocyte", "08_CD14_Monocyte", "09_CD16_Monocyte", "10_Other", "05_Lymphoid_Progenitor", "11_Pre_B", "12_B", "13_Plasma", "14_T", "14_T","14_T","14_T","14_T","14_T","15_NK", "10_Other"))
  query$celltype<-factor(query$nn_celltype)
  levels(query$celltype)<-df$new[match(levels(query$celltype), df$old)]
  seu$celltype_nn<-factor(as.character(query$celltype))
Idents(seu)<-seu$celltype_nn
DimPlot(seu, group.by = "celltype_nn", label = F, pt.size = 0.4)+scale_color_manual(values=bmcols)
  query<-viewmastR(query, ref, 
    ref_celldata_col = "BioClassification", 
    selected_genes = vg, 
    verbose=T, 
    FUNC = "softmax_regression",
    tf_idf = F)
  query$celltype<-factor(query$smr_celltype)
  levels(query$celltype)<-df$new[match(levels(query$celltype), df$old)]
  seu$celltype_smr<-factor(as.character(query$celltype))
DimPlot(seu, group.by = "celltype_smr", label = F, pt.size = 0.4)+scale_color_manual(values=bmcols)



roxygen2::roxygenize(".")

#test array fire sparse functions
data<-keras::dataset_mnist()
dim(data$train$x)<-c(60000, 28*28)
data$train$x<-data$train$x/255
data$train$y<-model.matrix(~0+factor(data$train$y))
colnames(data$train$y)<-0:9
dim(data$test$x)<-c(10000, 28*28)
data$test$y<-data$test$y/255
data$test$y<-model.matrix(~0+factor(data$test$y))
colnames(data$test$y)<-0:9
out<-smr(t(data$train$x)[,1:5000], 
           t(data$test$x)[,1:1000], 
           data$train$y[1:5000,], 
           data$test$y[1:1000,],
           lambda = 2.0,
           max_error = 0.01,
           learning_rate = 1, 
           num_classes = 10, 
           query = t(data$test$x),
           verbose = T)
# 
# out<-smr_sparse(t(data$train$x)[,1:5000],
#          t(data$test$x)[,1:1000],
#          data$train$y[1:5000,],
#          data$test$y[1:1000,],
#          lambda = 1.0,
#          max_error = 0.01,
#          learning_rate = 0.1,
#          num_classes = 10,
#          query = t(data$test$x),
#          verbose = T)
# 
# smr_sparse_test(t(data$train$x)[,1:5000],
#                 t(data$test$x)[,1:1000],
#                 data$train$y[1:5000,],
#                 data$test$y[1:1000,],
#                 lambda = 1.0,
#                 max_error = 0.01,
#                 learning_rate = 0.1,
#                 num_classes = 10,
#                 query = t(data$test$x),
#                 verbose = T)



#mnist arrayfire working well.
library(viewmastR)
data<-keras::dataset_mnist()
dim(data$train$x)<-c(60000, 28*28)
data$train$x<-data$train$x/255
data$train$y<-model.matrix(~0+factor(data$train$y))
colnames(data$train$y)<-0:9
dim(data$test$x)<-c(10000, 28*28)
data$test$y<-data$test$y/255
data$test$y<-model.matrix(~0+factor(data$test$y))
colnames(data$test$y)<-0:9
out<-af_nn(t(data$train$x)[,1:5000], 
           t(data$test$x)[,1:1000], 
           data$train$y[1:5000,], 
           data$test$y[1:1000,],
           max_error = 0.5,
           learning_rate = 1.5, 
           num_classes = 10, 
           query = t(data$test$x),
           layers = c(784, 100, 50, 10),  
           verbose = T)


#build relu function
sigmoid<-function(x){
  return(1/(1+exp(-x)))
  }
#test sigmoid
data<-keras::dataset_mnist()
dim(data$train$x)<-c(60000, 28*28)
d<-data$train$x[1,]
all.equal(get_sigmoid(d),sigmoid(d))

relu<-function(input){
  input[input<0]<-0
  input
}
r<-rnorm(784)
all.equal(relu(r), get_relu(r))
#test relu
library(viewmastR)
data<-keras::dataset_mnist()
dim(data$train$x)<-c(60000, 28*28)
data$train$x<-data$train$x/255
data$train$y<-model.matrix(~0+factor(data$train$y))
colnames(data$train$y)<-0:9
dim(data$test$x)<-c(10000, 28*28)
data$test$y<-data$test$y/255
data$test$y<-model.matrix(~0+factor(data$test$y))
colnames(data$test$y)<-0:9
out<-af_nn(t(data$train$x)[,1:3000], 
           t(data$test$x)[,1:300], 
           data$train$y[1:3000,], 
           data$test$y[1:300,],
           relu_activation = T,
           max_error = 0.5,
           learning_rate =1, 
           num_classes = 10, 
           query = t(data$test$x),
           layers = c(784, 100, 50, 10),  
           verbose = T)




ann_demo()

mnl<-viewmastR::get_mnist()
mnl[[3]]
dim(mnl[[1]])

mnl[[1]][1:784,1]

## arrayfire testing
library(viewmastR)
data(iris)
data<-as.matrix(iris[,1:4])
colnames(data)<-NULL
labels<-iris$Species
labn<-as.numeric(labels)-1
labf<-as.factor(labels)
labels<-levels(labels)

train_frac<-0.8
train_idx<-sample(1:dim(data)[1], round(train_frac*dim(data)[1]))
test_idx<-which(!1:dim(data)[1] %in% train_idx)

train_feats = t(data[train_idx,])
test_feats = t(data[test_idx,])
train_labels = labn[train_idx]
test_labels = labn[test_idx]
num_classes = length(labels)
laboh<-matrix(model.matrix(~0+labf), ncol = length(labels))
colnames(laboh)<-NULL
train_targets= laboh[train_idx,]
test_targets= laboh[test_idx,]

dim(train_feats) 
dim(test_feats)
length(train_labels)
length(test_labels)
num_classes
dim(train_targets)
dim(test_targets)

out<-af_nn(train_feats, test_feats, train_targets, test_targets, num_classes, layers = c(4,4,3), query = test_feats, learning_rate =0.01, verbose = T)



af_nn(train_feats, test_feats, train_targets, test_targets, num_classes, query = test_feats, verbose = T)
naive_bayes(train_feats, test_feats, train_labels, test_labels, num_classes, query = test_feats, verbose = T)
bagging(train_feats, test_feats, train_labels, test_labels, num_classes, query = test_feats, verbose = T)
smr(train_feats, test_feats, train_targets, test_targets, num_classes, query = test_feats, verbose = T)
af_dbn(train_feats, test_feats, train_targets, test_targets, num_classes, query = test_feats, verbose = T)
lr(train_feats, test_feats, train_targets, test_targets, num_classes, query = test_feats, verbose = T)
perceptron(train_feats, test_feats, train_targets, test_targets, num_classes, query = test_feats, verbose = T)


test_backends()
naive_bayes_demo()
ann_demo()
bagging_demo()
smr_demo()
dbn_demo()
lr_demo()
perceptron_demo()


usethis::use_build_ignore("debug.R")
usethis::use_build_ignore("debug.Rmd")

cds<-readRDS("data/m3cds.RDS")



sm<-as.matrix(normalized_counts(cds[1:1000,sample(ncol(cds), 1000)]))
dim(sm)
x <- matrix(rnorm(100), nrow = 20)
coords<-spRing(x, method="euclidean")

bs = backspin(sm)

library(RcppCNPy)
npySave("data/sm.npy", sm)
getwd()

#in python
import numpy as np
sm = np.load('/Users/sfurla/Box Sync/PI_FurlanS/computation/Rproj/m3addon/data/sm.npy')
backSPIN(sm)
#seems to work


# system.time({
# rs<-m3addon:::rowStdDev(exprs(cds))
# })
# 
# system.time({
# cs<-m3addon:::colStdDev(t(exprs(cds)))
# })
# all(rs[,1]==cs[1,])

system.time({
  rt<-m3addon:::rowStdDev(exprs(cds))
})

system.time({
  rr<-rowSds(as.matrix(exprs(cds)))
})
# all(as.numeric(rt[1,])[1:20]==rr[1:20])


Rcpp::sourceCpp("src/scores.cpp")

cds<-calculate_gene_dispersion(cds, q=5)
plot_gene_dispersion(cds)
cds<-select_genes(cds)
plot_gene_dispersion(cds)

ord_genes<-get_ordering_genes(cds)
cds<-preprocess_cds(cds, use_genes = ord_genes, verbose = T, num_dim = 100)
plot_pc_variance_explained(cds)
cds<-reduce_dimension(cds, reduction_method = "UMAP", num_dim = 35, verbose=T, cores = detectCores())
plot_cells(cds, color_cells_by = "Group")


levels(factor(pData(cds)$Group))
cdsDE<-cds[,pData(cds)$Group %in% c("Colon_Donor", "Colon_Host")]
levels(factor(pData(cdsDE)$Group))

gene_fits <-fit_models(cdsDE[1:10,], model_formula_str = "~Group", verbose = TRUE, cores = detectCores())


###########AF#############
s <- matrix(seq(0, 100, by = .0001), ncol = 1)
rbenchmark::benchmark(Arma = put_option_pricer_arma(s, 60, .01, .02, 1, .05),
                      AF = put_option_pricer_af(s, 60, .01, .02, 1, .05),
                      order = "relative", 
                      replications = 100)[,1:4]

Rcpp::sourceCpp("src/armatut.cpp")
Rcpp::sourceCpp("src/aftut.cpp", rebuild = T)

######

cds <- readRDS(file.path("/Users/sfurla/Box Sync/PI_FurlanS/computation", "Analysis", "NHPTreg_mm", "cds", "4thRound", "190820_m3_CDS.RDS"))
cds<-estimate_size_factors(cds)
cds<-detect_genes(cds)
cds<-calculate_gene_dispersion(cds, method = "m3addon")
plot_gene_dispersion(cds)
cds<-select_genes(cds, top_n = 2000)
plot_gene_dispersion(cds)
get_ordering_genes(cds)

plot_cells(cds, color_cells_by = "Clust2", reduction_method = "tSNE", label_cell_groups = F)
plot_heatmap(cds, c("FOXP3", "IL2RA"), group_by = "Clust2")
debug(plot_heatmap)

monocle3::plot_percent_cells_positive(cds[fData(cds)$gene_short_name %in% c("FOXP3", "IL2RA"),], group_cells_by = "Clust2")

debug(doubletFinder_v3)

t<-doubletFinder_v3(cds, PCs=1:25, genes = "recalc")

if (!requireNamespace("drat", quietly = TRUE)) install.packages("drat")
drat::addRepo("daqana")
install.packages("RcppArrayFire")

RcppArrayFire.package.skeleton("af")



