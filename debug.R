roxygen2::roxygenize(".")

dyn.load('/app/software/ArrayFire/3.8.1/lib64/libaf.so.3')
library(RcppArrayFire)
library(viewmastR)
test_backends()
devtools::install_github("furlan-lab/viewmastR", ref="develop", force=T)

library(Seurat)
library(monocle3)


seu<-readRDS("/fh/fast/furlan_s/user/mmendoza/cds_objects_from_OneDrive/2201107_LV1_cds.RDS")
DimPlot(seu)
rna<-readRDS("/fh/fast/furlan_s/grp/data/ddata/BM_data/230329_rnaAugmented_seurat.RDS")

vg<-common_variant_genes(rna, seu, top_n = 3000)
seu<-viewmastR(seu, rna, ref_celldata_col = "SFClassification", selected_genes = vg, FUNC = "softmax_regression")

DimPlot(seu, group.by="smr_celltype")
##install arrayfire
#download arrayfire
sFH2
sinfo
srun --pty -c 22 --mem=240G -p campus-new -t "7-0" --gres=gpu --nodelist=gizmok136 /bin/bash -il
ml CUDA/11.3.1
ml CMake/3.22.1-GCCcore-11.2.0
cd /home/sfurlan/software/arrayfire/share/ArrayFire
cp -r examples ~/arrayfire_examples
cd ~/arrayfire_examples
mkdir build
cd build
cmake -DArrayFire_DIR=$HOME/software/arrayfire/share/ArrayFire/cmake ..
make
cd machine_learning
./naive_bayes_cuda
./naive_bayes_cpu
./machine_learning/naive_bayes_opencl

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
devtools::install_github("furlan-lab/viewmastR", branch="keras")
install.packages("keras")



sFH2
sinfo
srun --pty -c 22 --mem=240G -p campus-new -t "7-0" --gres=gpu --nodelist=gizmok135 /bin/bash -il
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

rm(list=ls())
library(viewmastR)
library(Seurat)
library(monocle3)
library(tidyr)
library(keras)
library(reticulate)
library(ggplot2)
py_config()
seu<-readRDS("/Users/sfurlan/Library/CloudStorage/OneDrive-SharedLibraries-FredHutchinsonCancerResearchCenter/Furlan_Lab - General/experiments/MB_10X_5p/cds/220302_final_object.RDS")
rna<-readRDS("/Users/sfurlan/Library/CloudStorage/OneDrive-SharedLibraries-FredHutchinsonCancerResearchCenter/Furlan_Lab - General/datasets/Healthy_BM_greenleaf/230329_rnaAugmented.RDS")

seur<-monocle3_to_seurat(rna, normalize=F)
vg<-common_variant_genes(seu, seur, plot=F, top_n = 2000)

seur<-seur[,sample( 1:dim(seur)[2], 5000)]
DimPlot(seur, group.by = "SFClassification", cols = rna@metadata$colorMap$classification)

seu<-viewmastR(seu, seur, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "lasso", verbose=T, cores=8)
DimPlot(seu, group.by = "viewmastR", cols=rna@metadata$colorMap$classification)

seu<-viewmastR(seu, seur, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "xgboost", verbose=T, cores=8)
DimPlot(seu, group.by = "viewmastR", cols=rna@metadata$colorMap$classification)

seu<-viewmastR(seu, seur, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "softmax_regression", verbose=T, cores=8)
DimPlot(seu, group.by = "viewmastR", cols=rna@metadata$colorMap$classification)

seu<-viewmastR(seu, seur, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "naive_bayes", verbose=T, cores=8)
DimPlot(seu, group.by = "viewmastR", cols=rna@metadata$colorMap$classification)


model <- keras_model_sequential() %>%
  layer_dense(units = 1000, activation = 'relu', input_shape = 1651) %>%
  layer_dense(units = 500, activation = 'relu') %>%
  layer_dense(units = 200, activation = 'relu') %>%
  layer_dense(units = length(levels(factor(rna[["SFClassification"]]))), activation = 'softmax')

model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

seu<-viewmastR(seu, seur, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "keras", verbose=T, keras_model = model)
DimPlot(seu, group.by = "viewmastR", cols=rna@metadata$colorMap$classification)

seu<-viewmastR(seu, seur, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "neural_network", verbose=T, learning_rate = 0.05)
DimPlot(seu, group.by = "viewmastR", cols=rna@metadata$colorMap$classification)

seu<-viewmastR(seu, seur, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "logistic_regression", verbose=T, learning_rate = 0.05)
DimPlot(seu, group.by = "viewmastR", cols=rna@metadata$colorMap$classification)

seu<-viewmastR(seu, seur, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "bagging", verbose=T, learning_rate = 0.05)
DimPlot(seu, group.by = "viewmastR", cols=rna@metadata$colorMap$classification)

seu<-viewmastR(seu, seur, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "deep_belief_nn", verbose=T)
DimPlot(seu, group.by = "viewmastR", cols=rna@metadata$colorMap$classification)

seu<-viewmastR(seu, seur, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "perceptron", verbose=T)
DimPlot(seu, group.by = "viewmastR", cols=rna@metadata$colorMap$classification)




cds<-seurat_to_monocle3(seu)
cdsr<-seurat_to_monocle3(seur)
cds<-viewmastR(cds, cdsr, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "lasso", verbose=T, cores=8)
plot_cells(cds, color_cells_by  = "viewmastR")+scale_color_manual(values=rna@metadata$colorMap$classification)

cds<-viewmastR(cds, cdsr, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "xgboost", verbose=T, cores=8)
plot_cells(cds, color_cells_by  = "viewmastR")+scale_color_manual(values=rna@metadata$colorMap$classification)

cds<-viewmastR(cds, cdsr, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "softmax_regression", verbose=T, cores=8)
plot_cells(cds, color_cells_by  = "viewmastR")+scale_color_manual(values=rna@metadata$colorMap$classification)

cds<-viewmastR(cds, cdsr, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "naive_bayes", verbose=T, cores=8)
plot_cells(cds, color_cells_by  = "viewmastR")+scale_color_manual(values=rna@metadata$colorMap$classification)

model <- keras_model_sequential() %>%
  layer_dense(units = 1000, activation = 'relu', input_shape = 1651) %>%
  layer_dense(units = 500, activation = 'relu') %>%
  layer_dense(units = 200, activation = 'relu') %>%
  layer_dense(units = length(levels(factor(rna[["SFClassification"]]))), activation = 'softmax')

model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

cds<-viewmastR(cds, cdsr, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "keras", verbose=T, keras_model = model)
plot_cells(cds, color_cells_by  = "viewmastR")+scale_color_manual(values=rna@metadata$colorMap$classification)

cds<-viewmastR(cds, cdsr, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "neural_network", verbose=T, learning_rate = 0.05)
plot_cells(cds, color_cells_by  = "viewmastR")+scale_color_manual(values=rna@metadata$colorMap$classification)

cds<-viewmastR(cds, cdsr, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "logistic_regression", verbose=T, learning_rate = 0.05)
plot_cells(cds, color_cells_by  = "viewmastR")+scale_color_manual(values=rna@metadata$colorMap$classification)

cds<-viewmastR(cds, cdsr, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "bagging", verbose=T, learning_rate = 0.05)
plot_cells(cds, color_cells_by  = "viewmastR")+scale_color_manual(values=rna@metadata$colorMap$classification)

cds<-viewmastR(cds, cdsr, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "deep_belief_nn", verbose=T)
plot_cells(cds, color_cells_by  = "viewmastR")+scale_color_manual(values=rna@metadata$colorMap$classification)

cds<-viewmastR(cds, cdsr, ref_celldata_col = "SFClassification", 
               query_celldata_col = "viewmastR", selected_genes = vg,
               FUNC = "perceptron", verbose=T)
plot_cells(cds, color_cells_by  = "viewmastR")+scale_color_manual(values=rna@metadata$colorMap$classification)



undebug(viewmastR)

DimPlot(seu, group.by = "celltype")
DimPlot(seur, group.by = "SFClassification", cols = rna@metadata$colorMap$classification)

seut<-calculate_gene_dispersion(seu)
seut@misc$dispersion
plot_gene_dispersion(seut)
seut<-select_genes(seut, top_n = 3000)

vg<-common_variant_genes(seu, seur, plot=F)

seu<-viewmastR(seu, seur, ref_celldata_col = "SFClassification", query_celldata_col = "viewmastR", selected_genes = vg)
DimPlot(seu, group.by = "viewmastR", cols=rna@metadata$colorMap$classification)

### glmNET USING SPARSE - parallel
require(doMC)
registerDoMC(cores = detectCores())
cds<-seurat_to_monocle3(seu)
rna<-seurat_to_monocle3(seuM)
common_list<-viewmastR::common_features(list(rna, cds))
vg<-common_variant_genes(cds, rna, top_n = commonvariant)
length(vg)
X<-t(monocle3::normalized_counts(common_list[[1]][vg,]))
Xnew<-t(monocle3::normalized_counts(common_list[[2]][colnames(X),]))
labf<-as.factor(colData(rna)[["fCelltype"]])
labn<-as.numeric(labf)-1
labels<-levels(labf)
y<-matrix(model.matrix(~0+labf), ncol = length(labels))
ind <- sample(2, nrow(X), replace = TRUE, prob = c(0.8, 0.2))
Xtrain <- X[ind==1,]
Xtest <- X[ind==2,]
ytrain <- y[ind==1,]
ytest <- y[ind==2,]

startTime <- Sys.time()
cv.lasso <- cv.glmnet(X, y, family = "multinomial", type.multinomial = "grouped", parallel = T, trace.it=T) #lasso
endTime <- Sys.time()
print(endTime - startTime)

plot(cv.lasso)
cv.lasso$lambda.min

pred<-predict(cv.lasso, newx =Xnew, s = c("lambda.min"))
seu$lasso_celltype_g<-labels[apply(pred, 1, which.max)]
DimPlot_scCustom(seu, group.by = "lasso_celltype_g", colors_use =sfc(14))
DimPlot_scCustom(seu, group.by = "lasso_celltype", colors_use =sfc(14))


cm <- confusionMatrix(factor(seu$lasso_celltype_g), factor(seu$lasso_celltype), dnn = c("Prediction", "Reference"))

tab<-sweep(cm$table, 2, colSums(cm$table), "/")
plt <- as.data.frame(tab)
plt$Freq<-round(plt$Freq, 3)
plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))

ggplot(plt, aes(Prediction,Reference, fill= Freq)) +
  geom_tile() + geom_text(aes(label=Freq), color="white" ) +
  scale_fill_viridis_c(option = "A") +
  labs(x = "Reference",y = "Prediction") +
  theme(axis.text.x = element_text(angle=90))


#1 hour for grouped vs 5 min for default.  not worth the wait


debug(viewmastR)

undebug(common_variant_genes)
debug(viewmastR:::common_variant_seurat)

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



