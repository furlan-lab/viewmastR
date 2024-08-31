#debug.R
#remotes::install_github("extendr/rextendr")

rm(list=ls())


library(rextendr)

library(viewmastR)
#run_burn_helper()
#run_burn_helper2()
# run_mnist_mlr()
# run_mnist()
#readR(list(hello="hello"))


roxygen2::roxygenise()


rextendr::clean()
#rextendr::use_extendr()
rextendr::document()

# Run once to configure package to use pkgdown
#usethis::use_pkgdown()
usethis::use_pkgdown_github_pages()
# Run to build the website
pkgdown::build_site()

pkgdown::build_articles()

pkgdown::build_news()


pkgdown::build_home_index(); pkgdown::init_site()

library(viewmastR)
#remotes::install_github("satijalab/seurat", "seurat5", quiet = TRUE)
library(Seurat)
#library(viewmastR2)
library(ggplot2)

if(grepl("^gizmo", Sys.info()["nodename"])){
  ROOT_DIR1<-"/fh/fast/furlan_s/experiments/MB_10X_5p/cds"
  ROOT_DIR2<-"/fh/fast/furlan_s/grp/data/ddata/BM_data"
} else {
  ROOT_DIR1<-"/Users/sfurlan/Library/CloudStorage/OneDrive-SharedLibraries-FredHutchinsonCancerCenter/Furlan_Lab - General/experiments/MB_10X_5p/cds"
  ROOT_DIR2<-"/Users/sfurlan/Library/CloudStorage/OneDrive-SharedLibraries-FredHutchinsonCancerCenter/Furlan_Lab - General/datasets/Healthy_BM_greenleaf"
}
seu<-readRDS(file.path(ROOT_DIR1, "220302_final_object.RDS"))
seur<-readRDS(file.path(ROOT_DIR2, "230329_rnaAugmented_seurat.RDS"))

#seur<-seur[,sample(1:dim(seur)[2], 25000)]
DimPlot(seur)

labels<-levels(factor(seur$SFClassification))

seu$ground_truth<-factor(seu$seurat_clusters)
levels(seu$ground_truth)<-c(labels[11], #0
                            labels[18], #1
                            labels[11], #2
                            labels[17], #3
                            labels[20], #4
                            labels[21], #5
                            labels[14], #6
                            labels[12], #7
                            labels[16], #8,
                            labels[18], #9,
                            labels[14], #10,
                            labels[19], #11,
                            labels[10], #12,
                            labels[11], #13,
                            labels[11])
seu$ground_truth<-as.character(seu$ground_truth)
DimPlot(seu, group.by = "ground_truth", cols = seur@misc$colors)


seu<-calculate_gene_dispersion(seu)
plot_gene_dispersion(seu)
seu<-select_genes(seu, top_n = 10000, logmean_ul = -1, logmean_ll = -8)
plot_gene_dispersion(seu)
vgq<-get_selected_genes(seu)

seur<-calculate_gene_dispersion(seur)
plot_gene_dispersion(seur)
seur<-select_genes(seur, top_n = 10000, logmean_ul = -1, logmean_ll = -8)
plot_gene_dispersion(seur)
vgr<-get_selected_genes(seur)

vg<-intersect(vgq, vgr)

DimPlot(seur, group.by = "SFClassification", cols = seur@misc$colors)


seuC<-viewmastR::viewmastR(seu, seur, ref_celldata_col = "SFClassification", selected_genes = vg)

DimPlot(seuC, group.by = "viewmastR_pred", cols = seur@misc$colors)


# seu<-viewmastR(seu, seur, ref_celldata_col = "SFClassification", 
#                query_celldata_col = "viewmastR", selected_genes = vg,
#                FUNC = "softmax_regression", verbose=T)
#DimPlot(seu, group.by = "viewmastR", cols=seur@misc$colors)
undebug(setup_training)
training_list<-setup_training(seu, seur, ref_celldata_col = "SFClassification",selected_genes = vg, verbose=T, return_type = "list", use_sparse = F)
training_mats<-setup_training(seu, seur, ref_celldata_col = "SFClassification",selected_genes = vg, verbose=T, return_type = "matrix", use_sparse = F)
training_s4s<-setup_training(seu, seur, ref_celldata_col = "SFClassification",selected_genes = vg, verbose=T, return_type = "S4obj", use_sparse = F)
training_list$features
names(training_list)
training_list$labels


export_list<-process_learning_obj_mlr(train = training_list[["train"]], 
                                       test = training_list[["test"]], 
                                       query = training_list[["query"]], 
                                       labels = training_list[["labels"]], 
                                       learning_rate = 1e-3, num_epochs = 10, 
                                       directory = "/tmp/sc_local", verbose = TRUE, backend = "wgpu")

export_list$duration

#works, now make like old viewmastR



export_list<-process_learning_obj_mlr(train = training_list[["train"]], 
                                      test = training_list[["test"]], 
                                      query = training_list[["query"]], 
                                      labels = training_list[["labels"]], 
                                      learning_rate = 1e-3, num_epochs = 10, 
                                      directory = "/tmp/sc_local", verbose = TRUE, backend = "candle")

export_list$duration

export_list<-process_learning_obj_mlr(train = training_list[["train"]], 
                                      test = training_list[["test"]], 
                                      query = training_list[["query"]], 
                                      labels = training_list[["labels"]], 
                                      learning_rate = 1e-3, num_epochs = 10, 
                                      directory = "/tmp/sc_local", verbose = TRUE, backend = "nd")

export_list$duration


accuracy<-rbind(
  data.frame(epoch=1:length(export_list$history$train_acc), 
             metric=as.numeric(format(export_list$history$train_acc*100, digits=5)), 
             label="train_accuracy"),
  data.frame(epoch=1:length(export_list$history$test_acc), 
             metric=as.numeric(format(export_list$history$test_acc*100, digits=5)), 
             label="validation_accuracy"))
loss<-rbind(
  data.frame(epoch=1:length(export_list$history$test_loss), 
             metric=as.numeric(format(export_list$history$train_loss, digits=5)),
             label="train_loss"),
  data.frame(epoch=1:length(export_list$history$test_loss), 
             metric=as.numeric(format(export_list$history$test_loss, digits=5)),
             label="validation_loss"))

library(highcharter)
library(tidyr)

highcharter::hw_grid(ncol = 1,rowheight = 280,
  hchart(
    tibble::tibble(accuracy),
    "line",
    hcaes(x = epoch , y = metric, group = label),
    color = c(pals::glasbey(2))
  ) |> 
    hc_chart(
      backgroundColor = list(
        linearGradient = c(0, 0, 500, 500),
        stops = list(
          list(0, 'rgb(255, 255, 255)'),
          list(1, 'rgb(170, 230, 255)')
        )
      )
    ),
  hchart(
    tibble::tibble(loss),
    "line",
    hcaes(x = epoch , y = metric, group = label),
    color = c(pals::glasbey(2))
  ) |> 
    hc_chart(
      backgroundColor = list(
        linearGradient = c(0, 0, 500, 500),
        stops = list(
          list(0, 'rgb(255, 255, 255)'),
          list(1, 'rgb(170, 230, 255)')
        )
      )
    )
) %>% htmltools::browsable()


export_list<-process_learning_obj_ann(train = training_list[["train"]], 
                                      test = training_list[["test"]], 
                                      query = training_list[["query"]], 
                                      labels = training_list[["labels"]], 
                                      hidden_size = c(10000,10000), 
                                      learning_rate = 1e-3, num_epochs = 2, 
                                      directory = "/tmp/sc_local", verbose = TRUE)
export_list$params

export_list<-process_learning_obj_ann(train = training_list[["train"]], 
                                      test = training_list[["test"]], 
                                      query = training_list[["query"]], 
                                      labels = training_list[["labels"]], 
                                      hidden_size = 200, 
                                      learning_rate = 1e-3, num_epochs = 2, 
                                      directory = "/tmp/sc_local", verbose = TRUE)
export_list$params
export_list$duration

seu$viewmastRust_mlr<-training_list[["labels"]][export_list$predictions[[1]]+1]
DimPlot(seu, group.by = "viewmastRust_mlr", cols = seur@misc$colors)

confusion_matrix(factor(seu$viewmastRust_mlr), factor(seu$ground_truth), cols = seur@misc$colors)

confusion_matrix<-function(pred, gt, cols=NULL){
  mat<-table( pred, gt)
  labels = union(colnames(mat), rownames(mat))
  levels(gt)<-c(levels(gt), levels(pred)[!levels(pred) %in% levels(gt)])
  mat_full<-table( pred, gt)
  #deal with null colors
  if(is.null(cols)){
    cols = sample(grDevices::colors()[grep('gr(a|e)y', grDevices::colors(), invert = T)], length(labels))
    names(cols)<-labels
  }
  # } else {
  #   if(length(cols)!=length(labels)) stop("length of color vector provided is incorrect")
  # }
  mat_full<-mat_full[,match(rownames(mat_full), colnames(mat_full))]
  data<-caret::confusionMatrix(mat_full)
  pmat<-sweep(mat, MARGIN = 2, colSums(mat), "/")*100
  acc =format(as.numeric(data$overall[1])*100, digits=4)
  column_ha = ComplexHeatmap::HeatmapAnnotation(
     
    labels = colnames(mat),
    col = list(labels=cols),
    na_col = "black"
  )
  row_ha = ComplexHeatmap::rowAnnotation(
    
    labels = rownames(mat),
    col = list(labels=cols),
    na_col = "black"
  )
    ComplexHeatmap::Heatmap(pmat, col = scCustomize::viridis_light_high, cluster_rows = F, cluster_columns = F, 
                            row_names_side = "left", row_title = "Predicted Label", column_title = "True Label", 
                            name = "Percent of Column", column_title_side = "top", column_names_side = "top", 
                            bottom_annotation = column_ha, left_annotation = row_ha,
                            heatmap_legend_param = list(
                              title = paste0("Acc. ", acc, "\nPercent of Row")), 
                              rect_gp = grid::gpar(col = "white", lwd = 2),
                            cell_fun = function(j, i, x, y, width, height, fill){
    if(is.na(pmat[i,j])){
      grid::grid.text("NA", x, y, gp = grid::gpar(col="black", fontsize = 10))
    }else{
      if(pmat[i,j]>60){
        grid::grid.text(sprintf("%.f", mat[i, j]), x, y, gp = grid::gpar(col="black", fontsize = 10))
      }else{
        grid::grid.text(sprintf("%.f", mat[i, j]), x, y, gp = grid::gpar(col="white", fontsize = 10))
      }
    }
  })
  
  
  #pheatmap::pheatmap(tabd, color = c("white", colorRampPalette(c("thistle1", "purple"))(4000)),display_numbers = T,  cluster_cols = F, cluster_rows = F)
}

undebug(confusion_matrix)
confusion_matrix(factor(seu$viewmastRust), factor(seu$ground_truth), cols = seur@misc$colors)
confusion_matrix(factor(seu$viewmastRust_mlr), factor(seu$ground_truth), cols = seur@misc$colors)
confusion_matrix(factor(seu$viewmastRust_mlr_hilofeat), factor(seu$ground_truth), cols = seur@misc$colors)


src<-"
Rcpp::NumericVector computeSparseRowVariancesCPP(IntegerVector j, NumericVector val, NumericVector rm, int n) {
  const int nv = j.size();
  const int nm = rm.size();
  Rcpp::NumericVector rv(nm);
  Rcpp::NumericVector rit(nm);
  int current;
  // Calculate RowVars Initial
  for (int i = 0; i < nv; ++i) {
    current = j(i) - 1;
    rv(current) = rv(current) + (val(i) - rm(current)) * (val(i) - rm(current));
    rit(current) = rit(current) + 1;
  }
  // Calculate Remainder Variance
  for (int i = 0; i < nm; ++i) {
    rv(i) = rv(i) + (n - rit(i))*rm(i)*rm(i);
  }
  rv = rv / (n - 1);
  return(rv);
}
"
Rcpp::cppFunction(src)

mat<-seu@assays$RNA@counts
rM<-Matrix::rowMeans(mat)

#BiocManager::install("karyoploteR")
#BiocManager::install("ballgown")
library(karyoploteR)
library(ballgown)
# library(BSgenome.Hsapiens.UCSC.hg38)
# library(TxDb.Hsapiens.UCSC.hg38.knownGene)
# txdb <- TxDb.Hsapiens.UCSC.hg38.knownGene
# ls(txdb)
# transcripts(txdb)
f<-"/Users/sfurlan/Library/CloudStorage/OneDrive-SharedLibraries-FredHutchinsonCancerCenter/Furlan_Lab - General/resources/refs/cellranger_hg38.gtf"
gr<-gffReadGR(f)
gr<-gr[gr$type=="gene",]

regions <- createRandomRegions(nregions=100, length.mean = 1e3, mask=NA, non.overlapping=FALSE, genome = "hg38")

DimPlot(seu, group.by = "celltype")

library(scCustomize)
DimPlot(seu, group.by = "seurat_clusters")
DefaultAssay(seu)<-"ADT"
seu<-NormalizeData(seu, normalization.method = "CLR")
scCustomize::FeaturePlot_scCustom(seu, "FOXP3")

FeaturePlot_scCustom(seu, features = "TCRVa7.2")
FeaturePlot_scCustom(seu, features = "TCRVd2")
FeaturePlot_scCustom(seu, features = "CD161")
FeaturePlot_scCustom(seu, features = "CD45RA")
FeaturePlot_scCustom(seu, features = "CD45RO")
rownames(seu@assays$ADT)

DefaultAssay(seu)<-"RNA"
#label dataset
seu$ground_truth<-factor(seu$seurat_clusters)
levels(seu$ground_truth)<-c(training_list$labels[11], #0
                            training_list$labels[18], #1
                            training_list$labels[11], #2
                            training_list$labels[17], #3
                            training_list$labels[20], #4
                            training_list$labels[21], #5
                            training_list$labels[14], #6
                            training_list$labels[12], #7
                            training_list$labels[16], #8,
                            training_list$labels[18], #9,
                            training_list$labels[14], #10,
                            training_list$labels[19], #11,
                            training_list$labels[10], #12,
                            training_list$labels[11], #13,
                            training_list$labels[11])
seu$ground_truth<-as.character(seu$ground_truth)

gri<-gr[gr$gene_name %in% vg,]

kp <- plotKaryotype()
kpPlotCoverage(kp, data=gri)

training_list$labels

plot_signature<-function(training_list, granges_obj, celltype, flatten=F, log=F){
  tl<-training_list$test[sapply(training_list$test, "[[", 2)==celltype]
  d<-lapply(tl, "[[", 1)
  m<-do.call(cbind, d)
  rownames(m)<-training_list$features
  if(log){
    coverage<-log((rowSums(m)*10)+1)
  }else{
    coverage<-rowSums(m)
  }
  coverage<-ceiling(coverage)
  if(flatten){
    cv<-names(coverage)[as.numeric(coverage)>0]
  }else{
    cv<-rep(names(coverage), as.numeric(coverage))
  }
  grcov<-granges_obj[match(cv, granges_obj$gene_name),]
  kp <- plotKaryotype()
  kpPlotCoverage(kp, data=grcov, main=celltype)
}

plot_sc_signature<-function(training_list, granges_obj, celltype, flatten=F){
  tl<-training_list$test[sapply(training_list$test, "[[", 2)==celltype]
  d<-lapply(tl, "[[", 1)
  m<-d[[sample(1:length(d), 1)]]
  # m<-do.call(cbind, d)
  names(m)<-training_list$features
  coverage<-log((m*10)+1)
  coverage<-ceiling(coverage)
  if(flatten){
    cv<-names(coverage)[as.numeric(coverage)>0]
  }else{
    cv<-rep(names(coverage), as.numeric(coverage))
  }
  grcov<-granges_obj[match(cv, granges_obj$gene_name),]
  kp <- plotKaryotype()
  kpPlotCoverage(kp, data=grcov, main=celltype)
}
plot_sc_signature(training_list, granges_obj=gri, celltype=0, flatten = T)
plot_sc_signature(training_list, granges_obj=gri, celltype=0, flatten = F)
debug(plot_sc_signature)
plot_signature(training_list, granges_obj=gri, celltype=0, flatten = T)
plot_signature(training_list, granges_obj=gri, celltype=0, flatten = F)
plot_signature(training_list, granges_obj=gri, celltype=6, flatten = T)
plot_signature(training_list, granges_obj=gri, celltype=6, flatten = F)
plot_signature(training_list, granges_obj=gri, celltype=16, flatten = T)
plot_signature(training_list, granges_obj=gri, celltype=20, flatten = F)
plot_signature(training_list, granges_obj=gri, celltype=14, flatten = T)
plot_signature(training_list, granges_obj=gri, celltype=20, flatten = T)

library(rbenchmark)
benchmark(
  "cpp" = {
    computeSparseRowVariancesCPP(mat@i + 1, mat@x, rM, ncol(mat))
  },
  "rust" = {
    computeSparseRowVariances(mat@i + 1, mat@x, rM, ncol(mat))
  },
  replications = 10,
  columns = c("test", "replications", "elapsed",
              "relative", "user.self", "sys.self"))



system.time({ computeSparseRowVariancesCPP(mat@i + 1, mat@x, rM, ncol(mat)) })
system.time({ computeSparseRowVariances(mat@i + 1, mat@x, rM, ncol(mat)) })

lp<-installed.packages()[rownames(installed.packages())=="viewmastRust"][2]
lp
list.files(file.path(lp, "viewmastRust", "extdata", "mnist"))






install.packages("tensorflow")
library(keras)




model <- keras_model_sequential()

model %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = list('accuracy')
)
data <- matrix(rnorm(1000 * 32), nrow = 1000, ncol = 32)
labels <- matrix(rnorm(1000 * 10), nrow = 1000, ncol = 10)
val_data <- matrix(rnorm(1000 * 32), nrow = 100, ncol = 32)
val_labels <- matrix(rnorm(100 * 10), nrow = 100, ncol = 10)

model %>% fit(
  data,
  labels,
  epochs = 10,
  batch_size = 32,
  validation_data = list(val_data, val_labels)
)         

install.packages("highcharter")
install.packages("tidyseurat")
library(highcharter)
library(tidyseurat)

data("citytemp")

hc <- highchart() |> 
  hc_xAxis(categories = citytemp$month) |> 
  hc_add_series(
    name = "Tokyo", data = citytemp$tokyo
  ) |> 
  hc_add_series(
    name = "London", data = citytemp$london
  ) |> 
  hc_add_series(
    name = "Other city",
    data = (citytemp$tokyo + citytemp$london)/2
  )

hc



td<-tibble::tibble(data.frame(x=seur@reductions$umap@cell.embeddings[,1], y=seur@reductions$umap@cell.embeddings[,2], celltype = seur@meta.data$SFClassification))

hc <- hchart(
  td,
  "scatter",
  hcaes(x = x, y = y, group = celltype),
  colors = seur@misc$colors
)

hc


library(microbenchmark)
Rcpp::sourceCpp(
  code = '
#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
IntegerVector rcpp_zeros_intvec(int n) {
    IntegerVector my_vec(n);
    return my_vec;
}
// [[Rcpp::export]]
IntegerMatrix rcpp_zeros_intmat(int n) {
    IntegerMatrix my_mat(n, n);
    return my_mat;
}
// [[Rcpp::export]]
IntegerVector rcpp_zeros_intvec_dimmed(int n) {
    IntegerVector my_vec(n * n);
    my_vec.attr("dim") = Dimension(n, n);
    return my_vec;
}
')
rextendr::rust_source(
  profile = "release",
  extendr_deps = list(
    `extendr-api` = list(git = "https://github.com/extendr/extendr")
  ),
  code = '
/// @export
#[extendr]
fn rust_zeros_intvec(n: i32) -> Robj {
    let my_vec = vec!(0; n as usize);
    r!(my_vec)
}
/// @export
#[extendr]
fn rust_zeros_intmat(n: i32) -> Robj {
    let my_mat = RMatrix::new_matrix(n as usize, n as usize, |_, _| 0);
    r!(my_mat)
}
/// @export
#[extendr]
fn rust_zeros_intmat_viavec(n: i32) -> Robj {
    let my_vec = vec!(0; n as usize * n as usize);
    let my_mat: RMatrix<i32> = r!(my_vec).as_matrix().unwrap();
    r!(my_mat)
}
extendr_module! {
    mod rust_wrap;
    fn rust_zeros_intvec;
    fn rust_zeros_intmat;
    fn rust_zeros_intmat_viavec;
}
')
cat("\n************ integer vector\n")
cat("cpp and rust all.equal? ")
print(all.equal(rcpp_zeros_intvec(100), rust_zeros_intvec(100)))
cat("\n************ integer matrix direct\n")
cat("cpp and rust all.equal? ")
print(all.equal(rcpp_zeros_intmat(100), rust_zeros_intmat(100)))
cat("\n************ integer matrix via vec\n")
cat("cpp and rust all.equal? ")
cat("No rust solution to compare to yet.")
cat("\n************ microbench results\n")
for (N in cumprod(c(10, rep(2, 6)))) {
  cat("N = ", N, "\n")
  mb_ziv_out <- microbenchmark(rcpp_zeros_intvec(N), rust_zeros_intvec(N),
                               rcpp_zeros_intmat(N), rust_zeros_intmat(N),
                               rcpp_zeros_intvec_dimmed(N), times=1000)
  print(mb_ziv_out)
}

