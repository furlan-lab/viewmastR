#debug.R
#remotes::install_github("extendr/rextendr")

rm(list=ls())
roxygen2::roxygenise()
roxygen2::parse_package()
rextendr::clean()
rextendr::document()


# Run once to configure package to use pkgdown
usethis::use_pkgdown_github_pages()
pkgdown::clean_site()
pkgdown::build_site()
pkgdown::build_articles()
pkgdown::build_news()
pkgdown::build_home_index(); 
pkgdown::init_site()

pkgdown::build_site_github_pages()

rm(list = ls())
suppressPackageStartupMessages({
  library(viewmastR)
  library(Seurat)
  library(ggplot2)
  library(scCustomize)
})

if (grepl("^gizmo", Sys.info()["nodename"])) {
  ROOT_DIR1 <- "/fh/fast/furlan_s/experiments/MB_10X_5p/cds"
  ROOT_DIR2 <- "/fh/fast/furlan_s/grp/data/ddata/BM_data"
} else {
  ROOT_DIR1 <- "/Users/sfurlan/Library/CloudStorage/OneDrive-SharedLibraries-FredHutchinsonCancerCenter/Furlan_Lab - General/experiments/MB_10X_5p/cds"
  ROOT_DIR2 <- "/Users/sfurlan/Library/CloudStorage/OneDrive-SharedLibraries-FredHutchinsonCancerCenter/Furlan_Lab - General/datasets/Healthy_BM_greenleaf"
}

# Load query and reference datasets
seu <- readRDS(file.path(ROOT_DIR1, "240813_final_object.RDS"))
vg <- get_selected_genes(seu)
seur <- readRDS(file.path(ROOT_DIR2, "230329_rnaAugmented_seurat.RDS"))

# View training history
output_list <- viewmastR(seu, seur, ref_celldata_col = "SFClassification", selected_genes = vg, return_type = "list")
DimPlot(output_list$object, group.by = "viewmastR_pred")
table(output_list$object$viewmastR_pred)
## (ALL TIMES elapsed)
## on M1 
baseline <- system.time(seu <- viewmastR(seu, seur, ref_celldata_col = "SFClassification", selected_genes = vg, backend = "candle", max_epochs = 4))
#28.870
## on M2 
baseline <- system.time(seu <- viewmastR(seu, seur, ref_celldata_col = "SFClassification", selected_genes = vg, backend = "candle", max_epochs = 4))
#25.041
edits1 <- system.time(seu <- viewmastR(seu, seur, ref_celldata_col = "SFClassification", selected_genes = vg, backend = "candle", max_epochs = 4))
#24.901 #fixed Recreating Loss Function in Each Iteration
edits2 <- system.time(seu <- viewmastR(seu, seur, ref_celldata_col = "SFClassification", selected_genes = vg, backend = "candle", max_epochs = 4))
#25.160 #fixed  Inefficient Computation of num_predictions
edits3 <- system.time(seu <- viewmastR(seu, seur, ref_celldata_col = "SFClassification", selected_genes = vg, backend = "candle", max_epochs = 4))
#24.493 #fixed query prediction
edits5 <- system.time(seu <- viewmastR(seu, seur, ref_celldata_col = "SFClassification", selected_genes = vg, backend = "candle", max_epochs = 4))
#38.857 model save - removed clone (battery low)

## on intel
baseline <- system.time(seu <- viewmastR(seu, seur, ref_celldata_col = "SFClassification", selected_genes = vg, backend = "candle", max_epochs = 4))
#41.322
baseline_wgpu <- system.time(seu <- viewmastR(seu, seur, ref_celldata_col = "SFClassification", selected_genes = vg, backend = "wgpu", max_epochs = 4))
#38.792
edits6 <- system.time(seu <- viewmastR(seu, seur, ref_celldata_col = "SFClassification", selected_genes = vg, backend = "wgpu", max_epochs = 4))
#36.400 #minor changes
edits7 <- system.time(seu <- viewmastR(seu, seur, ref_celldata_col = "SFClassification", selected_genes = vg, backend = "wgpu", max_epochs = 4))
#38.857 #minor changes
edits8 <- system.time(seu <- viewmastR(seu, seur, ref_celldata_col = "SFClassification", selected_genes = vg, backend = "wgpu", max_epochs = 4))
#34.034 #minor changes (update number of cores)
edits9 <- system.time(seu <- viewmastR(seu, seur, ref_celldata_col = "SFClassification", selected_genes = vg, backend = "wgpu", max_epochs = 4))
#32.875 #refactored code to run using step functions...
edits9cl <- system.time(seu <- viewmastR(seu, seur, ref_celldata_col = "SFClassification", selected_genes = vg, backend = "candle", max_epochs = 4))
#38.548
edits10 <- system.time(seu <- viewmastR(seu, seur, ref_celldata_col = "SFClassification", selected_genes = vg, backend = "wgpu", max_epochs = 4))
#32.371 moved batch reporting to the last batch
edits11 <- system.time(seu <- viewmastR(seu, seur, ref_celldata_col = "SFClassification", selected_genes = vg, backend = "wgpu", max_epochs = 4))
#32.609 removed cloning of batch.targets

#####NN######
## on intel
baseline <- system.time(seu <- viewmastR(seu, seur, ref_celldata_col = "SFClassification", FUNC = "nn", hidden_layers = c(1000), selected_genes = vg, backend = "wgpu", max_epochs = 4))
#41.322
edits11 <- system.time(seu <- viewmastR(seu, seur, ref_celldata_col = "SFClassification", FUNC = "nn", hidden_layers = c(1000), selected_genes = vg, backend = "wgpu", max_epochs = 4))
#41.322



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

