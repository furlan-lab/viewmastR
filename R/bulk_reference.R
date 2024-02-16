#' Bulk reference
#' @description This function creates a seurat object (typically single cell genomics) of multiple single cell profiles from each sample from a 
#' bulk object (SummarizedExperiment object currently supported).  In doing so, the function creates single cell profiles with a size distribution
#' that approximates the provided single cell object (query)
#' @param query a single cell object (Seurat) with a size distribution of counts to be mimicked in the assay argument
#' @param ref the reference object (Summarized Experiment)
#' @param assay the assay slot of the query (Seurat)
#' @param bulk_feature_row the column name of gene symbols in ref
#' @param bulk_assay_name the name of the assay object in the ref
#' @param dist IN DEVELOPMENT
#' @return a classified seurat object labeled according to the bulk reference
#' @importFrom pbmcapply pbmclapply
#' @importFrom parallel detectCores
#' @export
splat_bulk_reference<-function(query=NULL, 
                               ref, N = 2, assay="RNA", 
                               bulk_feature_row = "gene_short_name", 
                               bulk_assay_name = "counts", 
                               dist=c("sc-direct", "sc-mimic", "bulk")){
  dist<-match.arg(dist)
  if(dist=="sc-mimic" | dist == "sc-direct"){
    message("Finding count distribution of query")
    sizes <- colSums(get_counts_seurat(query))
    den <- density(sizes)
    replace_counts<-F
  }else{
    sizes <- colSums(assays(ref)[[bulk_assay_name]])
    den <- density(sizes)
    replace_counts=T
  }
  message("Finding common features between ref and query")
  universe<-intersect(rowData(ref)[[bulk_feature_row]], rownames(query))
  message(paste0("Simulating ", N, " single cells for every bulk dataset case"))
  
  counts<-get_counts_se(ref, bulk_assay_name)[match(universe, rowData(ref)[[bulk_feature_row]]),]
  rownames(counts)<-universe
  newdata<-pbmcapply::pbmclapply(1:(dim(ref)[2]), function(n){
    newsizes <- sample(sizes, N, replace = TRUE) + rnorm(N * 10, 0, den$bw)
    trimmed_newdata <- round(newsizes[newsizes > min(sizes) &
                                        max(sizes)], 0)
    final_newsizes <- sample(trimmed_newdata, N)
    rsums <- counts[,n]
    if(is.null(names(rsums))){
      names(rsums)<-rownames(counts)
    }
    splat <- names(rsums)[rep(seq_along(rsums), rsums)]
    dl <- lapply(final_newsizes, function(i) {
      tab <- table(sample(splat, i, replace=replace_counts))
      nf <- universe[!universe %in% names(tab)]
      all <- c(tab, setNames(rep(0, length(nf)), nf))
      all[match(universe, names(all))]
    })
    return(as.sparse(do.call(cbind, dl)))
  }, mc.cores = parallel::detectCores())

  output<-lapply(newdata, function(x) class(x)[1])
  good_ind<-lapply(output, function(x) x=="dgCMatrix")
  good_ind<-which(unlist(good_ind))

  metai<-lapply(good_ind, function(n) rep(n, N)) %>% do.call("c", .)
  meta<-colData(ref)
  newmeta<-meta[metai,]
  gooddata<-newdata[good_ind]
  tmpdata<-Reduce(cbind, gooddata[-1], gooddata[[1]])
  dim(tmpdata)
  dim(newmeta)
  rownames(newmeta)<-make.unique(as.character(metai))
  colnames(tmpdata)<-rownames(newmeta)
  CreateSeuratObject(tmpdata, meta.data = as.data.frame(newmeta))
}


get_counts_se<-function(obj, bulk_assay_name = "counts"){
  #step 1 check for counts
  counts<-assays(obj)[[bulk_assay_name]]
  #step 2 if not found grab any item in the assays slot if == 1
  if(is.null(counts)){
    if(length(assays(obj))>1){stop("More than one assay object found")}
    if(length(assays(obj))==0){stop("No assay object found")}
    counts<-assays(obj)[[1]]
  }
  counts
}