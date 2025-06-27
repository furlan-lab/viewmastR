

#' @export
plot_gene_dispersion<-function(cds, size=1, alpha=0.4){
  if(class(cds)=="cell_data_set"){
    if(is.null(cds@int_metadata$dispersion$disp_func)){
      prd<-cds@int_metadata$dispersion
      prd$selected_features<-prd$use_for_ordering
      g<-ggplot2::ggplot(prd, ggplot2::aes(x = log_mean, y = fit)) 
      if("use_for_ordering" %in% colnames(cds@int_metadata$dispersion)){
        g <- g + ggplot2::geom_point(data=prd, ggplot2::aes(x=log_mean, y=log_dispersion, color=selected_features), alpha=alpha, size=size)+
          scale_color_manual(values=c("lightgray", "firebrick1"))
      }else{
        g <- g + ggplot2::geom_point(data=prd, ggplot2::aes( x=log_mean, y=log_dispersion), color="grey", alpha=alpha, size=size)
      }
      g<-g+
        ggplot2::theme_bw() +
        ggplot2::geom_line() # + 
      #ggplot2::geom_smooth(data=prd, ggplot2::aes(ymin = lci, ymax = uci), stat = "identity")
      return(g)
    }else{
      prd<-cds@int_metadata$dispersion$disp_table
      prd$fit<-log(cds@int_metadata$dispersion$disp_func(prd$mu))
      prd$mu<-log(prd$mu)
      prd$disp<-log(prd$disp)
      colnames(prd)<-c("log_mean", "log_dispersion", "gene_id", "fit")
      g<-ggplot2::ggplot(prd, ggplot2::aes(x = log_mean, y = fit)) 
      if("use_for_ordering" %in% names(cds@int_metadata$dispersion)){
        prd$selected_features = cds@int_metadata$dispersion$use_for_ordering
        g <- g + ggplot2::geom_point(data=prd, ggplot2::aes(x=log_mean, y=log_dispersion, color=selected_features, alpha=alpha), size=size)
      }else{
        g <- g + ggplot2::geom_point(data=prd, ggplot2::aes( x=log_mean, y=log_dispersion, color="grey", alpha=alpha), size=size)
      }
      g<-g+
        ggplot2::theme_bw() +
        ggplot2::geom_line(data=prd, ggplot2::aes( x=log_mean, y=fit)) +
        ggplot2::geom_smooth(data=prd, formula = fit ~ log_mean, stat = "identity") + 
        xlab("log mean expression")+ylab("log dispersion")
      return(g)
    }
  }
  if(class(cds)=="Seurat"){
    prd<-cds@misc$dispersion
    prd$selected_features<-prd$use_for_ordering
    g<-ggplot2::ggplot(prd, ggplot2::aes(x = log_mean, y = fit)) 
    if("use_for_ordering" %in% colnames(cds@misc$dispersion)){
      g <- g + ggplot2::geom_point(data=prd, ggplot2::aes(x=log_mean, y=log_dispersion, color=selected_features), alpha=alpha, size=size)+
        scale_color_manual(values=c("lightgray", "firebrick1"))
    }else{
      g <- g + ggplot2::geom_point(data=prd, ggplot2::aes( x=log_mean, y=log_dispersion), color="grey", alpha=alpha, size=size)
    }
    g<-g+
      ggplot2::theme_bw() +
      ggplot2::geom_line() # + 
    #ggplot2::geom_smooth(data=prd, ggplot2::aes(ymin = lci, ymax = uci), stat = "identity")
    return(g)
  }
  
}

#' Select genes in a cell_data_set for dimensionality reduction
#'
#' @description Monocle3 aims to learn how cells transition through a
#' biological program of gene expression changes in an experiment. Each cell
#' can be viewed as a point in a high-dimensional space, where each dimension
#' describes the expression of a different gene. Identifying the program of
#' gene expression changes is equivalent to learning a \emph{trajectory} that
#' the cells follow through this space. However, the more dimensions there are
#' in the analysis, the harder the trajectory is to learn. Fortunately, many
#' genes typically co-vary with one another, and so the dimensionality of the
#' data can be reduced with a wide variety of different algorithms. Monocle3
#' provides two different algorithms for dimensionality reduction via
#' \code{reduce_dimensions} (UMAP and tSNE). The function
#' \code{select_features} is an optional step in the trajectory building
#' process before \code{preprocess_cds}.  After calculating dispersion for
#' a cell_data_set using the \code{calculate_gene_dispersion} function, the 
#' \code{select_genes} function allows the user to identify a set of genes
#' that will be used in downstream dimensionality reduction methods.
#'
#'
#' @param cds the cell_data_set upon which to perform this operation.
#' @param fit_min the minimum multiple of the dispersion fit calculation; default = 1
#' @param fit_max the maximum multiple of the dispersion fit calculation; default = Inf
#' @param logmean_ul the maximum multiple of the dispersion fit calculation; default = Inf
#' @param logmean_ll the maximum multiple of the dispersion fit calculation; default = Inf
#' @param top top_n if specified, will override the fit_min and fit_max to select the top n most 
#' variant features.  logmena_ul and logmean_ll can still be used.
#' @return an updated cell_data_set object with selected features 
#' @export

select_genes<-function(cds, fit_min=1, fit_max=Inf, logmean_ul=Inf, logmean_ll=-Inf, top_n=NULL){
  if(class(cds)=="cell_data_set"){
    if(is.null(cds@int_metadata$dispersion$disp_func)){
      df<-cds@int_metadata$dispersion
      df$ratio<-df$log_dispersion/df$fit
      df$index<-1:nrow(df)
      if(!is.null(top_n)){
        in_range<-df[which(df$log_mean > logmean_ll & df$log_mean < logmean_ul),]
        cds@int_metadata$dispersion$use_for_ordering <- df$index %in% in_range[order(-in_range$ratio),][1:top_n,]$index
      }else{
        cds@int_metadata$dispersion$use_for_ordering <- df$ratio > fit_min & df$ratio < fit_max & df$log_mean > logmean_ll & df$log_mean < logmean_ul
      }
      return(cds)
    }else{
      df<-cds@int_metadata$dispersion$disp_table
      df$fit<-cds@int_metadata$dispersion$disp_func(df$mu)
      df$ratio<-df$disp/df$fit
      df$log_disp=log(df$disp)
      df$log_mean<-log(df$mu)
      df$index<-1:nrow(df)
      if(!is.null(top_n)){
        in_range<-df[which(df$log_mean > logmean_ll & df$log_mean < logmean_ul),]
        cds@int_metadata$dispersion$use_for_ordering <- df$index %in% in_range[order(-in_range$ratio),][1:top_n,]$index
      }else{
        cds@int_metadata$dispersion$use_for_ordering <- df$ratio > fit_min & df$ratio < fit_max & df$log_mean > logmean_ll & df$log_mean < logmean_ul
      }
      return(cds)
    }
  }
  if(class(cds)=="Seurat"){
    df<-cds@misc$dispersion
    df$ratio<-df$log_dispersion/df$fit
    df$index<-1:nrow(df)
    if(!is.null(top_n)){
      in_range<-df[which(df$log_mean > logmean_ll & df$log_mean < logmean_ul),]
      cds@misc$dispersion$use_for_ordering <- df$index %in% in_range[order(-in_range$ratio),][1:top_n,]$index
    }else{
      cds@misc$dispersion$use_for_ordering <- df$ratio > fit_min & df$ratio < fit_max & df$log_mean > logmean_ll & df$log_mean < logmean_ul
    }
    return(cds)
  }
}


#' @export
get_selected_features<-function(cds, gene_column="id"){
  if(class(cds)=="cell_data_set"){
    if(is.null(cds@int_metadata$dispersion$disp_func)){
      return(as.character(cds@int_metadata$dispersion[[gene_column]][cds@int_metadata$dispersion$use_for_ordering]))
    }else{
      return(as.character(cds@int_metadata$dispersion$disp_table[[gene_column]][cds@int_metadata$dispersion$use_for_ordering]))
    }
  }
  if(class(cds)=="Seurat"){
    return(as.character(cds@misc$dispersion[[gene_column]][cds@misc$dispersion$use_for_ordering]))
  }
}

#' @export
set_selected_features<-function(cds, genes, gene_column="id", unique_column="id"){
  if(class(cds)=="cell_data_set"){
    if(is.null(cds@int_metadata$dispersion$disp_func)){
      if(is.null(cds@int_metadata$dispersion)){
        cds@int_metadata$dispersion$use_for_ordering = rownames(cds) %in% genes
        return(cds)
      }
      if(gene_column %in% colnames(cds@int_metadata$dispersion)){
        cds@int_metadata$dispersion$use_for_ordering = cds@int_metadata$dispersion[[gene_column]] %in% genes
      }
      if(length(which(cds@int_metadata$dispersion$use_for_ordering))<1) warning("No ordering genes found")
    }else{
      if(gene_column %in% colnames(cds@int_metadata$dispersion$disp_table)){
        cds@int_metadata$dispersion$use_for_ordering = cds@int_metadata$dispersion$disp_table[[gene_column]] %in% genes
      }else{
        found<-rownames(fData(cds))[fData(cds)[[gene_column]] %in% genes]
        cds@int_metadata$dispersion$use_for_ordering = cds@int_metadata$dispersion$disp_table[[unique_column]] %in% found
      }
    }
    return(cds)
  }
  if(class(cds)=="Seurat"){
    if(is.null(cds@misc$dispersion)){
      cds@misc$dispersion$use_for_ordering = rownames(cds) %in% genes
      return(cds)
    }
    if(gene_column %in% colnames(cds@misc$dispersion)){
      cds@misc$dispersion$use_for_ordering = cds@misc$dispersion[[gene_column]] %in% genes
    }
    if(length(which(cds@misc$dispersion$use_for_ordering))<1) warning("No ordering genes found")
    return(cds)
  }
}


#' Calculate dispersion genes in a cell_data_set object
#'
#' @description Monocle3 aims to learn how cells transition through a
#' biological program of gene expression changes in an experiment. Each cell
#' can be viewed as a point in a high-dimensional space, where each dimension
#' describes the expression of a different gene. Identifying the program of
#' gene expression changes is equivalent to learning a \emph{trajectory} that
#' the cells follow through this space. However, the more dimensions there are
#' in the analysis, the harder the trajectory is to learn. Fortunately, many
#' genes typically co-vary with one another, and so the dimensionality of the
#' data can be reduced with a wide variety of different algorithms. Monocle3
#' provides two different algorithms for dimensionality reduction via
#' \code{reduce_dimensions} (UMAP and tSNE). The function
#' \code{calculate_dispersion} is an optional step in the trajectory building
#' process before \code{preprocess_cds}.  After calculating dispersion for
#' a cell_data_set using the \code{calculate_gene_dispersion} function, the 
#' \code{select_genes} function allows the user to identify a set of genes
#' that will be used in downstream dimensionality reduction methods.  These
#' genes and their disperion and mean expression can be plotted using the 
#' \code{plot_gene_dispersion} function.
#'
#'
#' @param cds the cell_data_set upon which to perform this operation.
#' @param q the polynomial degree; default = 3.
#' @param id_tag the name of the feature data column corresponding to 
#' the unique id - typically ENSEMBL id; default = "id".
#' @param symbol_tag the name of the feature data column corresponding to 
#' the gene symbol; default = "gene_short_name".
#' @return an updated cell_data_set object with dispersion and mean expression saved
#' @export

calculate_gene_dispersion<-function(cds, q=3, id_tag="id", symbol_tag="gene_short_name", method="m3addon", removeOutliers=T){
  software<-NULL
  if(class(cds)=="Seurat"){
    software<-"seurat"
  }
  if(class(cds)=="cell_data_set"){
    software<-"monocle3"
  }
  if(is.null(software)){stop("software not found for input objects")}
  if(software=="monocle3"){
    cds@int_metadata$dispersion<-NULL
    if(method=="m2"){
      df<-data.frame(calc_dispersion_m2(obj = cds, min_cells_detected = 1, min_exprs=1, id_tag=id_tag))
      fdat<-fData(cds)
      if (!is.list(df)) 
        stop("Parametric dispersion fitting failed, please set a different lowerDetectionLimit")
      disp_table <- subset(df, is.na(mu) == FALSE)
      res <- monocle:::parametricDispersionFit(disp_table, verbose = T)
      fit <- res[[1]]
      coefs <- res[[2]]
      if (removeOutliers) {
        CD <- cooks.distance(fit)
        cooksCutoff <- 4/nrow(disp_table)
        message(paste("Removing", length(CD[CD > cooksCutoff]), 
                      "outliers"))
        outliers <- union(names(CD[CD > cooksCutoff]), setdiff(row.names(disp_table), 
                                                               names(CD)))
        res <- monocle:::parametricDispersionFit(disp_table[row.names(disp_table) %in% 
                                                              outliers == FALSE, ], verbose=T)
        fit <- res[[1]]
        coefs <- res[[2]]
        names(coefs) <- c("asymptDisp", "extraPois")
        ans <- function(q) coefs[1] + coefs[2]/q
        attr(ans, "coefficients") <- coefs
      }
      res <- list(disp_table = disp_table, disp_func = ans)
      cds@int_metadata$dispersion<-res
      return(cds)
    }
    if(method=="m3addon"){
      ncounts<-Matrix::t(Matrix::t(exprs(cds))/monocle3::size_factors(cds))
      m<-Matrix::rowMeans(ncounts)
      sd<-sqrt(sparseRowVariances(ncounts))
      fdat<-fData(cds)
      cv<-sd/m*100
      df<-data.frame(log_dispersion=log(cv), log_mean=log(m))
      df[[id_tag]]<-fdat[[id_tag]]
      df<-df[is.finite(df$log_dispersion),]
      model <- lm(data = df, log_dispersion ~ log_mean + poly(log_mean, degree=q))
      prd <- data.frame(log_mean = df$log_mean)
      err<-suppressWarnings(predict(model, newdata= prd, se.fit = T))
      prd$lci <- err$fit - 1.96 * err$se.fit
      prd$fit <- err$fit
      prd$uci <- err$fit + 1.96 * err$se.fit
      prd$log_dispersion<-df$log_dispersion
      prd[[id_tag]]<-df[[id_tag]]
      cds@int_metadata$dispersion<-prd
      return(cds)
    }
  }
  if(software=="seurat"){
    if(method=="m2"){
      stop("m2 method not supported for seurat objects")
    }
    if(method=="m3addon"){
      ncounts<-get_norm_counts(cds)
      m<-Matrix::rowMeans(ncounts)
      sd<-sqrt(sparseRowVariances(ncounts))
      #fdat<-fData(cds)
      cv<-sd/m*100
      df<-data.frame(log_dispersion=log(cv), log_mean=log(m))
      df[[id_tag]]<-rownames(cds)
      df<-df[is.finite(df$log_dispersion),]
      model <- lm(data = df, log_dispersion ~ log_mean + poly(log_mean, degree=q))
      prd <- data.frame(log_mean = df$log_mean)
      err<-suppressWarnings(predict(model, newdata= prd, se.fit = T))
      prd$lci <- err$fit - 1.96 * err$se.fit
      prd$fit <- err$fit
      prd$uci <- err$fit + 1.96 * err$se.fit
      prd$log_dispersion<-df$log_dispersion
      prd[[id_tag]]<-df[[id_tag]]
      #prd$gene_short_name<-fdat[[symbol_tag]][match(prd[[id_tag]], fdat[[id_tag]])]
      cds@misc$dispersion<-prd
      return(cds)
    }
  }
}

#' @export
#' @import monocle3
#' @importFrom Matrix rowSums
#' @importFrom DelayedMatrixStats rowMeans2
#' @importFrom DelayedMatrixStats rowVars
#' 

calc_dispersion_m2<-function (obj, expressionFamily, min_cells_detected=1, min_exprs = 1, id_tag="id") 
{
  if(class(obj)=="cell_data_set"){
    rounded <- round(exprs(cds))
    nzGenes <- rowSums(rounded > min_exprs)
    nzGenes <- names(nzGenes[nzGenes > min_cells_detected])
    x <- DelayedArray(t(t(rounded[nzGenes, ])/pData(cds[nzGenes, 
    ])$Size_Factor))
    xim <- mean(1/pData(cds[nzGenes, ])$Size_Factor)
    if (class(exprs(cds)) %in% c("dgCMatrix", "dgTMatrix")) {
      f_expression_mean <- as(DelayedMatrixStats::rowMeans2(x), 
                              "sparseVector")
    }else {
      f_expression_mean <- DelayedMatrixStats::rowMeans2(x)
    }
    f_expression_var <- DelayedMatrixStats::rowVars(x)
    disp_guess_meth_moments <- f_expression_var - xim * f_expression_mean
    disp_guess_meth_moments <- disp_guess_meth_moments/(f_expression_mean^2)
    res <- data.frame(mu = as.vector(f_expression_mean), disp = as.vector(disp_guess_meth_moments))
    res[res$mu == 0]$mu = NA
    res[res$mu == 0]$disp = NA
    res$disp[res$disp < 0] <- 0
    res[[id_tag]] <- row.names(fData(cds[nzGenes, ]))
    return(res)
  }
  
  if(class(obj) %in% "matrix"){
    sf<-DESeq2::estimateSizeFactorsForMatrix(obj)
    x <-DelayedArray(t(t(obj)/sf))
    f_expression_mean <- DelayedMatrixStats::rowMeans2(obj)
    f_expression_var <- DelayedMatrixStats::rowVars(obj)
    xim <- mean(1/sf)
    disp_guess_meth_moments <- f_expression_var - xim * f_expression_mean
    disp_guess_meth_moments <- disp_guess_meth_moments/(f_expression_mean^2)
    res <- data.frame(mu = as.vector(f_expression_mean), disp = as.vector(disp_guess_meth_moments))
    res[res$mu == 0]$mu = NA
    res[res$mu == 0]$disp = NA
    res$disp[res$disp < 0] <- 0
    res[[id_tag]] <- row.names(data)
    res
  }
  # m3vars<-m3addon:::rowStdDev(as(x, "sparseMatrix"))
  # head(m3vars[1,]^(2))
  # head(f_expression_var)
}




#' @export
#' @title Finds common features in a list of single cell objects
#'
#' @description Machine learning algorithms often require features to be the same across 
#' datasets.  This function finds common features between a list of cell data set objects (monocle3) and 
#' returns a list of cds's that have the same features.  Note that this function uses rownames 
#' of the 'fData' DataFrame (monocle3) and the rownames of the seurat_object to find the intersect of features common to all objects
#'
#' @param cds_list Input  object.
#' @export
common_features <- function(cds_list){
  len<-length(cds_list)
  common_features=vector()
  software<-NULL
  for(i in 1:len){
    if(i < 2){
      if(class(cds_list[[i]])=="Seurat"){software<-"seurat"}
      if(class(cds_list[[i]])=="cell_data_set"){software<-"monocle3"}
      if(is.null(software)){stop("software not found for input object 1")}
      if(software=="monocle3"){
        common_features<-rownames(fData(cds_list[[i]]))
      }
      if(software=="seurat"){
        common_features<-rownames(cds_list[[i]])
      }
    }else{
      if(software=="monocle3"){
        common_features<-unique(intersect(common_features, rownames(fData(cds_list[[i]]))))
      }
      if(software=="seurat"){
        common_features<-unique(intersect(common_features, rownames(cds_list[[i]])))
      }
    }
  }
  if(software=="monocle3"){
    for(i in 1:len){
      cds_list[[i]]<-cds_list[[i]][match(common_features, rownames(cds_list[[i]])),]
    }
    return(cds_list)
  }
  if(software=="seurat"){
    for(i in 1:len){
      # mat<-cds_list[[i]]@assays[[cds_list[[i]]@active.assay]]@counts
      mat <- get_counts_seurat(cds_list[[i]])
      seutmp<- CreateSeuratObject(counts = mat[common_features, ]) # Create a new Seurat object with just the genes of interest
      cds_list[[i]] <- AddMetaData(object = seutmp, metadata = cds_list[[i]]@meta.data) # Add the idents to the meta.data slot
      rm(seutmp, mat)
    }
    return(cds_list)
  }
}














