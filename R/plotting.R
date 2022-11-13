#' Plot geneset scores as summary plot
#'
#' @description Geneset scores are a score calculated for each single cell derived from \
#' more than one gene.  This function plots geneset scores grouped with a boxplot overlaying a violin\
#' overlaying a jitter of the raw data.
#' 
#' When using method 'totals', the sum of the size-factor corrected, log-normalized gene \
#' expression for a give set of genes is performed.  When using method 'corrected', single \
#' cell scores for a give gene set that have been "corrected" using 100X genes with similar \
#' expression levels.

#' @param cds Input cell_data_set object.
#' @param marker_set Vector of genes in the gene_metadata DataFrame (fData) that can be found in the column 'fData_col'
#' @param name Name given to the geneset
#' @param fData_col Character string denoting the gene_metadata DataFrame (fData) column that contains the genes in marker_set1.  Default = 'gene_short_name'
#' @param method 'totals' or 'corrected'.  See estimate_score and estimate_corrected_score for more information
#' @return Plot
#' @import ggplot2 monocle3
#' @export
#' @references Puram, S. V. et al. Single-Cell Transcriptomic Analysis of Primary and 
#' Metastatic Tumor Ecosystems in Head and Neck Cancer. Cell 171, 1611.e1–1611.e24 (2017).


plot_grouped_geneset<-function(cds, marker_set, name, by, fData_col= "gene_short_name", scale="width", facet=NULL, adjust=1.4, size=0.05, alpha=0.1,
                               method="totals", overlay_violinandbox=T, box_width=0.3, rotate_x=T, jitter=T, return_values=F){
  if(method=="totals") pData(cds)[[name]]<-estimate_score(cds, marker_set, fData_col= fData_col)
  if(method=="corrected") pData(cds)[[name]]<-estimate_corrected_score(cds, marker_set, fData_col= fData_col)
  scores<-data.frame(pData(cds)[[name]], as.factor(pData(cds)[[by]]), stringsAsFactors = F)
  if(!is.null(facet)){
    scores<-cbind(scores, pData(cds)[[facet]])
    colnames(scores)<-c(name, by, facet)
  }else{
    colnames(scores)<-c(name, by)
  }
  g<- ggplot(scores, aes_string(x=by, y=name, fill=by))
  if(jitter)g<-g + geom_jitter(size=size, alpha=alpha)
  if(overlay_violinandbox){
    g<-g+geom_violin(scale="width")+geom_boxplot(width=box_width, fill="white", outlier.size = 0)
  }
  if(!is.null(facet)){
    g<-g+facet_wrap(as.formula(paste("~", facet)), scales = "free")
  }
  if(rotate_x) {
    g<-g+theme(axis.text.x=element_text(angle=90, hjust=0.95,vjust=0.2))
  }
  if(return_values)list(plot=g, scores=scores) else(g)
}



#' Plot geneset scores as cells
#'
#' @description Geneset scores are a score calculated for each single cell derived from \
#' more than one gene.  This function plots geneset scores using monocle3's 'plot_genes' function.
#' 
#' When using method 'totals', the sum of the size-factor corrected, log-normalized gene \
#' expression for a give set of genes is performed.  When using method 'corrected', single \
#' cell scores for a give gene set that have been "corrected" using 100X genes with similar \
#' expression levels.

#' @param cds Input cell_data_set object.
#' @param marker_set Vector of genes in the gene_metadata DataFrame (fData) that can be found in the column 'fData_col'
#' @param name Name given to the geneset
#' @param cell_size size of point on plot
#' @param fData_col Character string denoting the gene_metadata DataFrame (fData) column that contains the genes in marker_set1.  Default = 'gene_short_name'
#' @return Plot
#' @importFrom Matrix colSums
#' @importFrom Matrix t
#' @export
#' @references Puram, S. V. et al. Single-Cell Transcriptomic Analysis of Primary and 
#' Metastatic Tumor Ecosystems in Head and Neck Cancer. Cell 171, 1611.e1–1611.e24 (2017).

plot_geneset<-function(cds, marker_set, name, fData_col="gene_short_name", method=c("totals", "corrected"), reduction_method="UMAP", cell_size=0.5){
  assertthat::assert_that(
    tryCatch(expr = ifelse(match.arg(method) == "",TRUE, TRUE),
             error = function(e) FALSE),
    msg = "method must be one of 'totals' or 'corrected'")
  method <- match.arg(method)
  if(method=="totals") pData(cds)[[name]]<-estimate_score(cds, marker_set, fData_col=fData_col)
  if(method=="corrected") pData(cds)[[name]]<-estimate_corrected_score(cds, marker_set, fData_col=fData_col)
  nc<-nchar(name)
  if(nc>50){fontsize<-10}else{fontsize=14}
  switch(method, totals={loca="log(sums)"}, 
         corrected={loca="log(corr.)"})
  plot_cells(cds, color_cells_by = name, label_cell_groups = F, cell_size = cell_size, reduction_method = reduction_method)+ 
    #theme(legend.position="top", legend.title = element_blank())+
    theme(legend.position="top")+
    #ggtitle(paste0(name, ": ", loca))+ 
    ggtitle(name)+  
    theme(plot.title = element_text(size = fontsize, face = "bold"), legend.text = element_text(size=9, angle = 90, vjust=0.5, hjust=0.3))+
    labs(color = loca)+
    scale_color_gradientn(colors=c( "darkblue","skyblue", "white", "red", "darkred"))
}

monocle_theme_opts <- function()
{
  theme(strip.background = element_rect(colour = 'white', fill = 'white')) +
    theme(panel.border = element_blank()) +
    theme(axis.line.x = element_line(size=0.25, color="black")) +
    theme(axis.line.y = element_line(size=0.25, color="black")) +
    theme(panel.grid.minor.x = element_blank(),
          panel.grid.minor.y = element_blank()) +
    theme(panel.grid.major.x = element_blank(),
          panel.grid.major.y = element_blank()) +
    theme(panel.background = element_rect(fill='white')) +
    theme(legend.key=element_blank())
}


