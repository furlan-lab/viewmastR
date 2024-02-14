#' Confusion matrix
#' @description This function will generate a confusion matrix between two factors; pred (short for prediction) and gt (short for ground truth).  One 
#' may optionally supply a named vector of colors to annotate the row and column legends.
#' @param pred factor of predictions
#' @param gt factor of ground truth
#' @param cols named vector of colors
#' @return a confusion matrix plot
#' @importFrom grDevices colors
#' @importFrom caret confusionMatrix
#' @importFrom grid gpar
#' @importFrom grid grid.text
#' @import ComplexHeatmap
#' @importFrom scCustomize viridis_light_high
#' @export

confusion_matrix<-function(pred, gt, cols=NULL){
  mat<-table( pred, gt)
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
  data<-confusionMatrix(mat_full)
  pmat<-sweep(mat, MARGIN = 2, colSums(mat), "/")*100
  acc =format(as.numeric(data$overall[1])*100, digits=4)
  column_ha = HeatmapAnnotation(
    
    labels = colnames(mat),
    col = list(labels=cols),
    na_col = "black", show_legend = F
  )
  row_ha = rowAnnotation(
    
    labels = rownames(mat),
    col = list(labels=cols),
    na_col = "black"
  )
  Heatmap(pmat, col = viridis_light_high, cluster_rows = F, cluster_columns = F, 
                          row_names_side = "left", row_title = "Predicted Label", column_title = "True Label", 
                          name = "Percent of Column", column_title_side = "top", column_names_side = "top",
                          top_annotation = column_ha, left_annotation = row_ha,
                          heatmap_legend_param = list(
                            title = paste0("Acc. ", acc, "\nPercent of Row")), 
                          rect_gp = gpar(col = "white", lwd = 2),
                          cell_fun = function(j, i, x, y, width, height, fill){
                            if(is.na(pmat[i,j])){
                              grid.text("NA", x, y, gp = gpar(col="black", fontsize = 10))
                            }else{
                              if(pmat[i,j]>60){
                                grid.text(sprintf("%.f", mat[i, j]), x, y, gp = gpar(col="black", fontsize = 10))
                              }else{
                                grid.text(sprintf("%.f", mat[i, j]), x, y, gp = gpar(col="white", fontsize = 10))
                              }
                            }
                          })
}

#' Training data plot
#' @description This function will display a plot of data generated during viewmastR training
#' @param output_list a list returned from running viewmastR using the return_type = "list" parameter.
#' @return a plot of training data
#' @importFrom plotly plot_ly
#' @importFrom plotly layout
#' @importFrom plotly subplot
#' @importFrom magrittr "%>%"
#' @export

plot_training_data<-function(output_list) {
accuracy<-rbind(
  data.frame(epoch=1:length(output_list$training_output$history$train_acc), 
             metric=as.numeric(format(output_list$training_output$history$train_acc*100, digits=5)), 
             label="train_accuracy"),
  data.frame(epoch=1:length(output_list$training_output$history$test_acc), 
             metric=as.numeric(format(output_list$training_output$history$test_acc*100, digits=5)), 
             label="validation_accuracy"))
loss<-rbind(
  data.frame(epoch=1:length(output_list$training_output$history$test_loss), 
             metric=as.numeric(format(output_list$training_output$history$train_loss, digits=5)),
             label="train_loss"),
  data.frame(epoch=1:length(output_list$training_output$history$test_loss), 
             metric=as.numeric(format(output_list$training_output$history$test_loss, digits=5)),
             label="validation_loss"))

fig1 <- plot_ly(x = accuracy$epoch, y =accuracy$metric, split = accuracy$label, type = 'scatter', mode = 'lines+markers', 
                marker = list(line = list(width = 3))) %>%
  layout(plot_bgcolor='#e5ecf6', 
                 xaxis = list( 
                   title = 'Epoch',
                   zerolinecolor = '#ffff', 
                   zerolinewidth = 2, 
                   gridcolor = 'ffff'), 
                 yaxis = list(
                   title = 'Accuracy (%)',
                   zerolinecolor = '#ffff', 
                   zerolinewidth = 2, 
                   gridcolor = 'ffff')) 
fig2 <- plot_ly(x = loss$epoch, y =loss$metric, split = loss$label, type = 'scatter', mode = 'lines+markers', 
                        marker = list(line = list(width = 3))) %>%
  layout(plot_bgcolor='#e5ecf6', 
                 xaxis = list( 
                   title = 'Epoch',
                   zerolinecolor = '#ffff', 
                   zerolinewidth = 2, 
                   gridcolor = 'ffff'), 
                 yaxis = list(
                   title = 'Loss',
                   zerolinecolor = '#ffff', 
                   zerolinewidth = 2, 
                   gridcolor = 'ffff')) 

fig <- subplot(fig1, fig2, nrows = 2)
fig


# highcharter::hw_grid(ncol = 1,rowheight = 280,
#                      hchart(
#                        tibble::tibble(accuracy),
#                        "line",
#                        hcaes(x = epoch , y = metric, group = label),
#                        color = c(pals::glasbey(2))
#                      ) |> 
#                        hc_chart(
#                          backgroundColor = list(
#                            linearGradient = c(0, 0, 500, 500),
#                            stops = list(
#                              list(0, 'rgb(255, 255, 255)'),
#                              list(1, 'rgb(170, 230, 255)')
#                            )
#                          )
#                        ),
#                      hchart(
#                        tibble::tibble(loss),
#                        "line",
#                        hcaes(x = epoch , y = metric, group = label),
#                        color = c(pals::glasbey(2))
#                      ) |> 
#                        hc_chart(
#                          backgroundColor = list(
#                            linearGradient = c(0, 0, 500, 500),
#                            stops = list(
#                              list(0, 'rgb(255, 255, 255)'),
#                              list(1, 'rgb(170, 230, 255)')
#                            )
#                          )
#                        )
# ) %>% htmltools::browsable()
}
