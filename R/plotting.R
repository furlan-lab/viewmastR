#' @title Plot Training and Validation Metrics from viewmastR Output
#'
#' @description
#' Displays plots of training and validation accuracy and loss over epochs from the output of the \code{viewmastR} training process.
#'
#' @param output_list A list returned from running \code{viewmastR} with the parameter \code{return_type = "list"}.
#'
#' @return A \code{plotly} object showing the training and validation accuracy and loss over epochs.
#'
#' @details
#' This function extracts training and validation accuracy and loss from the \code{output_list} and creates interactive plots using \code{plotly}.
#'
#' @importFrom plotly plot_ly layout subplot
#' @importFrom magrittr %>%
#' @importFrom reshape2 melt
#' @export
#'
#' @examples
#' \dontrun{
#' # Assuming 'output' is the result from viewmastR with return_type = "list"
#' plot_training_data(output)
#' }
plot_training_data <- function(output_list) {
  # Extract training history
  history <- output_list$training_output$history
  epochs <- seq_along(history$train_acc)
  
  # Create data frames for accuracy and loss
  accuracy_df <- data.frame(
    Epoch = epochs,
    Train = history$train_acc * 100,
    Validation = history$test_acc * 100
  )
  loss_df <- data.frame(
    Epoch = epochs,
    Train = history$train_loss,
    Validation = history$test_loss
  )
  
  # Reshape data for plotting
  accuracy_long <- reshape2::melt(accuracy_df, id.vars = "Epoch", variable.name = "Set", value.name = "Accuracy")
  loss_long <- reshape2::melt(loss_df, id.vars = "Epoch", variable.name = "Set", value.name = "Loss")
  
  # Create accuracy plot
  fig_acc <- plot_ly(
    data = accuracy_long,
    x = ~Epoch,
    y = ~Accuracy,
    color = ~Set,
    type = 'scatter',
    mode = 'lines+markers',
    colors = c("Train" = "blue", "Validation" = "orange")
  ) %>%
    layout(
      title = "Training and Validation Accuracy",
      xaxis = list(title = 'Epoch'),
      yaxis = list(title = 'Accuracy (%)')
    )
  
  # Create loss plot
  fig_loss <- plot_ly(
    data = loss_long,
    x = ~Epoch,
    y = ~Loss,
    color = ~Set,
    type = 'scatter',
    mode = 'lines+markers',
    colors = c("Train" = "blue", "Validation" = "orange")
  ) %>%
    layout(
      title = "Training and Validation Loss",
      xaxis = list(title = 'Epoch'),
      yaxis = list(title = 'Loss')
    )
  
  # Combine plots
  fig <- subplot(fig_acc, fig_loss, nrows = 2, shareX = TRUE, titleY = TRUE)
  fig
}


#' @title Plot Confusion Matrix
#'
#' @description
#' Generates a confusion matrix plot between two factors: predictions and ground truth. Optionally, a named vector of colors can be provided to annotate the row and column labels.
#'
#' @param pred A factor of predicted labels.
#' @param gt A factor of ground truth labels.
#' @param cols An optional named vector of colors for the labels. Names should correspond to the levels of \code{pred} and \code{gt}.
#'
#' @return A confusion matrix plot generated using the \code{ComplexHeatmap} package.
#'
#' @details
#' This function creates a confusion matrix heatmap showing the percentage of each true label predicted as each predicted label. It includes annotations for the labels and displays the counts within each cell. The overall accuracy is calculated and displayed in the legend.
#'
#' @importFrom grDevices colors
#' @importFrom caret confusionMatrix
#' @importFrom grid gpar grid.text
#' @importFrom ComplexHeatmap Heatmap HeatmapAnnotation rowAnnotation
#' @importFrom viridis viridis
#' @export
#'
#' @examples
#' \dontrun{
#' pred <- factor(sample(c("A", "B", "C"), 100, replace = TRUE))
#' gt <- factor(sample(c("A", "B", "C"), 100, replace = TRUE))
#' confusion_matrix(pred, gt)
#' }
confusion_matrix <- function(pred, gt, cols = NULL) {
  # Ensure inputs are factors
  if (!is.factor(pred)) pred <- factor(pred)
  if (!is.factor(gt)) gt <- factor(gt)
  
  # Combine levels to ensure all are represented
  all_labels <- union(levels(pred), levels(gt))
  pred <- factor(pred, levels = all_labels)
  gt <- factor(gt, levels = all_labels)
  
  # Create confusion matrix
  mat <- table(pred, gt)
  
  # Handle colors
  if (is.null(cols)) {
    available_colors <- colors()[grep('gr(a|e)y', colors(), invert = TRUE)]
    set.seed(123)  # For reproducibility
    cols <- setNames(sample(available_colors, length(all_labels), replace = FALSE), all_labels)
  } else {
    # Check that cols has names matching all_labels
    if (!all(all_labels %in% names(cols))) {
      stop("The 'cols' vector must have names matching all levels of 'pred' and 'gt'.")
    }
  }
  
  # Calculate overall accuracy using caret::confusionMatrix
  cm <- confusionMatrix(mat)
  acc <- formatC(cm$overall['Accuracy'] * 100, format = "f", digits = 2)
  
  # Calculate percentages for heatmap
  pmat <- prop.table(mat, margin = 2) * 100  # Percentage of each column (true label)
  
  # Create annotations
  column_ha <- HeatmapAnnotation(
    labels = colnames(mat),
    col = list(labels = cols),
    show_annotation_name = FALSE,
    show_legend = FALSE
  )
  row_ha <- rowAnnotation(
    labels = rownames(mat),
    col = list(labels = cols),
    show_annotation_name = FALSE,
    show_legend = FALSE
  )
  
  # Create heatmap
  heatmap <- Heatmap(
    pmat,
    name = "Percentage",
    col = viridis::viridis(100),
    cluster_rows = FALSE,
    cluster_columns = FALSE,
    row_names_side = "left",
    row_title = "Predicted Label",
    column_title = "True Label",
    top_annotation = column_ha,
    left_annotation = row_ha,
    heatmap_legend_param = list(
      title = paste0("Accuracy: ", acc, "%\nPercentage of True Label")
    ),
    rect_gp = gpar(col = "white", lwd = 1),
    cell_fun = function(j, i, x, y, width, height, fill) {
      grid.text(
        sprintf("%d", mat[i, j]),
        x,
        y,
        gp = gpar(
          col = ifelse(pmat[i, j] > 50, "black", "white"),
          fontsize = 10,
          fontface = "bold"
        )
      )
    }
  )
  
  # Draw heatmap
  draw(heatmap)
}
