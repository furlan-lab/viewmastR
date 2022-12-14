# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#' @export
bagging_demo <- function(perc = 50L, verbose = TRUE) {
    invisible(.Call('_viewmastR_bagging_demo', PACKAGE = 'viewmastR', perc, verbose))
}

#' @export
bagging <- function(train_feats, test_feats, train_labels, test_labels, num_classes, query, verbose = TRUE, benchmark = FALSE, num_models = 10L, sample_size = 1000L, device = 0L) {
    .Call('_viewmastR_bagging', PACKAGE = 'viewmastR', train_feats, test_feats, train_labels, test_labels, num_classes, query, verbose, benchmark, num_models, sample_size, device)
}

#' @export
af_dbn <- function(train_feats, test_feats, train_target, test_target, num_classes, query_feats, device = 0L, dts = "f32", rbm_learning_rate = 0.2, nn_learning_rate = 4.0, rbm_epochs = 15L, nn_epochs = 250L, batch_size = 100L, max_error = 0.5, verbose = TRUE, benchmark = FALSE) {
    .Call('_viewmastR_af_dbn', PACKAGE = 'viewmastR', train_feats, test_feats, train_target, test_target, num_classes, query_feats, device, dts, rbm_learning_rate, nn_learning_rate, rbm_epochs, nn_epochs, batch_size, max_error, verbose, benchmark)
}

#' @export
dbn_demo <- function(device = 0L, perc = 80L, dts = "f32") {
    .Call('_viewmastR_dbn_demo', PACKAGE = 'viewmastR', device, perc, dts)
}

#' @export
lr <- function(train_feats, test_feats, train_targets, test_targets, num_classes, query, learning_rate = 2.0, verbose = FALSE, benchmark = FALSE, device = 0L) {
    .Call('_viewmastR_lr', PACKAGE = 'viewmastR', train_feats, test_feats, train_targets, test_targets, num_classes, query, learning_rate, verbose, benchmark, device)
}

#' @export
lr_demo <- function(perc = 80L, verbose = TRUE) {
    invisible(.Call('_viewmastR_lr_demo', PACKAGE = 'viewmastR', perc, verbose))
}

#' @export
naive_bayes <- function(train_feats, test_feats, train_labels, test_labels, num_classes, query, verbose = FALSE, benchmark = FALSE, device = 0L) {
    .Call('_viewmastR_naive_bayes', PACKAGE = 'viewmastR', train_feats, test_feats, train_labels, test_labels, num_classes, query, verbose, benchmark, device)
}

#' @export
naive_bayes_demo <- function(perc = 80L, verbose = TRUE) {
    invisible(.Call('_viewmastR_naive_bayes_demo', PACKAGE = 'viewmastR', perc, verbose))
}

#' @export
test_backends <- function() {
    .Call('_viewmastR_test_backends', PACKAGE = 'viewmastR')
}

#' @export
af_nn <- function(train_feats, test_feats, train_target, test_target, num_classes, layers, query_feats, relu_activation = FALSE, device = 0L, dts = "f32", learning_rate = 2.0, max_epochs = 250L, batch_size = 100L, max_error = 0.5, verbose = TRUE, benchmark = FALSE) {
    .Call('_viewmastR_af_nn', PACKAGE = 'viewmastR', train_feats, test_feats, train_target, test_target, num_classes, layers, query_feats, relu_activation, device, dts, learning_rate, max_epochs, batch_size, max_error, verbose, benchmark)
}

#' @export
ann_demo <- function(device = 0L, perc = 80L, dts = "f32", verbose = TRUE, benchmark = FALSE) {
    .Call('_viewmastR_ann_demo', PACKAGE = 'viewmastR', device, perc, dts, verbose, benchmark)
}

#' @export
perceptron <- function(train_feats, test_feats, train_targets, test_targets, num_classes, query, verbose = FALSE, device = 0L) {
    .Call('_viewmastR_perceptron', PACKAGE = 'viewmastR', train_feats, test_feats, train_targets, test_targets, num_classes, query, verbose, device)
}

#' @export
perceptron_demo <- function(device = 0L, perc = 80L, verbose = TRUE) {
    invisible(.Call('_viewmastR_perceptron_demo', PACKAGE = 'viewmastR', device, perc, verbose))
}

#' @export
smr <- function(train_feats, test_feats, train_targets, test_targets, num_classes, query, lambda = 1.0, learning_rate = 2.0, iterations = 1000L, batch_size = 100L, max_error = 0.5, verbose = FALSE, benchmark = FALSE, device = 0L) {
    .Call('_viewmastR_smr', PACKAGE = 'viewmastR', train_feats, test_feats, train_targets, test_targets, num_classes, query, lambda, learning_rate, iterations, batch_size, max_error, verbose, benchmark, device)
}

#' @export
smr_demo <- function(perc = 80L, verbose = TRUE) {
    invisible(.Call('_viewmastR_smr_demo', PACKAGE = 'viewmastR', perc, verbose))
}

computeSparseRowVariances <- function(j, val, rm, n) {
    .Call('_viewmastR_computeSparseRowVariances', PACKAGE = 'viewmastR', j, val, rm, n)
}

#' @export
get_sigmoid <- function(input) {
    .Call('_viewmastR_get_sigmoid', PACKAGE = 'viewmastR', input)
}

#' @export
get_relu <- function(input) {
    .Call('_viewmastR_get_relu', PACKAGE = 'viewmastR', input)
}

#' @export
get_mnist <- function(perc = 80L, verbose = TRUE) {
    .Call('_viewmastR_get_mnist', PACKAGE = 'viewmastR', perc, verbose)
}

