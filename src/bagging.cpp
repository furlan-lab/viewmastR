// we only include RcppArrayFire.h which pulls Rcpp.h in for us
#include "RcppArrayFire.h"
#include <arrayfire.h>
#include <math.h>
#include <stdio.h>
#include <af/util.h>
#include <string>
#include <vector>
#include "mnist_common.h"
// via the depends attribute we tell Rcpp to create hooks for
// RcppFire so that the build process will know what to do
//
// [[Rcpp::depends(RcppArrayFire)]]

// RcppArrayFire needs C++11
// add the following comment when you export your
// C++ function to R via Rcpp::SourceCpp()
// [[Rcpp::plugins(cpp11)]]

// simple example of creating two matrices and
// returning the result of an operation on them
//
// via the exports attribute we tell Rcpp to make this function
// available from R
//

using namespace af;


// Get accuracy of the predicted results
static float accuracy(const array &predicted, const array &target) {
  return 100 * count<float>(predicted == target) / target.elements();
}

// Calculate all the distances from testing set to training set
static array distance(array train, array test) {
  const int feat_len  = train.dims(1);
  const int num_train = train.dims(0);
  const int num_test  = test.dims(0);
  array dist          = constant(0, num_train, num_test);
  // Iterate over each attribute
  for (int ii = 0; ii < feat_len; ii++) {
    // Get a attribute vectors
    array train_i = train(span, ii);
    array test_i  = test(span, ii).T();
    // Tile the vectors to generate matrices
    array train_tiled = tile(train_i, 1, num_test);
    array test_tiled  = tile(test_i, num_train, 1);
    // Add the distance for this attribute
    dist = dist + abs(train_tiled - test_tiled);
    dist.eval();  // Necessary to free up train_i, test_i
  }
  return dist;
}

static array knn(array &train_feats, array &test_feats, array &train_labels) {
  // Find distances between training and testing sets
  array dist = distance(train_feats, test_feats);
  // Find the neighbor producing the minimum distance
  array val, idx;
  min(val, idx, dist);
  // Return the labels
  return train_labels(idx);
}

static array bagging(array &train_feats, array &test_feats, array &train_labels,
              int num_classes, int num_models, int sample_size) {
  int num_train = train_feats.dims(0);
  int num_test  = test_feats.dims(0);
  array idx        = floor(randu(sample_size, num_models) * num_train);
  array labels_all = constant(0, num_test, num_classes);
  array off        = seq(num_test);
  for (int i = 0; i < num_models; i++) {
    array ii = idx(span, i);
    array train_feats_ii  = lookup(train_feats, ii, 0);
    array train_labels_ii = train_labels(ii);
    // Get the predicted results
    array labels_ii = knn(train_feats_ii, test_feats, train_labels_ii);
    array lidx      = labels_ii * num_test + off;
    labels_all(lidx) = labels_all(lidx) + 1;
  }
  array val, labels;
  max(val, labels, labels_all, 1);
  return labels;
}


//' @export
// [[Rcpp::export]]
void bagging_demo(int perc = 50, bool verbose = true) {
  array train_images, train_labels;
  array test_images, test_labels;
  int num_train, num_test, num_classes;
  // Load mnist data
  float frac = (float)(perc) / 100.0;
  setup_mnist<false>(&num_classes, &num_train, &num_test, train_images,
                     test_images, train_labels, test_labels, frac);
  int feature_length = train_images.elements() / num_train;
  array train_feats  = moddims(train_images, feature_length, num_train).T();
  array test_feats   = moddims(test_images, feature_length, num_test).T();
  int num_models  = 10;
  int sample_size = 1000;
  if(verbose){
    std::cerr << "Train feature dims:" << std::endl;
    std::cerr << train_feats.dims() << std::endl;
    std::cerr << "Test feature dims:" << std::endl;
    std::cerr << test_feats.dims() << std::endl;
    std::cerr << "Train labels dims:" << std::endl;
    std::cerr << train_labels.dims() << std::endl;
    std::cerr << "Test labels dims:" << std::endl;
    std::cerr << test_labels.dims() << std::endl;
    std::cerr << "Num classes:" << std::endl;
    std::cerr << num_classes << std::endl;
  }
  timer::start();
  // Get the predicted results
  array res_labels = bagging(train_feats, test_feats, train_labels,
                             num_classes, num_models, sample_size);
  double test_time = timer::stop();
  // Results
  if(verbose) {
    printf("Accuracy on testing  data: %2.2f\n",
           accuracy(res_labels, test_labels));
    printf("Prediction time: %4.4f\n", test_time);
  }
  // if (false && !console) {
  //   display_results<false>(test_images, res_labels, test_labels.T(), 20);
  // }
}

//' @export
// [[Rcpp::export]]
af::array bagging(RcppArrayFire::typed_array<f32> train_feats,
                      RcppArrayFire::typed_array<f32> test_feats,
                      RcppArrayFire::typed_array<s32> train_labels,
                      RcppArrayFire::typed_array<s32> test_labels,
                      int num_classes,
                      RcppArrayFire::typed_array<f32> query,
                      bool verbose = true,
                      bool benchmark = false,
                      int num_models  = 10,
                      int sample_size = 1000,
                      int device = 0) {
  try {
    af::setDevice(device);
    std::string info_string = af::infoString();
    if(verbose) {std::cerr << info_string;}
  } catch (af::exception &ae) { std::cerr << ae.what() << std::endl; }
  
  train_feats = train_feats.T();
  test_feats  = test_feats.T();
  query= query.T();
  //   // Get training parameters
  if(verbose){
    std::cerr << "Train feature dims:" << std::endl;
    std::cerr << train_feats.dims() << std::endl;
    std::cerr << "Test feature dims:" << std::endl;
    std::cerr << test_feats.dims() << std::endl;
    std::cerr << "Train labels dims:" << std::endl;
    std::cerr << train_labels.dims() << std::endl;
    std::cerr << "Test labels dims:" << std::endl;
    std::cerr << test_labels.dims() << std::endl;
    std::cerr << "Num classes:" << std::endl;
    std::cerr << num_classes << std::endl;
  }
  timer::start();
  // Get the predicted results
  array res_labels = bagging(train_feats, test_feats, train_labels,
                             num_classes, num_models, sample_size);
  double test_time = timer::stop();
  // Results
  if(verbose) {
    fprintf(stderr, "Accuracy on testing  data: %2.2f\n",
           accuracy(res_labels, test_labels));
    fprintf(stderr, "Prediction time: %4.4f\n", test_time);
  }
  af::array query_labels = bagging(train_feats, query, train_labels,
                                   num_classes, num_models, sample_size);
  return query_labels;
}
