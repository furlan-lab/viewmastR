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
static float nb_accuracy(const array &predicted, const array &target) {
  return 100 * count<float>(predicted == target) / target.elements();
}

static void naive_bayes_train(float *priors, array &mu, array &sig2,
                       const array &train_feats, const array &train_classes,
                       int num_classes) {
  const int feat_len    = train_feats.dims(0);
  const int num_samples = train_classes.elements();
  // Get mean and variance from trianing data
  mu   = constant(0, feat_len, num_classes);
  sig2 = constant(0, feat_len, num_classes);
  // std::cerr << "Mu:" << std::endl;
  // af_print(mu);
  // std::cerr << "Sig2:" << std::endl;
  // af_print(sig2);
  for (int ii = 0; ii < num_classes; ii++) {
    array idx            = where(train_classes == ii);
    array train_feats_ii = lookup(train_feats, idx, 1);
    mu(span, ii) = mean(train_feats_ii, 1);
    // Some pixels are always 0. Add a small variance.
    sig2(span, ii) = var(train_feats_ii, AF_VARIANCE_SAMPLE, 1) + 0.01;
    // Calculate priors
    priors[ii] = (float)idx.elements() / (float)num_samples;
  }
  mu.eval();
  sig2.eval();
}

static array naive_bayes_predict(float *priors, const array &mu, const array &sig2,
                          const array &test_feats, int &num_classes) {
  int num_test = test_feats.dims(1);
  // Predict the probabilities for testing data
  // Using log of probabilities to reduce rounding errors
  array log_probs = constant(1, num_test, num_classes);
  for (int ii = 0; ii < num_classes; ii++) {
    // Tile the current mean and variance to the testing data size
    array Mu   = tile(mu(span, ii), 1, num_test);
    array Sig2 = tile(sig2(span, ii), 1, num_test);
    // This is the same as log of the CDF of the normal distribution
    array Df    = test_feats - Mu;
    array log_P = (-(Df * Df) / (2 * Sig2)) - log(sqrt(2 * af::Pi * Sig2));
    // Accumulate the probabilities, multiply with priors (add log of
    // priors)
    log_probs(span, ii) = log(priors[ii]) + sum(log_P).T();
  }
  // Get the location of the maximum value
  array val, idx;
  max(val, idx, log_probs, 1);
  return idx;
}

static void benchmark_nb(const array &train_feats, const array &test_feats,
                  const array &train_labels, int &num_classes) {
  array mu, sig2;
  int iter      = 25;
  float *priors = new float[num_classes];
  timer::start();
  for (int i = 0; i < iter; i++) {
    naive_bayes_train(priors, mu, sig2, train_feats, train_labels, num_classes);
  }
  af::sync();
  fprintf(stderr,"Training time: %4.4lf s\n", timer::stop() / iter);
  timer::start();
  for (int i = 0; i < iter; i++) {
    naive_bayes_predict(priors, mu, sig2, test_feats, num_classes);
  }
  af::sync();
  fprintf(stderr,"Prediction time: %4.4lf s\n", timer::stop() / iter);
  delete[] priors;
}

//' @export
// [[Rcpp::export]]
af::array naive_bayes(RcppArrayFire::typed_array<f32> train_feats,
                 RcppArrayFire::typed_array<f32> test_feats,
                 RcppArrayFire::typed_array<s32> train_labels,
                 RcppArrayFire::typed_array<s32> test_labels,
                 int num_classes,
                 RcppArrayFire::typed_array<f32> query,
                 bool verbose = false,
                 bool benchmark = false,
                 int device = 0) {
  try {
    af::setDevice(device);
    std::string info_string = af::infoString();
    if(verbose) {std::cerr << info_string;}
  } catch (af::exception &ae) { std::cerr << ae.what() << std::endl; }
//   // Get training parameters
  array mu, sig2;
  float *priors = new float[num_classes];

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
  naive_bayes_train(priors, mu, sig2, train_feats, train_labels, num_classes);
  // Predict the classes
  array res_labels = naive_bayes_predict(priors, mu, sig2, test_feats, num_classes);
  array query_labels = naive_bayes_predict(priors, mu, sig2, query, num_classes);
  if(verbose) {
    fprintf(stderr,"Training samples: %4d, Testing samples: %4d\n", train_labels.dims(0), test_labels.dims(0));
    fprintf(stderr,"Accuracy on testing  data: %2.2f\n", nb_accuracy(res_labels, test_labels));
    if(benchmark){
      benchmark_nb(train_feats, test_feats, train_labels, num_classes);
    }
  }
  return query_labels;
}




void naive_bayes_demo_run(int perc, bool verbose = false) {
  array train_images, train_labels;
  array test_images, test_labels;
  int num_train, num_test, num_classes;
  // Load mnist data
  float frac = (float)(perc) / 100.0;
  setup_mnist<false>(&num_classes, &num_train, &num_test, train_images,
                     test_images, train_labels, test_labels, frac);
  int feature_length = train_images.elements() / num_train;
  array train_feats  = moddims(train_images, feature_length, num_train);
  array test_feats   = moddims(test_images, feature_length, num_test);
  // Get training parameters
  array mu, sig2;
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
  float *priors = new float[num_classes];
  naive_bayes_train(priors, mu, sig2, train_feats, train_labels, num_classes);
  // Predict the classes
  array res_labels =
    naive_bayes_predict(priors, mu, sig2, test_feats, num_classes);
  delete[] priors;
  // Results
  fprintf(stderr,"Trainng samples: %4d, Testing samples: %4d\n", num_train, num_test);
  fprintf(stderr,"Accuracy on testing  data: %2.2f\n",
         nb_accuracy(res_labels, test_labels));
  benchmark_nb(train_feats, test_feats, train_labels, num_classes);
  // if (!console) {
  //     test_images = test_images.T();
  //     test_labels = test_labels.T();
  //     // FIXME: Crashing in mnist_common.h::classify
  //     // display_results<false>(test_images, res_labels, test_labels , 20);
  // }
}

//' @export
// [[Rcpp::export]]
void naive_bayes_demo(int perc = 80, bool verbose = true) {
  // int device   = argc > 1 ? atoi(argv[1]) : 0;
  // bool console = argc > 2 ? argv[2][0] == '-' : false;
  // int perc     = argc > 3 ? atoi(argv[3]) : 60;
  af::setDevice(0);
  std::string info_string = af::infoString();
  std::cerr << info_string;
  naive_bayes_demo_run(perc, verbose);
  // try {
  //     af::setDevice(0);
  //     af::info();
  //     naive_bayes_demo(perc);
  // } catch (af::exception &ae) { std::cerr << ae.what() << std::endl; }
}