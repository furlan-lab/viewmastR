// we only include RcppArrayFire.h which pulls Rcpp.h in for us
#include "RcppArrayFire.h"
#include <arrayfire.h>
#include <math.h>
#include <stdio.h>
#include <af/util.h>
#include <string>
#include <vector>
#include "mnist_common.h"
#include <chrono>

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
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

// Get accuracy of the predicted results
static float accuracy(const array &predicted, const array &target) {
  array val, plabels, tlabels;
  max(val, tlabels, target, 1);
  max(val, plabels, predicted, 1);
  return 100 * count<float>(plabels == tlabels) / tlabels.elements();
}

static float abserr(const array &predicted, const array &target) {
  return 100 * sum<float>(abs(predicted - target)) / predicted.elements();
}

static array divide(const array &a, const array &b) { return a / b; }
// Predict based on given parameters

static array predict(const array &X, const array &Weights) {
  array Z   = matmul(X, Weights);
  array EZ  = exp(Z);
  array nrm = sum(EZ, 1);
  return batchFunc(EZ, nrm, divide);
}

static void cost(array &J, array &dJ, const array &Weights, const array &X,
          const array &Y, double lambda = 1.0) {
  // Number of samples
  int m = Y.dims(0);
  // Make the lambda corresponding to Weights(0) == 0
  array lambdat = constant(lambda, Weights.dims());
  // No regularization for bias weights
  lambdat(0, span) = 0;
  // Get the prediction
  array H = predict(X, Weights);
  // Cost of misprediction
  array Jerr = -sum(Y * log(H));
  // Regularization cost
  array Jreg = 0.5 * sum(lambdat * Weights * Weights);
  // Total cost
  J = (Jerr + Jreg) / m;
  // Find the gradient of cost
  array D = (H - Y);
  dJ      = (matmulTN(X, D) + lambdat * Weights) / m;
}

static array train(const array &X, const array &Y, double alpha = 0.1,
            double lambda = 1.0, double maxerr = 0.01, int maxiter = 1000,
            bool verbose = false) {
  // Initialize parameters to 0
  array Weights = constant(0, X.dims(1), Y.dims(1));
  array J, dJ;
  float err = 0;
  for (int i = 0; i < maxiter; i++) {
    // Get the cost and gradient
    cost(J, dJ, Weights, X, Y, lambda);
    err = max<float>(abs(J));
    if (err < maxerr) {
      if(verbose){
        fprintf(stderr, "Iteration %4d Err: %.4f\n", i + 1, err);
        fprintf(stderr, "Training converged\n");
      }
      return Weights;
    }
    if (verbose && ((i + 1) % 10 == 0)) {
      fprintf(stderr, "Iteration %4d Err: %.4f\n", i + 1, err);
    }
    // Update the parameters via gradient descent
    Weights = Weights - alpha * dJ;
  }
  if(verbose){fprintf(stderr, "Training stopped after %d iterations\n", maxiter);}
  return Weights;
}

static void benchmark_softmax_regression(const array &train_feats,
                                  const array &train_targets,
                                  const array &test_feats) {
  timer::start();
  array Weights = train(train_feats, train_targets, 0.1, 1.0, 0.01, 1000);
  af::sync();
  fprintf(stderr, "Training time: %4.4lf s\n", timer::stop());
  timer::start();
  const int iter = 100;
  for (int i = 0; i < iter; i++) {
    array test_outputs = predict(test_feats, Weights);
    test_outputs.eval();
  }
  af::sync();
  fprintf(stderr, "Prediction time: %4.4lf s\n", timer::stop() / iter);
}


// Demo of one vs all logistic regression
static int logit_demo_run (int perc, std::string lib_path, bool verbose = true, bool benchmark = false) {
  array train_images, train_targets;
  array test_images, test_targets;
  int num_train, num_test, num_classes;
  // Load mnist data
  float frac = (float)(perc) / 100.0;
  setup_mnist<true>(&num_classes, &num_train, &num_test, train_images,
                    test_images, train_targets, test_targets, frac, lib_path);
  // Reshape images into feature vectors
  int feature_length = train_images.elements() / num_train;
  array train_feats  = moddims(train_images, feature_length, num_train).T();
  array test_feats   = moddims(test_images, feature_length, num_test).T();
  train_targets = train_targets.T();
  test_targets  = test_targets.T();
  // Add a bias that is always 1
  train_feats = join(1, constant(1, num_train, 1), train_feats);
  test_feats  = join(1, constant(1, num_test, 1), test_feats);
  // Train logistic regression parameters
  auto t1 = high_resolution_clock::now();
  array Weights =
    train(train_feats, train_targets,
          0.1,    // learning rate (aka alpha)
          1.0,    // regularization constant (aka weight decay, aka lamdba)
          0.01,   // maximum error
          1000,   // maximum iterations
          true);  // verbose
  if(verbose){
    std::cerr << "Train feature dims:" << std::endl;
    std::cerr << train_feats.dims() << std::endl;
    std::cerr << "Test feature dims:" << std::endl;
    std::cerr << test_feats.dims() << std::endl;
    std::cerr << "Train targets dims:" << std::endl;
    std::cerr << train_targets.dims() << std::endl;
    std::cerr << "Test targets dims:" << std::endl;
    std::cerr << test_targets.dims() << std::endl;
    std::cerr << "Num classes:" << std::endl;
    std::cerr << num_classes << std::endl;
  }
  // Predict the results
  array train_outputs = predict(train_feats, Weights);
  array test_outputs  = predict(test_feats, Weights);
  auto t2 = high_resolution_clock::now();
  if(verbose){
    fprintf(stderr, "Accuracy on training data: %2.2f\n",
            accuracy(train_outputs, train_targets));
    fprintf(stderr, "Accuracy on testing data: %2.2f\n",
            accuracy(test_outputs, test_targets));
    fprintf(stderr, "Maximum error on testing data: %2.2f\n",
            abserr(test_outputs, test_targets));
  }
  if(benchmark){
    benchmark_softmax_regression(train_feats, train_targets, test_feats);
  }
  
  
  /* Getting number of milliseconds as a double. */
  duration<double, std::milli> ms_double = t2 - t1;

  std::cerr << "Training took: " << ms_double.count() << " ms\n";
  return 0;
}

//' @export
// [[Rcpp::export]]

af::array smr(RcppArrayFire::typed_array<f32> train_feats,
                 RcppArrayFire::typed_array<f32> test_feats,
                 RcppArrayFire::typed_array<s32> train_targets,
                 RcppArrayFire::typed_array<s32> test_targets,
                 int num_classes,
                 RcppArrayFire::typed_array<f32> query,
                 float lambda = 1.0,
                 float learning_rate = 2.0,    // learning rate / alpha
                 int iterations = 1000,    //iterations
                 int batch_size = 100,    // batch size
                 float max_error = 0.5,    // max error
                 bool verbose = false,
                 bool benchmark = false,
                 int device = 0) {
  try {
    af::setDevice(device);
    std::string info_string = af::infoString();
    if(verbose) {std::cerr << info_string;}
  } catch (af::exception &ae) { std::cerr << ae.what() << std::endl; }
  train_feats = train_feats.T();
  test_feats  = test_feats.T();
  // train_targets = train_targets.T();
  // test_targets  = test_targets.T();
  query  = query.T();
//   // Get training parameters
  if(verbose){
    std::cerr << "Train feature dims:" << std::endl;
    std::cerr << train_feats.dims() << std::endl;
    std::cerr << "Test feature dims:" << std::endl;
    std::cerr << test_feats.dims() << std::endl;
    std::cerr << "Train targets dims:" << std::endl;
    std::cerr << train_targets.dims() << std::endl;
    std::cerr << "Test targets dims:" << std::endl;
    std::cerr << test_targets.dims() << std::endl;
    std::cerr << "Num classes:" << std::endl;
    std::cerr << num_classes << std::endl;
    std::cerr << "Query dims:" << std::endl;
    std::cerr << query.dims()<< std::endl;
  }
  // Add a bias that is always 1
  train_feats = join(1, constant(1, train_feats.dims(0), 1), train_feats);
  test_feats  = join(1, constant(1, test_feats.dims(0), 1), test_feats);
  query  = join(1, constant(1, query.dims(0), 1), query);
  // Train logistic regression parameters
  array Weights =
    train(train_feats, train_targets,
          learning_rate,    // learning rate (aka alpha)
          lambda,    // regularization constant (aka weight decay, aka lamdba)
          max_error,   // maximum error
          iterations,   // maximum iterations
          verbose);  // verbose
  // Predict the results
  array train_outputs = predict(train_feats, Weights);
  array query_outputs = predict(query, Weights);
  array test_outputs  = predict(test_feats, Weights);
  if(verbose){
    fprintf(stderr, "Accuracy on training data: %2.2f\n",
            accuracy(train_outputs, train_targets));
    fprintf(stderr, "Accuracy on testing data: %2.2f\n",
            accuracy(test_outputs, test_targets));
    fprintf(stderr, "Maximum error on testing data: %2.2f\n",
            abserr(test_outputs, test_targets));
  }
  if(benchmark){
    benchmark_softmax_regression(train_feats, train_targets, test_feats);
  }
  return query_outputs;
}


 // [[Rcpp::export]]
void smr_demo_helper(std::string lib_path, int perc = 80, bool verbose = true) {
  // int device   = argc > 1 ? atoi(argv[1]) : 0;
  // bool console = argc > 2 ? argv[2][0] == '-' : false;
  // int perc     = argc > 3 ? atoi(argv[3]) : 60;
  af::setDevice(0);
  std::string info_string = af::infoString();
  std::cerr << info_string;
  logit_demo_run(perc, lib_path);
  // try {
  //     af::setDevice(0);
  //     af::info();
  //     naive_bayes_demo(perc);
  // } catch (af::exception &ae) { std::cerr << ae.what() << std::endl; }
}