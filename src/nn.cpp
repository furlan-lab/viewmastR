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

using std::vector;


static std::string toStr(const dtype dt) {
  switch (dt) {
  case f32: return "f32";
  case f16: return "f16";
  default: return "N/A";
  }
}

static float accuracy(const array &predicted, const array &target) {
  array val, plabels, tlabels;
  max(val, tlabels, target, 1);
  max(val, plabels, predicted, 1);
  return 100 * count<float>(plabels == tlabels) / tlabels.elements();
}

// Derivative of the activation function
static array deriv(const array &out) { return out * (1 - out); }
// Cost function
static double error(const array &out, const array &pred) {
  array dif = (out - pred);
  return sqrt((double)(sum<float>(dif * dif)));
}

class ann {
private:
  int num_layers;
  vector<array> weights;
  dtype datatype;
  // Add bias input to the output from previous layer
  array add_bias(const array &in);
  vector<array> forward_propagate(const array &input, bool relu_activation);
  void back_propagate(const vector<array> signal, const array &pred,
                      const double &alpha);
public:
  // Create a network with given parameters
  ann(vector<int> layers, double range, dtype dt = f32, bool verbose = false, bool relu_activation = false);
  // Output after single pass of forward propagation
  array predict(const array &input, bool relu_activation);
  array relu (const array &input);
  // Method to train the neural net
  double train(const array &input, const array &target, double alpha = 1.0,
               int max_epochs = 300, int batch_size = 100,
               double maxerr = 1.0, bool verbose = false, bool relu_activation = false);
};
array ann::add_bias(const array &in) {
  // Bias input is added on top of given input
  return join(1, constant(1, in.dims(0), 1, datatype), in);
}
vector<array> ann::forward_propagate(const array &input, bool relu_activation = false) {
  // Get activations at each layer
  vector<array> signal(num_layers);
  signal[0] = input;
  for (int i = 0; i < num_layers - 1; i++) {
    array in      = add_bias(signal[i]);
    array out     = matmul(in, weights[i]);
    // signal[i + 1] = sigmoid(out);
    if(i < num_layers - 1 && relu_activation){
      signal[i + 1] = relu(out);
    } else{
      signal[i + 1] = sigmoid(out);
    }

  }
  return signal;
}

void ann::back_propagate(const vector<array> signal, const array &target,
                         const double &alpha) {
  // Get error for output layer
  array out = signal[num_layers - 1];
  array err = (out - target);
  int m = target.dims(0);
  for (int i = num_layers - 2; i >= 0; i--) {
    array in    = add_bias(signal[i]);
    array delta = (deriv(out) * err).T();
    // Adjust weights
    array tg   = alpha * matmul(delta, in);
    array grad = -(tg) / m;
    weights[i] += grad.T();
    // Input to current layer is output of previous
    out = signal[i];
    err = matmulTT(delta, weights[i]);
    // Remove the error of bias and propagate backward
    err = err(span, seq(1, out.dims(1)));
  }
}

ann::ann(vector<int> layers, double range, dtype dt, bool verbose, bool relu_activation)
  : num_layers(layers.size()), weights(layers.size() - 1), datatype(dt) {
  if(verbose){
    std::cerr
    << "Initializing weights using a random uniformly distribution between "
    << -range / 2 << " and " << range / 2 << " at precision "
    << toStr(datatype) << std::endl;
  }
  for (int i = 0; i < num_layers - 1; i++) {
    weights[i] = range * randu(layers[i] + 1, layers[i + 1]) - range / 2;
    if (datatype != f32) weights[i] = weights[i].as(datatype);
  }
}


array ann::relu(const array &input) {
  return( max(input, 0.0) );
}

array ann::predict(const array &input, bool relu_activation) {
  vector<array> signal = forward_propagate(input, relu_activation);
  array out            = signal[num_layers - 1];
  return out;
}

double ann::train(const array &input, const array &target, double alpha,
                  int max_epochs, int batch_size, double maxerr, bool verbose, bool relu_activation) {
  const int num_samples = input.dims(0);
  const int num_batches = num_samples / batch_size;
  double err = 0;
  // Training the entire network
  for (int i = 0; i < max_epochs; i++) {
    for (int j = 0; j < num_batches - 1; j++) {
      int st = j * batch_size;
      int en = st + batch_size - 1;
      array x = input(seq(st, en), span);
      array y = target(seq(st, en), span);
      // Propagate the inputs forward
      vector<array> signals = forward_propagate(x, relu_activation);
      array out             = signals[num_layers - 1];
      // Propagate the error backward
      back_propagate(signals, y, alpha);
    }
    // Validate with last batch
    int st    = (num_batches - 1) * batch_size;
    int en    = num_samples - 1;
    array out = predict(input(seq(st, en), span), relu_activation);
    err       = error(out, target(seq(st, en), span));
    // Check if convergence criteria has been met
    if (err < maxerr) {
      if(verbose){fprintf(stderr,"Converged on Epoch: %4d\n", i + 1);}
      return err;
    }
    if (verbose) {
      if ((i + 1) % 10 == 0)
        fprintf(stderr,"Epoch: %4d, Error: %0.4f\n", i + 1, err);
    }
  }
  return err;
}


static int ann_demo_run(std::string lib_path, int perc, const dtype dt, bool verbose = false, bool benchmark = false, bool relu_activation = false) {
  fprintf(stderr,"** ArrayFire ANN Demo **\n");
  array train_images, test_images;
  array train_target, test_target;
  int num_classes, num_train, num_test;
  // Load mnist data
  float frac = (float)(perc) / 100.0;
  setup_mnist<true>(&num_classes, &num_train, &num_test, train_images,
                    test_images, train_target, test_target, frac, lib_path);
  if (dt != f32) {
    train_images = train_images.as(dt);
    test_images  = test_images.as(dt);
    train_target = train_target.as(dt);
  }
  int feature_size = train_images.elements() / num_train;
  // Reshape images into feature vectors
  array train_feats = moddims(train_images, feature_size, num_train).T();
  array test_feats  = moddims(test_images, feature_size, num_test).T();
  train_target = train_target.T();
  test_target  = test_target.T();
  if(verbose){
    std::cerr << "Train feature dims:" << std::endl;
    std::cerr << train_feats.dims() << std::endl;
    std::cerr << "Test feature dims:" << std::endl;
    std::cerr << test_feats.dims() << std::endl;
    std::cerr << "Train labels dims:" << std::endl;
    std::cerr << train_target.dims() << std::endl;
    std::cerr << "Test labels dims:" << std::endl;
    std::cerr << test_target.dims() << std::endl;
    std::cerr << "Num classes:" << std::endl;
    std::cerr << num_classes << std::endl;
  }
  // Network parameters
  vector<int> layers;
  layers.push_back(train_feats.dims(1));
  layers.push_back(100);
  layers.push_back(50);
  layers.push_back(num_classes);
  // Create network: architecture, range, datatype
  ann network(layers, 0.05, dt, verbose);
  // Train network
  timer::start();
  network.train(train_feats, train_target,
                2.0,    // learning rate / alpha
                250,    // max epochs
                100,    // batch size
                0.5,    // max error
                true,   // verbose
                relu_activation);
  af::sync();
  double train_time = timer::stop();
  // Run the trained network and test accuracy.
  array train_output = network.predict(train_feats, relu_activation);
  array test_output  = network.predict(test_feats, relu_activation);
  // Benchmark prediction
  double test_time = 0.0;
  if(benchmark){
    af::sync();
    timer::start();
    for (int i = 0; i < 100; i++) { network.predict(test_feats, relu_activation); }
    af::sync();
    test_time = timer::stop() / 100;
  }
  fprintf(stderr,"\nTraining set:\n");
  fprintf(stderr,"Accuracy on training data: %2.2f\n",
         accuracy(train_output, train_target));
  fprintf(stderr,"\nTest set:\n");
  fprintf(stderr,"Accuracy on testing  data: %2.2f\n",
         accuracy(test_output, test_target));
  fprintf(stderr,"\nTraining time: %4.4lf s\n", train_time);
  if(benchmark){
    fprintf(stderr,"Prediction time: %4.4lf s\n\n", test_time);
  }
  // if (!console) {
  //   // Get 20 random test images.
  //   test_output = test_output.T();
  //   display_results<true>(test_images, test_output, test_target.T(), 20);
  // }
  return 0;
}


//' @export
// [[Rcpp::export]]
af::array af_nn(RcppArrayFire::typed_array<f32> train_feats,
                 RcppArrayFire::typed_array<f32> test_feats,
                 RcppArrayFire::typed_array<s32> train_target,
                 RcppArrayFire::typed_array<s32> test_target,
                 int num_classes,
                 std::vector<int> layers,
                 RcppArrayFire::typed_array<f32> query_feats,
                 bool relu_activation = false,
                 int device = 0,
                 std::string dts = "f32",
                 float learning_rate = 2.0,    // learning rate / alpha
                 int max_epochs = 250,    // max epochs
                 int batch_size = 100,    // batch size
                 float max_error = 0.5,    // max error
                 bool verbose = true,
                 bool benchmark = false) {
  if(verbose) {fprintf(stderr,"** ArrayFire ANN**\n");}
  if (device < 0 || device > 1) {
    std::cerr << "Bad device: " <<device << std::endl;
    return EXIT_FAILURE;
  }
  
  dtype dt        = f32;
  if (dts == "f16")
    dt = f16;
  else if (dts != "f32") {
    std::cerr << "Unsupported datatype " << dts << ". Supported: f32 or f16"
              << std::endl;
    return EXIT_FAILURE;
  }
  
  if (dts == "f16" && !af::isHalfAvailable(device)) {
    std::cerr << "Half not available for device " << device << std::endl;
    return EXIT_FAILURE;
  }
  try {
    af::setDevice(device);
    std::string info_string = af::infoString();
    if(verbose) {std::cerr << info_string;}
  } catch (af::exception &ae) { std::cerr << ae.what() << std::endl; }
  // Reshape images into feature vectors
  train_feats = train_feats.T();
  test_feats  = test_feats.T();
  // train_target = train_target.T();
  // test_target  = test_target.T();
  query_feats  = query_feats.T();
  if(verbose){
    std::cerr << "Train feature dims:" << std::endl;
    std::cerr << train_feats.dims() << std::endl;
    std::cerr << "Test feature dims:" << std::endl;
    std::cerr << test_feats.dims() << std::endl;
    std::cerr << "Train labels dims:" << std::endl;
    std::cerr << train_target.dims() << std::endl;
    std::cerr << "Test labels dims:" << std::endl;
    std::cerr << test_target.dims() << std::endl;
    std::cerr << "Query dims:" << std::endl;
    std::cerr << query_feats.dims() << std::endl;
    std::cerr << "Num classes:" << std::endl;
    std::cerr << num_classes << std::endl;
    // Network parameters
    // vector<int> layers;
    // layers.push_back(train_feats.dims(1));
    // layers.push_back(100);
    // layers.push_back(50);
    // layers.push_back(num_classes);
    // Create network: architecture, range, datatype
    std::cerr << "Creating network with the following layers:" << std::endl;
    for (auto i: layers)
      std::cerr << i << ' ';
    std::cerr << std::endl;
    std::cerr << "Learning Rate:" << std::endl;
    std::cerr << learning_rate << std::endl;
    std::cerr << "Max epochs:" << std::endl;
    std::cerr << max_epochs << std::endl;
    std::cerr << "Batch size:" << std::endl;
    std::cerr << batch_size << std::endl;
    std::cerr << "Max error:" << std::endl;
    std::cerr << max_error << std::endl;
  }
  ann network(layers, 0.05, dt, verbose);
  // Train network
  timer::start();
  network.train(train_feats, train_target, learning_rate,
                max_epochs, batch_size, max_error, verbose);
  af::sync();
  double train_time = timer::stop();
  // Run the trained network and test accuracy.
  array train_output = network.predict(train_feats, relu_activation);
  array test_output  = network.predict(test_feats, relu_activation);
  // array query_output  = network.predict(query_feats);
  // Benchmark prediction
  af::sync();
  array query_output  = network.predict(query_feats, relu_activation);
  double test_time = 0.0;
  if(verbose) {
    if(benchmark){
      timer::start();
      for (int i = 0; i < 100; i++) { network.predict(test_feats, relu_activation); }
      af::sync();
      test_time = timer::stop() / 100;
    }
    fprintf(stderr,"\nTraining set:\n");
    fprintf(stderr,"Accuracy on training data: %2.2f\n",
           accuracy(train_output, train_target));
    fprintf(stderr,"\nTest set:\n");
    fprintf(stderr,"Accuracy on testing  data: %2.2f\n",
           accuracy(test_output, test_target));
    fprintf(stderr,"\nTraining time: %4.4lf s\n", train_time);
    if(benchmark){
      fprintf(stderr,"Prediction time: %4.4lf s\n\n", test_time);
    }
  }
  deviceGC();
  return query_output;
}


//' @export
// [[Rcpp::export]]
int ann_demo(std::string lib_path, int device = 0, int perc = 80, std::string dts = "f32", bool verbose = true, bool benchmark = false) {
  if (device < 0 || device > 1) {
    std::cerr << "Bad device: " <<device << std::endl;
    return EXIT_FAILURE;
  }
  if (perc < 0 || perc > 100) {
    std::cerr << "Bad perc arg: " << perc << std::endl;
    return EXIT_FAILURE;
  }
  dtype dt        = f32;
  if (dts == "f16")
    dt = f16;
  else if (dts != "f32") {
    std::cerr << "Unsupported datatype " << dts << ". Supported: f32 or f16"
              << std::endl;
    return EXIT_FAILURE;
  }
  if (dts == "f16" && !af::isHalfAvailable(device)) {
    std::cerr << "Half not available for device " << device << std::endl;
    return EXIT_FAILURE;
  }
  try {
    af::setDevice(device);
    std::string info_string = af::infoString();
    std::cerr << info_string;
    return ann_demo_run(lib_path, perc, dt, verbose, benchmark);
  } catch (af::exception &ae) { std::cerr << ae.what() << std::endl; }
  return 0;
}



