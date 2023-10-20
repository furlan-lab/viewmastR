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

// static std::string toStr(const dtype dt) {
//   switch (dt) {
//   case f32: return "f32";
//   case f16: return "f16";
//   default: return "N/A";
//   }
// }

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

static array sigmoid_binary(const array in) {
  // Choosing "1" with probability sigmoid(in)
  return (sigmoid(in) > randu(in.dims())).as(f32);
}

class rbm {
private:
  array weights;
  array h_bias;
  array v_bias;
public:
  rbm(int v_size, int h_size)
    : weights(randu(h_size, v_size) / 100.f)
  , h_bias(constant(0, 1, h_size))
  , v_bias(constant(0, 1, v_size)) {}
  array get_weights() {
    return transpose(join(1, weights, transpose(h_bias)));
  }
  void train(const array &in, double lr, int num_epochs, int batch_size,
             bool verbose) {
    const int num_samples = in.dims(0);
    const int num_batches = num_samples / batch_size;
    for (int i = 0; i < num_epochs; i++) {
      double err = 0;
      for (int j = 0; j < num_batches - 1; j++) {
        int st  = j * batch_size;
        int en  = std::min(num_samples - 1, st + batch_size - 1);
        int num = en - st + 1;
        array v_pos = in(seq(st, en), span);
        array h_pos = sigmoid_binary(tile(h_bias, num) +
          matmulNT(v_pos, weights));
        array v_neg =
          sigmoid_binary(tile(v_bias, num) + matmul(h_pos, weights));
        array h_neg = sigmoid_binary(tile(h_bias, num) +
          matmulNT(v_neg, weights));
        array c_pos = matmulTN(h_pos, v_pos);
        array c_neg = matmulTN(h_neg, v_neg);
        array delta_w  = lr * (c_pos - c_neg) / num;
        array delta_vb = lr * sum(v_pos - v_neg) / num;
        array delta_hb = lr * sum(h_pos - h_neg) / num;
        weights += delta_w;
        v_bias += delta_vb;
        h_bias += delta_hb;
        if (verbose) { err += error(v_pos, v_neg); }
      }
      if (verbose) {fprintf(stderr, "Epoch %d: Reconstruction error: %0.4f\n", i + 1,
               err / num_batches);};
    }
  }
  array prop_up(const array &in) {
    return sigmoid(tile(h_bias, in.dims(0)) + matmulNT(in, weights));
  }
};

class dbn {
private:
  const int in_size;
  const int out_size;
  const int num_hidden;
  const int num_total;
  std::vector<array> weights;
  std::vector<int> hidden;
  array add_bias(const array &in) {
    // Bias input is added on top of given input
    return join(1, constant(1, in.dims(0), 1), in);
  }
  vector<array> forward_propagate(const array &input) {
    // Get activations at each layer
    vector<array> signal(num_total);
    signal[0] = input;
    for (int i = 0; i < num_total - 1; i++) {
      array in      = add_bias(signal[i]);
      array out     = matmul(in, weights[i]);
      signal[i + 1] = sigmoid(out);
    }
    return signal;
  }
  void back_propagate(const vector<array> signal, const array &target,
                      const double &alpha) {
    // Get error for output layer
    array out = signal[num_total - 1];
    array err = (out - target);
    int m     = target.dims(0);
    for (int i = num_total - 2; i >= 0; i--) {
      array in    = add_bias(signal[i]);
      array delta = (deriv(out) * err).T();
      // Adjust weights
      array grad = -(alpha * matmul(delta, in)) / m;
      weights[i] += grad.T();
      // Input to current layer is output of previous
      out = signal[i];
      err = matmulTT(delta, weights[i]);
      // Remove the error of bias and propagate backward
      err = err(span, seq(1, out.dims(1)));
    }
  }
public:
  dbn(const int in_sz, const int out_sz, const std::vector<int> hidden_layers)
    : in_size(in_sz)
  , out_size(out_sz)
  , num_hidden(hidden_layers.size())
  , num_total(hidden_layers.size() + 2)
  , weights(hidden_layers.size() + 1)
  , hidden(hidden_layers) {}
  void train(const array &input, const array &target, double lr_rbm = 1.0,
             double lr_nn = 1.0, const int epochs_rbm = 15,
             const int epochs_nn = 300, const int batch_size = 100,
             double maxerr = 1.0, bool verbose = false) {
    // Pre-training hidden layers
    array X = input;
    for (int i = 0; i < num_hidden; i++) {
      if (verbose) { fprintf(stderr, "Training Hidden Layer %d\n", i); }
      int visible = (i == 0) ? in_size : hidden[i - 1];
      rbm r(visible, hidden[i]);
      r.train(X, lr_rbm, epochs_rbm, batch_size, verbose);
      X          = r.prop_up(X);
      weights[i] = r.get_weights();
      if (verbose) { fprintf(stderr, "\n"); }
    }
    weights[num_hidden] =
      0.05 * randu(hidden[num_hidden - 1] + 1, out_size) - 0.0025;
    const int num_samples = input.dims(0);
    const int num_batches = num_samples / batch_size;
    // Training the entire network
    for (int i = 0; i < epochs_nn; i++) {
      for (int j = 0; j < num_batches; j++) {
        int st = j * batch_size;
        int en = std::min(num_samples - 1, st + batch_size - 1);
        array x = input(seq(st, en), span);
        array y = target(seq(st, en), span);
        // Propagate the inputs forward
        vector<array> signals = forward_propagate(x);
        array out             = signals[num_total - 1];
        // Propagate the error backward
        back_propagate(signals, y, lr_nn);
      }
      // Validate with last batch
      int st     = (num_batches - 1) * batch_size;
      int en     = num_samples - 1;
      array out  = predict(input(seq(st, en), span));
      double err = error(out, target(seq(st, en), span));
      // Check if convergence criteria has been met
      if (err < maxerr) {
        if (verbose) { fprintf(stderr,"Converged on Epoch: %4d\n", i + 1); };
        return;
      }
      if (verbose) {
        if ((i + 1) % 10 == 0)
          fprintf(stderr, "Epoch: %4d, Error: %0.4f\n", i + 1, err);
      }
    }
  }
  array predict(const array &input) {
    vector<array> signal = forward_propagate(input);
    array out            = signal[num_total - 1];
    return out;
  }
};


static int dbn_demo_run(std::string lib_path, int perc, const dtype dt, bool verbose = false) {
  if (verbose) { fprintf(stderr,"** ArrayFire DBN Demo **\n"); };
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
  layers.push_back(100);
  layers.push_back(50);
  // Create network
  dbn network(train_feats.dims(1), num_classes, layers);
  // Train network
  timer::start();
  network.train(train_feats, train_target,
                0.2,    // rbm learning rate
                4.0,    // nn learning rate
                15,     // rbm epochs
                250,    // nn epochs
                100,    // batch_size
                0.5,    // max error
                verbose);  // verbose
  af::sync();
  double train_time = timer::stop();
  // Run the trained network and test accuracy.
  array train_output = network.predict(train_feats);
  array test_output  = network.predict(test_feats);
  // Benchmark prediction

  double test_time = timer::stop() / 100;
  if (verbose){
    af::sync();
    timer::start();
    for (int i = 0; i < 100; i++) { network.predict(test_feats); }
    af::sync();
    fprintf(stderr,"\nTraining set:\n");
    fprintf(stderr,"Accuracy on training data: %2.2f\n",
           accuracy(train_output, train_target));
    fprintf(stderr,"\nTest set:\n");
    fprintf(stderr,"Accuracy on testing  data: %2.2f\n",
           accuracy(test_output, test_target));
    fprintf(stderr,"\nTraining time: %4.4lf s\n", train_time);
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
af::array af_dbn(RcppArrayFire::typed_array<f32> train_feats,
                 RcppArrayFire::typed_array<f32> test_feats,
                 RcppArrayFire::typed_array<s32> train_target,
                 RcppArrayFire::typed_array<s32> test_target,
                 int num_classes,
                 RcppArrayFire::typed_array<f32> query_feats,
                 int device = 0,
                 std::string dts = "f32",
                 float rbm_learning_rate = 0.2,    // rbm learning rate 
                 float nn_learning_rate = 4.0,    // nn learning rate 
                 int rbm_epochs = 15,    // rbm epochs
                 int nn_epochs = 250,    // nn epochs
                 int batch_size = 100,    // batch size
                 float max_error = 0.5,    // max error
                 bool verbose = true,
                 bool benchmark = false) {
  if (verbose) { fprintf(stderr,"** ArrayFire DBN**\n"); };
  if (device < 0 || device > 1) {
    std::cerr << "Bad device: " <<device << std::endl;
    return EXIT_FAILURE;
  }
  
  // dtype dt        = f32;
  // if (dts == "f16")
  //   dt = f16;
  // else if (dts != "f32") {
  //   std::cerr << "Unsupported datatype " << dts << ". Supported: f32 or f16"
  //             << std::endl;
  //   return EXIT_FAILURE;
  // }
  // 
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
  }
  // Network parameters
  vector<int> layers;
  layers.push_back(100);
  layers.push_back(50);
  // Create network
  dbn network(train_feats.dims(1), num_classes, layers);
  
  // Train network
  timer::start();
  network.train(train_feats, train_target,  
                rbm_learning_rate,
                nn_learning_rate,
                rbm_epochs,
                nn_epochs,
                batch_size,
                max_error,
                verbose);
  af::sync();
  double train_time = timer::stop();
  // Run the trained network and test accuracy.
  array train_output = network.predict(train_feats);
  array test_output  = network.predict(test_feats);
  // array query_output  = network.predict(query_feats);
  // Benchmark prediction
  double test_time = 0.0;
  if(benchmark){
    af::sync();
    timer::start();
    for (int i = 0; i < 100; i++) { network.predict(test_feats); }
    af::sync();
    test_time = timer::stop() / 100;
  }
  if (verbose) {
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
  array query_output  = network.predict(query_feats);
  return query_output;
}


//' @export
// [[Rcpp::export]]
int dbn_demo(std::string lib_path, int device = 0, int perc = 80, std::string dts = "f32") {
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
    return dbn_demo_run(lib_path, perc, dt);
  } catch (af::exception &ae) { std::cerr << ae.what() << std::endl; }
  return 0;
}



