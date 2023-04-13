// we only include RcppArrayFire.h which pulls Rcpp.h in for us
#include "RcppArrayFire.h"
#include <arrayfire.h>
#include <math.h>
#include <stdio.h>
#include <af/util.h>
#include <string>
#include <vector>
#include "mnist_common.h"
#include <algorithm>
#include <cstdio>

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

std::vector<float> input(100);

// Generate a random number between 0 and 1
// return a uniform number in [0,1].
double unifRand() { return rand() / double(RAND_MAX); }

//' @export
// [[Rcpp::export]]
void testBackend() {
  af::info();
  
  af::dim4 dims(10, 10, 1, 1);
  
  af::array A(dims, &input.front());
  af_print(A);
  
  af::array B = af::constant(0.5, dims, f32);
  af_print(B);
}

int main(int, char**) {
  std::generate(input.begin(), input.end(), unifRand);
  
  try {
    printf("Trying CPU Backend\n");
    af::setBackend(AF_BACKEND_CPU);
    testBackend();
  } catch (af::exception& e) {
    printf("Caught exception when trying CPU backend\n");
    fprintf(stderr, "%s\n", e.what());
  }
  
  try {
    printf("Trying CUDA Backend\n");
    af::setBackend(AF_BACKEND_CUDA);
    testBackend();
  } catch (af::exception& e) {
    printf("Caught exception when trying CUDA backend\n");
    fprintf(stderr, "%s\n", e.what());
  }
  
  try {
    printf("Trying OpenCL Backend\n");
    af::setBackend(AF_BACKEND_OPENCL);
    testBackend();
  } catch (af::exception& e) {
    printf("Caught exception when trying OpenCL backend\n");
    fprintf(stderr, "%s\n", e.what());
  }
  
  return 0;
}