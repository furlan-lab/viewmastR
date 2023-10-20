/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <algorithm>
#include <sstream>
#include <utility>
#include "idxio.h"
#include <string.h>

// extern std:string ASSETS_DIR;

static inline bool compare(const std::pair<float, int> l, const std::pair<float, int> r) {
    return l.first >= r.first;
}

typedef std::pair<float, int> sort_type;

template<bool expand_labels>
std::string classify(af::array arr, int k) {
    std::stringstream ss;
    if (expand_labels) {
        af::array vec = arr(af::span, k).as(f32);
        float *h_vec  = vec.host<float>();
        std::vector<sort_type> data;

        for (int i = 0; i < (int)vec.elements(); i++)
            data.push_back(std::make_pair(h_vec[i], i));

        std::stable_sort(data.begin(), data.end(), compare);

        af::freeHost(h_vec);
        ss << data[0].second;
    } else {
        ss << (int)(arr(k).as(f32).scalar<float>());
    }
    return ss.str();
}

template<bool expand_labels>
static void setup_mnist(int *num_classes, int *num_train, int *num_test,
                        af::array &train_images, af::array &test_images,
                        af::array &train_labels, af::array &test_labels,
                        float frac, std::string lib_folder) {
    std::vector<dim_t> idims;
    std::vector<float> idata;
    // std::cout << "ASSETS located at:\n";
    // std::cout << ASSETS_DIR << std::endl;
    // char *fpath = std::getenv("R_HOME");
    // if (fpath){
    //   // std::cout << "R_HOME: " << fpath << '\n';
    // }
    // else{
    //   std::cout << "R_HOME not Found" << '\n';
    //   return;
    // }
    // char *rpath = realpath(fpath, NULL);
    std::string root = "/viewmastR/extdata/mnist";
    std::string path1 = lib_folder;
    std::string ifile = "/images-subset";
    // std::strcat(fpath, mpath.c_str());
    std::string images_f = path1.append(root).append(ifile);
    read_idx(idims, idata, images_f.c_str());

    std::vector<dim_t> ldims;
    std::vector<unsigned> ldata;
    std::string path2 = lib_folder;
    std::string lfile = "/labels-subset";
    // std::strcat(fpath, mpath.c_str());
    std::string labels_f = path2.append(root).append(lfile);
    read_idx(ldims, ldata, labels_f.c_str());

    std::reverse(idims.begin(), idims.end());
    unsigned numdims = idims.size();
    af::array images = af::array(af::dim4(numdims, &idims[0]), &idata[0]);

    af::array R             = af::randu(10000, 1);
    af::array cond          = R < std::min(frac, 0.8f);
    af::array train_indices = where(cond);
    af::array test_indices  = where(!cond);

    train_images = lookup(images, train_indices, 2) / 255;
    test_images  = lookup(images, test_indices, 2) / 255;

    *num_classes = 10;
    *num_train   = train_images.dims(2);
    *num_test    = test_images.dims(2);

    if (expand_labels) {
        train_labels = af::constant(0, *num_classes, *num_train);
        test_labels  = af::constant(0, *num_classes, *num_test);

        unsigned *h_train_idx = train_indices.host<unsigned>();
        unsigned *h_test_idx  = test_indices.host<unsigned>();

        for (int ii = 0; ii < *num_train; ii++) {
            train_labels(ldata[h_train_idx[ii]], ii) = 1;
        }

        for (int ii = 0; ii < *num_test; ii++) {
            test_labels(ldata[h_test_idx[ii]], ii) = 1;
        }

        af::freeHost(h_train_idx);
        af::freeHost(h_test_idx);
    } else {
        af::array labels = af::array(ldims[0], &ldata[0]);
        train_labels     = labels(train_indices);
        test_labels      = labels(test_indices);
    }

    return;
}

#if 0
static af::array randidx(int num, int total)
{
    af::array locs;
    do {
        locs = af::where(af::randu(total, 1) < float(num * 2) / total);
    } while (locs.elements() < num);

    return locs(af::seq(num));
}
#endif

template<bool expand_labels>
static void display_results(const af::array &test_images,
                            const af::array &test_output,
                            const af::array &test_actual, int num_display) {
#if 0
    af::array locs = randidx(num_display, test_images.dims(2));

    af::array disp_in  = test_images(af::span, af::span, locs);
    af::array disp_out = expand_labels ? test_output(af::span, locs) : test_output(locs);

    for (int i = 0; i < 5; i++) {

        int imgs_per_iter = num_display / 5;
        for (int j = 0; j < imgs_per_iter; j++) {

            int k = i * imgs_per_iter + j;
            af::fig("sub", 2, imgs_per_iter / 2, j+1);

            af::image(disp_in(af::span, af::span, k).T());
            std::string pred_name = std::string("Predicted: ");
            pred_name = pred_name + classify<expand_labels>(disp_out, k);
            af::fig("title", pred_name.c_str());
        }

        printf("Press any key to see next set");
        getchar();
    }
#else
    using namespace af;
    for (int i = 0; i < num_display; i++) {
        std::cout << "Predicted: " << classify<expand_labels>(test_output, i)
                  << std::endl;
        std::cout << "Actual: " << classify<expand_labels>(test_actual, i)
                  << std::endl;

        unsigned char *img =
            (test_images(span, span, i) > 0.1f).as(u8).host<unsigned char>();
        for (int j = 0; j < 28; j++) {
            for (int k = 0; k < 28; k++) {
                std::cout << (img[j * 28 + k] ? "\u2588" : " ") << " ";
            }
            std::cout << std::endl;
        }
        af::freeHost(img);
        getchar();
    }
#endif
}
