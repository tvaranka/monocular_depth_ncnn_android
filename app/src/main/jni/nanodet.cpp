// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "nanodet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"
#include <string>

#include <android/log.h>

#define  LOG_TAG "media-lib"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)


NanoDet::NanoDet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int NanoDet::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    nanodet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    nanodet.opt = ncnn::Option();

#if NCNN_VULKAN
    nanodet.opt.use_vulkan_compute = use_gpu;
#endif

    nanodet.opt.num_threads = ncnn::get_big_cpu_count();
    nanodet.opt.blob_allocator = &blob_pool_allocator;
    nanodet.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    std::string s_modeltype = modeltype;
    if (s_modeltype.compare("m")) {
        sprintf(parampath, "nanodet-%s.param", modeltype);
        sprintf(modelpath, "nanodet-%s.bin", modeltype);
    } else if(s_modeltype.compare("midas")) {
        sprintf(parampath, "midas_v21_small-int8.param");
        sprintf(modelpath, "midas_v21_small-int8.bin");
    }

    nanodet.load_param(parampath);
    nanodet.load_model(modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int NanoDet::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    nanodet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    nanodet.opt = ncnn::Option();

#if NCNN_VULKAN
    nanodet.opt.use_vulkan_compute = use_gpu;
#endif

    nanodet.opt.num_threads = ncnn::get_big_cpu_count();
    nanodet.opt.blob_allocator = &blob_pool_allocator;
    nanodet.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    std::string s_modeltype = modeltype;
    if (s_modeltype.compare("m") == 0) {
        sprintf(parampath, "nanodet-%s.param", modeltype);
        sprintf(modelpath, "nanodet-%s.bin", modeltype);
    } else {
        sprintf(parampath, "midas_v21_small-int8.param");
        sprintf(modelpath, "midas_v21_small-int8.bin");
    }

    nanodet.load_param(mgr, parampath);
    nanodet.load_model(mgr, modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int NanoDet::detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
    int width = rgb.cols;
    int height = rgb.rows;

    int w = target_size;
    int h = target_size;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, width, height, w, h);

    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = nanodet.create_extractor();

    ex.input("input.1", in);
    ncnn::Mat out;
    ex.extract("649", out);

    ncnn::resize_bilinear(out, out, width, height);
    cv::Mat cv_out(out.h, out.w, CV_8UC1);
    out.to_pixels(cv_out.data, ncnn::Mat::PIXEL_GRAY);
    //ncnn::Mat::to_pixels_resize(cv_out.data, width, height)
    //out.to_pixels_resize(cv_out.data, ncnn::Mat::PIXEL_GRAY, width, height);
    //LOGI("out width: %d, out height: %d\n", out.w, out.h);
    //cv::Mat cv_out(height, width, CV_8UC1, out.data);
    cv::Mat cv_out_rgb(cv_out.rows, cv_out.cols, CV_8UC3);
    cv::cvtColor(cv_out, cv_out_rgb, cv::COLOR_GRAY2RGB);
    cv_out_rgb.copyTo(rgb);

    return 0;
}