// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <errno.h>
#include <fstream>
#include <iomanip>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/time.h>
#include <vector>

#include "file_utils.h"
#include "image_utils.h"
#include "yolo26.h"

static const int COCO_ID_LIST[OBJ_CLASS_NUM] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
};
static const float DEFAULT_CONF_THRESH = 0.001f;
static const float DEFAULT_NMS_THRESH = 0.70f;
static const char *DEFAULT_DATASET_LIST = "/userdata/coco_dataset_test_path.txt";
static const char *DEFAULT_OUTPUT_JSON = "results_rknn_bbox.json";

struct EvalConfig {
    const char *model_path;
    const char *weight_path;
    uint32_t core_mask;
    const char *postprocess_plugin_path;
    std::string dataset_list;
    std::string output_json;
    float conf_thresh;
    float nms_thresh;
};

struct DetectionResult {
    int image_id;
    int category_id;
    double bbox[4];
    double score;
};

static void print_usage(const char *argv0)
{
    printf("%s <model_path> <weight_path> <core_mask> [postprocess_plugin_path] "
           "[--dataset-list path] [--output json] [--conf-thresh value] [--nms-thresh value]\n",
           argv0);
}

static bool parse_core_mask(const char *text, uint32_t *core_mask)
{
    char *endptr = nullptr;
    errno = 0;
    unsigned long value = strtoul(text, &endptr, 16);
    if (errno != 0 || endptr == text || *endptr != '\0' || value == 0 || value > UINT32_MAX)
    {
        return false;
    }
    *core_mask = (uint32_t)value;
    return true;
}

static bool parse_args(int argc, char **argv, EvalConfig *cfg)
{
    if (argc < 4)
    {
        print_usage(argv[0]);
        return false;
    }

    cfg->model_path = argv[1];
    cfg->weight_path = argv[2];
    if (!parse_core_mask(argv[3], &cfg->core_mask))
    {
        printf("invalid core_mask: %s\n", argv[3]);
        return false;
    }
    cfg->postprocess_plugin_path = nullptr;
    cfg->dataset_list = DEFAULT_DATASET_LIST;
    cfg->output_json = DEFAULT_OUTPUT_JSON;
    cfg->conf_thresh = DEFAULT_CONF_THRESH;
    cfg->nms_thresh = DEFAULT_NMS_THRESH;

    for (int i = 4; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--dataset-list" && i + 1 < argc)
        {
            cfg->dataset_list = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc)
        {
            cfg->output_json = argv[++i];
        }
        else if (arg == "--conf-thresh" && i + 1 < argc)
        {
            cfg->conf_thresh = strtof(argv[++i], nullptr);
        }
        else if (arg == "--nms-thresh" && i + 1 < argc)
        {
            cfg->nms_thresh = strtof(argv[++i], nullptr);
        }
        else if (arg.size() > 0 && arg[0] != '-' && cfg->postprocess_plugin_path == nullptr)
        {
            cfg->postprocess_plugin_path = argv[i];
        }
        else
        {
            print_usage(argv[0]);
            return false;
        }
    }

    return true;
}

static std::vector<std::string> read_lines_from_file(const std::string &file_path)
{
    std::vector<std::string> lines;
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        printf("Error: Cannot open file: %s\n", file_path.c_str());
        return lines;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (!line.empty())
        {
            lines.push_back(line);
        }
    }
    return lines;
}

static std::string basename_no_ext(const std::string &path)
{
    size_t pos = path.find_last_of("/\\");
    std::string base = (pos == std::string::npos) ? path : path.substr(pos + 1);
    size_t dot = base.find_last_of(".");
    return (dot == std::string::npos) ? base : base.substr(0, dot);
}

static int get_image_id_from_path(const std::string &img_path)
{
    try
    {
        return std::stoi(basename_no_ext(img_path));
    }
    catch (...)
    {
        return -1;
    }
}

static double clamp_double(double value, double min_value, double max_value)
{
    if (value < min_value)
    {
        return min_value;
    }
    if (value > max_value)
    {
        return max_value;
    }
    return value;
}
static void write_results_to_json(const std::vector<DetectionResult> &results, const std::string &output_file)
{
    std::ofstream json_fp(output_file);
    if (!json_fp.is_open())
    {
        printf("Error: Cannot open output file: %s\n", output_file.c_str());
        return;
    }

    json_fp << "[";
    bool first = true;
    for (const auto &res : results)
    {
        if (res.image_id < 0 || res.bbox[2] <= 0.0 || res.bbox[3] <= 0.0)
        {
            continue;
        }

        if (!first)
        {
            json_fp << ",";
        }
        first = false;

        json_fp << "{";
        json_fp << "\"image_id\":" << res.image_id << ",";
        json_fp << "\"category_id\":" << res.category_id << ",";
        json_fp << "\"bbox\":[";
        json_fp << std::fixed << std::setprecision(3)
                << res.bbox[0] << "," << res.bbox[1] << ","
                << res.bbox[2] << "," << res.bbox[3] << "],";
        json_fp << "\"score\":" << std::fixed << std::setprecision(5) << res.score;
        json_fp << "}";
    }
    json_fp << "]";
    json_fp.close();

    printf("Total detections: %zu\n", results.size());
    printf("Results saved to %s\n", output_file.c_str());
}

int main(int argc, char **argv)
{
    EvalConfig cfg;
    if (!parse_args(argc, argv, &cfg))
    {
        return -1;
    }

    printf("YOLO26 segment COCO bbox eval\n");
    printf("  model        : %s\n", cfg.model_path);
    printf("  weight       : %s\n", cfg.weight_path);
    printf("  dataset list : %s\n", cfg.dataset_list.c_str());
    printf("  output json  : %s\n", cfg.output_json.c_str());
    printf("  conf/nms     : %.4f/%.2f\n", cfg.conf_thresh, cfg.nms_thresh);

    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    int ret = init_post_process();
    if (ret != 0)
    {
        return ret;
    }

    printf("--> init model\n");
    ret = init_yolo26_model(cfg.model_path, cfg.weight_path, &rknn_app_ctx,
                            cfg.core_mask, cfg.postprocess_plugin_path);
    if (ret != 0)
    {
        printf("init_yolo26_model fail! ret=%d model_path=%s\n", ret, cfg.model_path);
        deinit_post_process();
        return -1;
    }

    std::vector<std::string> image_paths = read_lines_from_file(cfg.dataset_list);
    if (image_paths.empty())
    {
        printf("No images found in dataset list: %s\n", cfg.dataset_list.c_str());
        release_yolo26_model(&rknn_app_ctx);
        deinit_post_process();
        return -1;
    }

    std::vector<DetectionResult> results;
    double total_npu_time = 0.0;

    for (size_t img_idx = 0; img_idx < image_paths.size(); img_idx++)
    {
        const std::string &image_path = image_paths[img_idx];
        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(image_buffer_t));

        ret = read_image(image_path.c_str(), &src_image);
        if (ret != 0)
        {
            printf("skip unreadable image: %s ret=%d\n", image_path.c_str(), ret);
            continue;
        }

        object_detect_result_list od_results;
        memset(&od_results, 0, sizeof(object_detect_result_list));
        ret = inference_yolo26_model(&rknn_app_ctx, &src_image, 
                                           &od_results, &total_npu_time, cfg.nms_thresh, cfg.conf_thresh);
        if (ret != 0)
        {
            printf("inference_yolo26_model_coco fail! ret=%d image_path=%s\n", ret, image_path.c_str());
            release_object_detect_result_list(&od_results);
            if (src_image.virt_addr != NULL)
            {
                free(src_image.virt_addr);
            }
            break;
        }

        int image_id = get_image_id_from_path(image_path);
        double img_w = (double)src_image.width;
        double img_h = (double)src_image.height;
        for (int i = 0; i < od_results.count; i++)
        {
            object_detect_result *det = &(od_results.results[i]);
            if (det->cls_id < 0 || det->cls_id >= OBJ_CLASS_NUM)
            {
                continue;
            }

            double x1 = clamp_double(det->box_float[0], 0.0, img_w);
            double y1 = clamp_double(det->box_float[1], 0.0, img_h);
            double x2 = clamp_double(det->box_float[2], 0.0, img_w);
            double y2 = clamp_double(det->box_float[3], 0.0, img_h);
            x2 = std::max(x1, x2);
            y2 = std::max(y1, y2);

            DetectionResult res;
            res.image_id = image_id;
            res.category_id = COCO_ID_LIST[det->cls_id];
            res.bbox[0] = x1;
            res.bbox[1] = y1;
            res.bbox[2] = x2 - x1;
            res.bbox[3] = y2 - y1;
            res.score = det->prop;
            results.push_back(res);
        }

        release_object_detect_result_list(&od_results);
        if (src_image.virt_addr != NULL)
        {
            free(src_image.virt_addr);
        }

        if (img_idx == 0 || (img_idx + 1) % 50 == 0 || img_idx + 1 == image_paths.size())
        {
            printf("  processed %zu/%zu images, detections=%zu\n",
                   img_idx + 1, image_paths.size(), results.size());
        }
    }

    write_results_to_json(results, cfg.output_json);
    if (!image_paths.empty())
    {
        printf("Average NPU time: %.2f ms\n", total_npu_time / image_paths.size());
    }

    ret = release_yolo26_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolo26_model fail! ret=%d\n", ret);
    }
    deinit_post_process();

    return ret;
}
