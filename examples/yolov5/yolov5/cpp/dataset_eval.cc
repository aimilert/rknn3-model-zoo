// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "yolov5.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include <fstream>

// COCO 91-class index mapping (from 80-class to 91-class)
const std::vector<int> coco91class = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75,
    76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
};

/**
 * @brief 读取文本文件，返回每一行的内容
 * @param file_path 文件路径
 * @return 包含每一行字符串的 vector（不含换行符）
 */
std::vector<std::string> readLinesFromFile(const std::string& file_path) {
    std::vector<std::string> lines;
    std::ifstream file(file_path);
    if (!file.is_open()) {
        printf("Error: Cannot open file: %s\n", file_path.c_str());
        return lines;
    }
    std::string line;
    while (std::getline(file, line)) {
        // getline 会自动去掉 \n 或 \r\n，line 中不包含换行符
        lines.push_back(line);
    }
    file.close();
    return lines;
}

std::string basename(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

std::string remove_extension(const std::string& filename) {
    size_t pos = filename.find_last_of(".");
    return (pos == std::string::npos) ? filename : filename.substr(0, pos);
}

int getImageIdFromPath(const std::string& img_path) {
    std::string base = basename(img_path);
    std::string stem = remove_extension(base);
    try {
        return std::stoi(stem);
    } catch (...) {
        return -1;
    }
}

/**
 * @brief 写入所有检测结果到 JSON 文件（完整数组格式）
 * @param results_list 检测结果列表
 * @param output_file 输出文件路径
 */
struct DetectionResult {
    std::string img_path;
    int cls;
    double xywh[4];  // cx, cy, w, h
    double conf;
};

void writeResultsToJson(const std::vector<DetectionResult>& results_list, const std::string& output_file) {
    std::ofstream json_fp(output_file);
    if (!json_fp.is_open()) {
        printf("Error: Cannot open output file: %s\n", output_file.c_str());
        return;
    }

    // 写入开头 [
    json_fp << "[";
    bool first = true;
    for (const auto& res : results_list) {
        int img_id = getImageIdFromPath(res.img_path);

        double bbox_x = res.xywh[0], bbox_y = res.xywh[1], w = res.xywh[2], h = res.xywh[3];
        int category_id = coco91class[res.cls];

        // 添加逗号（如果不是第一条）
        if (!first) {
            json_fp << ",";
        } else {
            first = false;
        }

        // 写入 JSON 对象
        json_fp << "  {"
                << "\"image_id\":" << img_id << ","
                << "\"category_id\":" << category_id << ","
                << "\"bbox\":[" << bbox_x << "," << bbox_y << "," << w << "," << h << "],"
                << "\"score\":" << res.conf
                << "}";
    }

    // 写入结尾 ]
    json_fp << "]";
    json_fp.close();

    printf("Total detections: %zu\n", results_list.size());
    printf("Results saved to %s\n", output_file.c_str());
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc < 4 || argc > 5) {
        printf("%s <model_path> <weight_path> <core_mask> [postprocess_plugin_path]\n", argv[0]);
        return -1;
    }

    const char* model_path  = argv[1];
    const char* weight_path = argv[2];
    uint32_t    core_mask   = strtoul(argv[3], nullptr, 16);
    const char* postprocess_plugin_path = NULL;
    if (argc == 5) {
        postprocess_plugin_path = argv[4];
    }
    const std::string txt_path = "/userdata/coco_dataset_test_path.txt";

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    printf("--> init model\n");
    ret = init_yolov5_model(model_path, weight_path, &rknn_app_ctx, core_mask, postprocess_plugin_path);
    if (ret != 0)
    {
        printf("init_yolov5_model fail! ret=%d model_path=%s\n", ret, model_path);
        deinit_post_process();
        return -1;
    }
    // 打开 JSON 文件（追加或新建）
    std::ofstream json_fp("results_rknn.json", std::ios::app);
    if (!json_fp.is_open()) {
        printf("Error: Cannot open results.json\n");
        return 1;
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    std::vector<DetectionResult> results;
    // 遍历文件夹中图片
    std::vector<std::string> image_paths = readLinesFromFile(txt_path);
    for (const auto& image_path : image_paths) {
        printf("Processing image: %s\n", image_path.c_str());

        ret = read_image(image_path.c_str(), &src_image);
        if (ret != 0)
        {
            printf("read image fail! ret=%d image_path=%s\n", ret, image_path.c_str());
            goto out;
        }

        object_detect_result_list od_results;

        printf("--> inference model\n");
        ret = inference_yolov5_model(&rknn_app_ctx, &src_image, &od_results, NULL);
        if (ret != 0)
        {
            printf("inference_yolov5_model fail! ret=%d\n", ret);
            goto out;
        }
        printf("--> inference model done\n");

        // 画框和概率
        char text[256];
        for (int i = 0; i < od_results.count; i++)
        {
            object_detect_result *det_result = &(od_results.results[i]);
            printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
                det_result->box.left, det_result->box.top,
                det_result->box.right, det_result->box.bottom,
                det_result->prop);
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;

            // sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            DetectionResult res;
            res.img_path = image_path;
            res.cls = det_result->cls_id;
            res.xywh[0] = x1;  // x1
            res.xywh[1] = y1;  // y1
            res.xywh[2] = x2 - x1;  // width
            res.xywh[3] = y2 - y1;  // height
            res.conf = det_result->prop; // confidence
            results.push_back(res);
        }
    }

    // 写入所有检测结果到 JSON 文件
    writeResultsToJson(results, "results_rknn.json");

out:
    deinit_post_process();
    ret = release_yolov5_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolov5_model fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }

    return 0;
}
