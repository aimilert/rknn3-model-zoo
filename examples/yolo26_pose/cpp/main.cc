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

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yolo26.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

static const int COCO_POSE_SKELETON[][2] = {
    {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
    {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2},
    {1, 3}, {2, 4}, {3, 5}, {4, 6},
};

/*-------------------------------------------
                  Main Function
-------------------------------------------*/

int main(int argc, char **argv)
{
    if (argc < 5) {
        printf("%s <model_path> <weight_path> <image_path> <core_mask> [postprocess_plugin_path]\n", argv[0]);
        return -1;
    }

    const char* model_path  = argv[1];
    const char* weight_path = argv[2];
    const char* image_path  = argv[3];
    uint32_t    core_mask   = strtoul(argv[4], nullptr, 16);
    const char* postprocess_plugin_path = NULL;
    
    
    if (argc == 6) {
        postprocess_plugin_path = argv[5];
    }

    int ret;
    double average_time = 0.0;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));

    ret = init_post_process();

    printf("--> init model\n");
    ret = init_yolo26_model(model_path, weight_path, &rknn_app_ctx, core_mask, postprocess_plugin_path);
    if (ret != 0)
    {
        printf("init_yolo26_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }

    ret = read_image(image_path, &src_image);
    if (ret != 0)
    {
        printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
        goto out;
    }

    object_detect_result_list od_results;

    printf("--> inference model\n");

    ret = inference_yolo26_model(&rknn_app_ctx, &src_image, &od_results, &average_time, NMS_THRESH, BOX_THRESH);
    if (ret != 0)
    {
        printf("inference_yolo26_model fail! ret=%d\n", ret);
        goto out;

    }


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

        draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

        snprintf(text, sizeof(text), "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
        draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);

        if (det_result->keypoint_count == OBJ_KEYPOINT_NUM)
        {
            for (int j = 0; j < (int)(sizeof(COCO_POSE_SKELETON) / sizeof(COCO_POSE_SKELETON[0])); j++)
            {
                int p0 = COCO_POSE_SKELETON[j][0];
                int p1 = COCO_POSE_SKELETON[j][1];
                pose_keypoint_t *k0 = &det_result->keypoints[p0];
                pose_keypoint_t *k1 = &det_result->keypoints[p1];
                if (k0->prop >= KEYPOINT_THRESH && k1->prop >= KEYPOINT_THRESH)
                {
                    draw_line(&src_image, k0->x, k0->y, k1->x, k1->y, COLOR_GREEN, 2);
                }
            }
            for (int j = 0; j < det_result->keypoint_count; j++)
            {
                pose_keypoint_t *kpt = &det_result->keypoints[j];
                if (kpt->prop >= KEYPOINT_THRESH)
                {
                    draw_circle(&src_image, kpt->x, kpt->y, 3, COLOR_YELLOW, -1);
                }
            }
        }
    }

    ret = write_image("out.png", &src_image);
    
    out:

        deinit_post_process();

        ret = release_yolo26_model(&rknn_app_ctx);
        if (ret != 0)
        {
            printf("release_yolo26_model fail! ret=%d\n", ret);
        }

        if (src_image.virt_addr != NULL)
        {
            free(src_image.virt_addr);
        }

        return 0;
}
