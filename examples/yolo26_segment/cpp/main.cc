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


static void draw_segment_mask(image_buffer_t *image, const object_detect_result *det)
{
    static const unsigned char colors[][3] = {
        {255, 56, 56}, {56, 255, 56}, {56, 56, 255}, {255, 255, 56},
        {255, 56, 255}, {56, 255, 255}, {255, 128, 56}, {128, 56, 255},
    };

    if (image == NULL || det == NULL || det->mask.data == NULL ||
        image->format != IMAGE_FORMAT_RGB888 || det->cls_id < 0)
    {
        return;
    }

    const unsigned char *color = colors[det->cls_id % (sizeof(colors) / sizeof(colors[0]))];
    const float alpha = 0.45f;
    int left = det->box.left < 0 ? 0 : det->box.left;
    int top = det->box.top < 0 ? 0 : det->box.top;
    int right = det->box.right > image->width ? image->width : det->box.right;
    int bottom = det->box.bottom > image->height ? image->height : det->box.bottom;
    if (left >= right || top >= bottom)
    {
        return;
    }

    bool full_image_mask = det->mask.width == image->width && det->mask.height == image->height;
    bool box_local_mask = det->mask.width == right - left && det->mask.height == bottom - top;
    if (!full_image_mask && !box_local_mask)
    {
        return;
    }

    for (int y = top; y < bottom; y++)
    {
        for (int x = left; x < right; x++)
        {
            int mask_idx = full_image_mask ? y * det->mask.width + x
                                           : (y - top) * det->mask.width + (x - left);
            if (det->mask.data[mask_idx] == 0)
            {
                continue;
            }
            unsigned char *p = image->virt_addr + (y * image->width + x) * 3;
            p[0] = (unsigned char)(p[0] * (1.0f - alpha) + color[0] * alpha);
            p[1] = (unsigned char)(p[1] * (1.0f - alpha) + color[1] * alpha);
            p[2] = (unsigned char)(p[2] * (1.0f - alpha) + color[2] * alpha);
        }
    }
}

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
    object_detect_result_list od_results;
    memset(&od_results, 0, sizeof(object_detect_result_list));
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

        draw_segment_mask(&src_image, det_result);
        draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

        snprintf(text, sizeof(text), "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
        draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
    }

    ret = write_image("out.png", &src_image);



     out:
    
        deinit_post_process();

        release_object_detect_result_list(&od_results);

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
