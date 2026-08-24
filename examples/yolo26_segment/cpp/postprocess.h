#ifndef _RKNN_YOLO26_DEMO_POSTPROCESS_H_
#define _RKNN_YOLO26_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn3_api.h"
#include "common.h"
#include "image_utils.h"

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80
#define OBJ_MASK_DIM 32
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define SEG_MASK_THRESH 0.5f

// class rknn_app_context_t;

typedef struct {
    int width;
    int height;
    unsigned char *data;
} object_segment_mask_t;

typedef struct {
    image_rect_t box;
    float box_float[4]; // x1, y1, x2, y2 in original image coordinates.
    float prop;
    int cls_id;
    object_segment_mask_t mask;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

int init_post_process();
void deinit_post_process();
void release_object_detect_result_list(object_detect_result_list *od_results);
const char *coco_cls_to_name(int cls_id);
int post_process_after_exYoloPostProcess(rknn_app_context_t *app_ctx, void *outputs, rknn3_tensor_attr output_attrs, letterbox_t *letter_box, object_detect_result_list *od_results);
int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results);
#endif //_RKNN_YOLO26_DEMO_POSTPROCESS_H_
