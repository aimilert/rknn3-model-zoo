#ifndef _RKNN_YOLO26_DEMO_POSTPROCESS_H_
#define _RKNN_YOLO26_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn3_api.h"
#include "common.h"
#include "image_utils.h"

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 300
#define OBJ_CLASS_NUM 80
#define OBJ_KEYPOINT_NUM 17
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define KEYPOINT_THRESH 0.25

// class rknn_app_context_t;

typedef struct {
    int x;
    int y;
    float prop;
    float x_float;
    float y_float;
} pose_keypoint_t;

typedef struct {
    image_rect_t box;
    float box_float[4]; // x1, y1, x2, y2 in original image coordinates.
    float prop;
    int cls_id;
    int keypoint_count;
    pose_keypoint_t keypoints[OBJ_KEYPOINT_NUM];
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

int init_post_process();
void deinit_post_process();
const char *coco_cls_to_name(int cls_id);
int post_process_after_exYoloPostProcess(rknn_app_context_t *app_ctx, void *outputs, rknn3_tensor_attr output_attrs, letterbox_t *letter_box, object_detect_result_list *od_results);
int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results);
#endif //_RKNN_YOLO26_DEMO_POSTPROCESS_H_
