#ifndef _RKNN_YOLOV8_DEMO_POSTPROCESS_H_
#define _RKNN_YOLOV8_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include "rknn3_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_OBJ_NUM 256
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
} box_t;

typedef struct {
    float score;
    float class_id;
    box_t box;
} object_detect_result;

/**
 * YOLO 后处理私有数据结构体
 * 作为 rknn3_custom_op_context.priv_data 使用
 */
typedef struct _rknn3_yolo_postprocess_context
{
    uint32_t model_in_w;    // 模型输入宽度
    uint32_t model_in_h;    // 模型输入高度
    float conf_threshold;   // 置信度阈值
    float nms_threshold;    // NMS阈值
} rknn3_yolo_postprocess_context;

/**
 * YOLO 后处理函数
 * @param pp_ctx 后处理上下文
 * @param inputs 输入张量数组
 * @param n_inputs 输入张量数量
 * @param outputs 输出张量数组
 * @param n_outputs 输出张量数量
 * @return 成功返回检测到的对象数量（>= 0），失败返回 -1
 */
int post_process(rknn3_yolo_postprocess_context *pp_ctx, const rknn3_tensor inputs[], uint32_t n_inputs, rknn3_tensor outputs[], uint32_t n_outputs);

#ifdef __cplusplus
}
#endif

#endif //_RKNN_YOLOV8_DEMO_POSTPROCESS_H_
