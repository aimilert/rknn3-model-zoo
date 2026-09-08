// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Multi-channel YOLOv5 inference worker.
//
// Each channel owns an independent rknn3_context bound to one specific
// NPU core (core_mask = 1 << core_id) on one specific RK182x device
// (device_id from rknn3_find_devices).  All channels run concurrently in
// their own pthread, giving N independent single-core inference streams.
//
// Model requirement: use the "_rknn3" (mode C) model converted with the
// default core_num=1, i.e. yolov5s_rknn3.rknn.  The built-in post-process
// runs on the NPU, so there is no coprocessor-CPU contention across cores.

#include "channel_worker.h"
#include "image_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

volatile int g_channel_stop_all = 0;

static double _elapsed_ms(const struct timeval* start, const struct timeval* end)
{
    return (end->tv_sec - start->tv_sec) * 1000.0
         + (end->tv_usec - start->tv_usec) / 1000.0;
}

int channel_init(channel_context_t* ch, const char* model_path, const char* weight_path)
{
    if (!ch || !model_path || !weight_path)
        return -1;

    memset(&ch->rknn_ctx, 0, sizeof(rknn_app_context_t));
    ch->mutex_inited = false;

    // Bind to one NPU core: 0x01, 0x02, 0x04, ... 0x80
    uint32_t core_mask = 1u << ch->core_id;
    // device_id NULL -> default device; otherwise bind to the target board
    const char* dev_id = ch->device_id[0] ? ch->device_id : NULL;

    int ret = init_yolov5_model_ex(model_path, weight_path, &ch->rknn_ctx,
                                   core_mask, NULL /* no plugin for _rknn3 */,
                                   dev_id, true /* quiet */);
    if (ret != 0) {
        printf("[ch%02d] init_yolov5_model_ex failed ret=%d (dev=%s, core=%d, mask=0x%x)\n",
               ch->channel_id, ret, dev_id ? dev_id : "auto", ch->core_id, core_mask);
        return ret;
    }

    pthread_mutex_init(&ch->perf_mutex, NULL);
    ch->mutex_inited = true;
    ch->ctx_initialized = true;
    ch->perf.min_loop_ms = 0.0;
    ch->perf.max_loop_ms = 0.0;
    return 0;
}

void channel_release(channel_context_t* ch)
{
    if (!ch)
        return;

    if (ch->image_loaded && ch->input_image.virt_addr) {
        free(ch->input_image.virt_addr);
        ch->input_image.virt_addr = NULL;
        ch->image_loaded = false;
    }

    if (ch->ctx_initialized) {
        release_yolov5_model(&ch->rknn_ctx);
        ch->ctx_initialized = false;
    }

    if (ch->mutex_inited) {
        pthread_mutex_destroy(&ch->perf_mutex);
        ch->mutex_inited = false;
    }
}

int channel_load_image(channel_context_t* ch, const char* image_path)
{
    if (!ch || !image_path)
        return -1;

    if (ch->image_loaded && ch->input_image.virt_addr) {
        free(ch->input_image.virt_addr);
        ch->input_image.virt_addr = NULL;
        ch->image_loaded = false;
    }

    image_buffer_t img;
    memset(&img, 0, sizeof(img));
    int ret = read_image(image_path, &img);
    if (ret != 0) {
        printf("[ch%02d] read_image %s failed ret=%d\n", ch->channel_id, image_path, ret);
        return ret;
    }
    ch->input_image = img;
    ch->image_loaded = true;
    return 0;
}

void channel_reset_perf(channel_context_t* ch)
{
    if (!ch)
        return;
    pthread_mutex_lock(&ch->perf_mutex);
    ch->perf.total_frames = 0;
    ch->perf.total_preprocess_ms = 0.0;
    ch->perf.total_inference_ms = 0.0;
    ch->perf.total_postprocess_ms = 0.0;
    ch->perf.total_loop_ms = 0.0;
    ch->perf.total_wall_ms = 0.0;
    ch->perf.last_loop_ms = 0.0;
    ch->perf.min_loop_ms = 0.0;
    ch->perf.max_loop_ms = 0.0;
    pthread_mutex_unlock(&ch->perf_mutex);
}

void channel_get_perf(channel_context_t* ch, channel_perf_stats_t* out_perf)
{
    if (!ch || !out_perf)
        return;
    pthread_mutex_lock(&ch->perf_mutex);
    *out_perf = ch->perf;
    pthread_mutex_unlock(&ch->perf_mutex);
}

void* channel_worker_thread(void* arg)
{
    channel_context_t* ch = (channel_context_t*)arg;
    if (!ch)
        return NULL;

    if (!ch->image_loaded) {
        printf("[ch%02d] no input image loaded, worker exits\n", ch->channel_id);
        ch->running = false;
        return NULL;
    }

    object_detect_result_list od_results;
    rknn_yolov5_times times;
    uint64_t target_frames = ch->target_frames;
    int target_fps = ch->target_fps;
    // Requested interval between consecutive frames (microseconds).
    long frame_interval_us = target_fps > 0 ? (1000000L / target_fps) : 0;

    printf("[ch%02d] start on dev=%s core=%d\n", ch->channel_id,
           ch->device_id[0] ? ch->device_id : "auto", ch->core_id);

    struct timeval wall_t0;
    gettimeofday(&wall_t0, NULL);
    uint64_t done_frames = 0;
    while (ch->running && !ch->should_stop && !g_channel_stop_all) {
        struct timeval loop_start, loop_end;
        gettimeofday(&loop_start, NULL);

        int ret = inference_yolov5_model(&ch->rknn_ctx, &ch->input_image, &od_results, &times);

        gettimeofday(&loop_end, NULL);
        double loop_ms = _elapsed_ms(&loop_start, &loop_end);

        if (ret != 0) {
            printf("[ch%02d] inference failed ret=%d\n", ch->channel_id, ret);
            break;
        }

        pthread_mutex_lock(&ch->perf_mutex);
        ch->perf.total_frames++;
        ch->perf.total_preprocess_ms += times.preprocess_ms;
        ch->perf.total_inference_ms += times.inference_ms;
        ch->perf.total_postprocess_ms += times.postprocess_ms;
        ch->perf.total_loop_ms += loop_ms;
        ch->perf.last_loop_ms = loop_ms;
        if (ch->perf.min_loop_ms == 0.0 || loop_ms < ch->perf.min_loop_ms)
            ch->perf.min_loop_ms = loop_ms;
        if (loop_ms > ch->perf.max_loop_ms)
            ch->perf.max_loop_ms = loop_ms;
        pthread_mutex_unlock(&ch->perf_mutex);

        done_frames++;

        // Stop after reaching the target frame count (frames>0), else run until stopped.
        if (target_frames > 0 && done_frames >= target_frames) {
            break;
        }

        // Throttle to a target frame rate (simulate real-time video input).
        if (frame_interval_us > 0) {
            long sleep_us = frame_interval_us - (long)(loop_ms * 1000.0);
            if (sleep_us > 0) {
                usleep((useconds_t)sleep_us);
            }
        }
    }

    struct timeval wall_t1;
    gettimeofday(&wall_t1, NULL);
    pthread_mutex_lock(&ch->perf_mutex);
    ch->perf.total_wall_ms = _elapsed_ms(&wall_t0, &wall_t1);
    pthread_mutex_unlock(&ch->perf_mutex);

    printf("[ch%02d] done, %llu frames\n", ch->channel_id,
           (unsigned long long)done_frames);
    ch->running = false;
    return NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif
