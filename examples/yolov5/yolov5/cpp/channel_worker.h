#ifndef _RKNN_YOLOV5_CHANNEL_WORKER_H_
#define _RKNN_YOLOV5_CHANNEL_WORKER_H_

#include "rknn3_api.h"
#include "common.h"
#include "yolov5.h"   // defines rknn_app_context_t before including postprocess.h
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_CHANNELS_PER_BOARD 8
#define MAX_BOARDS 4
#define MAX_CHANNELS (MAX_BOARDS * MAX_CHANNELS_PER_BOARD)

// Global stop flag: set from a signal handler to stop all worker threads.
extern volatile int g_channel_stop_all;

// Per-channel performance statistics
typedef struct {
    uint64_t total_frames;
    double   total_preprocess_ms;
    double   total_inference_ms;
    double   total_postprocess_ms;
    double   total_loop_ms;         // sum of per-frame busy time (pre+infer+post)
    double   total_wall_ms;         // wall-clock from thread start to end (includes throttle sleep)
    double   last_loop_ms;          // most recent loop latency
    double   min_loop_ms;
    double   max_loop_ms;
} channel_perf_stats_t;

// Per-channel context — one per video channel, bound to a specific NPU core
typedef struct {
    int               channel_id;       // global channel index (0 ~ 31)
    int               board_id;         // which RK1828 board (0 ~ 3)
    int               core_id;          // NPU core index within board (0 ~ 7)
    uint32_t          core_mask;        // single-bit core mask (1 << core_id)
    char              device_id[128];   // device ID string for rknn3_init_extend

    // RKNN inference context (independent per channel)
    rknn_app_context_t rknn_ctx;
    bool              ctx_initialized;
    bool              mutex_inited;

    // Thread control
    pthread_t         thread;
    bool              running;
    bool              should_stop;

    // Benchmark parameters
    int               target_fps;       // throttle target frames/sec; 0 = run as fast as possible
    uint64_t          target_frames;    // total frames to run per channel; 0 = run until stopped

    // Input image (pre-loaded for benchmark; in production this would be a video source)
    image_buffer_t    input_image;
    bool              image_loaded;

    // Performance statistics
    channel_perf_stats_t perf;
    pthread_mutex_t   perf_mutex;       // protects perf fields
} channel_context_t;

// Per-board context
typedef struct {
    int               board_id;
    char              device_id[128];
    int               num_channels;     // number of active channels (default 8)
    channel_context_t channels[MAX_CHANNELS_PER_BOARD];
} board_context_t;

/**
 * @brief Initialize a single channel: create RKNN context, load model, bind to core
 *
 * @param ch           Channel context (pre-filled with channel_id, board_id, core_id, device_id)
 * @param model_path   RKNN model file path
 * @param weight_path  RKNN weight file path
 * @return 0 on success, negative on error
 */
int channel_init(channel_context_t* ch, const char* model_path, const char* weight_path);

/**
 * @brief Release a channel's RKNN context and resources
 */
void channel_release(channel_context_t* ch);

/**
 * @brief Load a test image into the channel (for benchmark mode)
 *
 * @return 0 on success, negative on error
 */
int channel_load_image(channel_context_t* ch, const char* image_path);

/**
 * @brief Worker thread entry: runs continuous inference loop until should_stop
 *        or target_frames is reached.
 *
 * @param arg  pointer to channel_context_t
 * @return NULL
 */
void* channel_worker_thread(void* arg);

/**
 * @brief Read a channel's performance stats (thread-safe copy)
 */
void channel_get_perf(channel_context_t* ch, channel_perf_stats_t* out_perf);

/**
 * @brief Reset a channel's performance stats
 */
void channel_reset_perf(channel_context_t* ch);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // _RKNN_YOLOV5_CHANNEL_WORKER_H_
