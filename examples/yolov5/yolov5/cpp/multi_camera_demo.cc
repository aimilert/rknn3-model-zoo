// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Multi-board / multi-channel YOLOv5 performance benchmark for RK1828.
//
// Runs N boards x M channels of independent single-core YOLOv5 inference in
// parallel.  Each channel binds to one specific RK182x device and one specific
// NPU core (core_mask = 1 << core_id), loading the "_rknn3" (mode C) model so
// the post-process runs on the NPU without coprocessor-CPU contention.
//
// Typical usage (4 boards x 8 cores = 32 streams):
//   ./rknn_multicam_demo \
//       --model model/yolov5s_rknn3.rknn \
//       --weight model/yolov5s_rknn3.weight \
//       --image model/bus.jpg \
//       --boards 4 --channels 8 --frames 200
//
// Press Ctrl-C at any time to stop and print the report (frames=0 runs forever).

#include "channel_worker.h"
#include "yolov5.h"
#include "image_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/time.h>

#include <string>
#include <vector>

static void usage(const char* prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("  --model <path>          RKNN model file (use yolov5*_rknn3.rknn, mode C)\n");
    printf("  --weight <path>         RKNN weight file\n");
    printf("  --image <path>          input image (JPEG/PNG/BMP) reused by every channel\n");
    printf("  --boards <n>            number of RK182x boards (default 4)\n");
    printf("  --channels <n>          channels per board, <=8 (default 8)\n");
    printf("  --frames <n>            frames per channel; 0 = run until Ctrl-C (default 200)\n");
    printf("  --fps <n>               throttle to n frames/sec per channel; 0 = max speed (default 0)\n");
    printf("  --device-id <id0#id1#>  explicit board device IDs (optional; auto-detect otherwise)\n");
    printf("  --help                  show this message\n");
}

static volatile sig_atomic_t g_sigint = 0;
static void on_sigint(int) { g_sigint = 1; g_channel_stop_all = 1; }

static bool parse_u64(const char* s, uint64_t* v)
{
    if (!s || !v || s[0] == '\0' || s[0] == '-') return false;
    char* end = nullptr;
    unsigned long long x = strtoull(s, &end, 10);
    if (end == s || *end != '\0') return false;
    *v = (uint64_t)x;
    return true;
}

static bool parse_int(const char* s, int* v)
{
    uint64_t x = 0;
    if (!parse_u64(s, &x) || x > 0x7fffffffULL) return false;
    *v = (int)x;
    return true;
}

static void split_device_ids(const std::string& in, std::vector<std::string>* out)
{
    size_t pos = 0;
    while (pos <= in.size()) {
        size_t end = in.find('#', pos);
        if (end == std::string::npos) {
            out->push_back(in.substr(pos));
            break;
        }
        out->push_back(in.substr(pos, end - pos));
        pos = end + 1;
    }
}

static void print_report(const std::vector<board_context_t>& boards, int n_channels_per_board,
                         double wall_ms)
{
    int total_channels = 0;
    uint64_t total_frames = 0;
    double   total_wall_ms = 0.0;

    printf("\n");
    printf("====================================================================================================\n");
    printf(" Channel | Board | Core |  Frames |  Avg Loop(ms) | Min(ms) | Max(ms) | Infer(ms) |  FPS   \n");
    printf("====================================================================================================\n");

    for (const auto& board : boards) {
        uint64_t board_frames = 0;
        double   board_wall_ms = 0.0;
        for (int c = 0; c < board.num_channels; ++c) {
            const channel_context_t& ch = board.channels[c];
            channel_perf_stats_t p;
            channel_get_perf(const_cast<channel_context_t*>(&ch), &p);

            double avg_loop = p.total_frames > 0 ? p.total_loop_ms / (double)p.total_frames : 0.0;
            double avg_infer = p.total_frames > 0 ? p.total_inference_ms / (double)p.total_frames : 0.0;
            double fps = p.total_wall_ms > 0.0 ? (double)p.total_frames / (p.total_wall_ms / 1e3) : 0.0;

            printf("  ch%02d    |  b%d    |  %d    | %7llu | %12.2f | %7.2f | %7.2f | %9.2f | %6.1f\n",
                   ch.channel_id, ch.board_id, ch.core_id,
                   (unsigned long long)p.total_frames, avg_loop, p.min_loop_ms, p.max_loop_ms,
                   avg_infer, fps);

            board_frames += p.total_frames;
            board_wall_ms = board_wall_ms < p.total_wall_ms ? p.total_wall_ms : board_wall_ms;
            total_channels++;
            total_frames += p.total_frames;
            total_wall_ms = total_wall_ms < p.total_wall_ms ? p.total_wall_ms : total_wall_ms;
        }
        printf("  -- board %d (dev=%s) total frames: %llu, board wall: %.1f ms\n",
               board.board_id, board.device_id, (unsigned long long)board_frames, board_wall_ms);
        printf("----------------------------------------------------------------------------------------------------\n");
    }

    double wall_s = wall_ms > 0.0 ? wall_ms / 1e3 : (total_wall_ms > 0.0 ? total_wall_ms / 1e3 : 1.0);
    printf("\nSummary:\n");
    printf("  Boards: %zu, Channels: %d, Wall time: %.1f ms\n", boards.size(), total_channels, wall_ms);
    printf("  Total frames processed: %llu\n", (unsigned long long)total_frames);
    printf("  Aggregate throughput: %.1f FPS (all channels combined)\n", (double)total_frames / wall_s);
    printf("  Avg per-channel FPS: %.1f\n",
           total_channels > 0 ? ((double)total_frames / wall_s) / total_channels : 0.0);
    printf("====================================================================================================\n");
}

int main(int argc, char** argv)
{
    const char* model_path = nullptr;
    const char* weight_path = nullptr;
    const char* image_path = nullptr;
    int boards = 4;
    int channels_per_board = 8;
    uint64_t frames = 200;
    int fps = 0;
    std::vector<std::string> device_ids;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--weight") == 0 && i + 1 < argc) {
            weight_path = argv[++i];
        } else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            image_path = argv[++i];
        } else if (strcmp(argv[i], "--boards") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &boards) || boards < 1 || boards > MAX_BOARDS) {
                printf("invalid --boards\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--channels") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &channels_per_board) ||
                channels_per_board < 1 || channels_per_board > MAX_CHANNELS_PER_BOARD) {
                printf("invalid --channels (must be 1..%d)\n", MAX_CHANNELS_PER_BOARD);
                return -1;
            }
        } else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], &frames)) {
                printf("invalid --frames\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--fps") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], &fps) || fps < 0) {
                printf("invalid --fps\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--device-id") == 0 && i + 1 < argc) {
            split_device_ids(argv[++i], &device_ids);
        } else {
            printf("unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return -1;
        }
    }

    if (!model_path || !weight_path || !image_path) {
        printf("--model, --weight and --image are required\n");
        usage(argv[0]);
        return -1;
    }
    if (!device_ids.empty() && (int)device_ids.size() != boards) {
        printf("--device-id expects exactly %d ids, got %zu\n", boards, device_ids.size());
        return -1;
    }

    // Load labels (needed by init_post_process; harmless for mode C).
    init_post_process();

    // Discover connected RK182x devices.
    rknn3_devices devs;
    memset(&devs, 0, sizeof(devs));
    int ret = rknn3_find_devices(&devs);
    if (ret != RKNN3_SUCCESS) {
        printf("rknn3_find_devices failed ret=%d\n", ret);
        return -1;
    }
    printf("found %u device(s):\n", devs.n_devices);
    for (uint32_t i = 0; i < devs.n_devices; ++i) {
        printf("  [%u] type=%s id=%s\n", i, devs.devices[i].type, devs.devices[i].id);
    }

    // Build board/channel table.
    std::vector<board_context_t> boards_vec(boards);
    for (int b = 0; b < boards; ++b) {
        board_context_t& board = boards_vec[b];
        memset(&board, 0, sizeof(board));
        board.board_id = b;
        board.num_channels = channels_per_board;

        if (!device_ids.empty()) {
            snprintf(board.device_id, sizeof(board.device_id), "%s", device_ids[b].c_str());
        } else if (b < (int)devs.n_devices) {
            snprintf(board.device_id, sizeof(board.device_id), "%s", devs.devices[b].id);
        } else {
            printf("not enough devices: need %d, found %u\n", boards, devs.n_devices);
            return -1;
        }

        for (int c = 0; c < channels_per_board; ++c) {
            channel_context_t& ch = board.channels[c];
            memset(&ch, 0, sizeof(ch));
            ch.channel_id = b * channels_per_board + c;
            ch.board_id = b;
            ch.core_id = c;
            ch.core_mask = 1u << c;
            snprintf(ch.device_id, sizeof(ch.device_id), "%s", board.device_id);
            ch.target_fps = fps;
            ch.target_frames = frames;

            printf("--> init ch%02d (board=%d core=%d dev=%s)\n",
                   ch.channel_id, b, c, board.device_id);
            ret = channel_init(&ch, model_path, weight_path);
            if (ret != 0) {
                printf("channel_init failed for ch%02d, aborting\n", ch.channel_id);
                goto init_failed;
            }
            ret = channel_load_image(&ch, image_path);
            if (ret != 0) {
                printf("channel_load_image failed for ch%02d, aborting\n", ch.channel_id);
                goto init_failed;
            }
        }
    }

    goto init_ok;
init_failed:
    for (auto& board : boards_vec) {
        for (int c = 0; c < board.num_channels; ++c) {
            channel_release(&board.channels[c]);
        }
    }
    deinit_post_process();
    return -1;
init_ok:

    // Install Ctrl-C handler for run-forever mode.
    signal(SIGINT, on_sigint);

    // Spawn all worker threads.
    printf("\n==> spawning %d channels\n", boards * channels_per_board);
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    for (auto& board : boards_vec) {
        for (int c = 0; c < board.num_channels; ++c) {
            channel_context_t& ch = board.channels[c];
            ch.running = true;
            ch.should_stop = false;
            if (pthread_create(&ch.thread, NULL, channel_worker_thread, &ch) != 0) {
                printf("pthread_create failed for ch%02d\n", ch.channel_id);
                ch.running = false;
            }
        }
    }

    // Wait for all channels to finish.
    for (auto& board : boards_vec) {
        for (int c = 0; c < board.num_channels; ++c) {
            channel_context_t& ch = board.channels[c];
            if (ch.running) {
                pthread_join(ch.thread, NULL);
            }
        }
    }
    gettimeofday(&t1, NULL);
    double wall_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0;

    // Report.
    print_report(boards_vec, channels_per_board, wall_ms);

    // Release everything.
    for (auto& board : boards_vec) {
        for (int c = 0; c < board.num_channels; ++c) {
            channel_release(&board.channels[c]);
        }
    }

    deinit_post_process();
    return 0;
}
