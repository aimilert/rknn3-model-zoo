# RK1828 多板多路 YOLOv5 性能测试 Demo

在 **4 块 RK1828** 上分别运行 **32 路视频**（每块 8 核 × 每核 1 路）的并行推理性能测试。

## 1. 架构设计

```
4 块 RK1828 (USB/NTB 连接到主控)
│
├── board0 (dev_id0)          board1 (dev_id1)          board2 (dev_id2)          board3 (dev_id3)
│   ├── core0  → ch00 线程      ├── core0  → ch08 线程      ├── core0  → ch16 线程      ├── core0  → ch24 线程
│   ├── core1  → ch01 线程      ├── core1  → ch09 线程      ├── core1  → ch17 线程      ├── core1  → ch25 线程
│   ├── ...                    ├── ...                    ├── ...                    ├── ...
│   └── core7  → ch07 线程      └── core7  → ch15 线程      └── core7  → ch23 线程      └── core7  → ch31 线程
```

- **每路 = 1 个独立 `rknn3_context`**：通过 `rknn3_init_extend.device_id` 绑定到指定板卡，
  通过 `config.run_core_mask = 1 << core_id` 绑定到指定核（单 bit，`0x01`~`0x80`）。
- **每路 = 1 个独立 pthread**：各自循环推理，互不阻塞。
- **模型使用模式 C（`_rknn3`）**：后处理（解码 + NMS）内置为模型算子，在 NPU 上执行。
  避免模式 A 的后处理插件在协处理器 CPU 上产生 8 核竞争。

## 2. 模型准备（关键）

必须使用带 `_rknn3` 后缀的模型，且**保持 `core_num=1`（默认值）转换**：

```bash
cd examples/yolov5/python
# convert.py 中 core_num 保持默认 1 —— 每路单核独立推理，不要改成 8
python convert.py ../model/yolov5s_rknn3.onnx RK1820 i8
```

> ⚠️ 不要修改 `convert.py` 里的 `core_num`。`core_num=1` 表示"每个实例使用 1 个核"，
> 运行时用 8 个独立 context 分别绑定 8 个不同核，即可实现 8 路并行。
> 若改成 `core_num=8` 则是"单实例用 8 核联合加速"，不适用于本场景。

## 3. 编译

复用官方 yolov5 的构建流程，`CMakeLists.txt` 已加入新目标 `rknn_multicam_demo`：

```bash
cd rknn3_model_zoo/
./build-linux.sh -t rk3588 -a aarch64 -d yolov5 -b Release
```

安装目录 `install/rk3588_linux_aarch64/rknn_yolov5_demo/` 下会生成 `rknn_multicam_demo`。

## 4. 运行

推送安装目录到主控（若主控本身就是板子则直接使用）：

```bash
adb push install/rk3588_linux_aarch64/rknn_yolov5_demo /data/
```

运行 4 板 × 8 路 = 32 路测试（每路跑 200 帧）：

```bash
cd /data/rknn_yolov5_demo
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH

taskset f0 ./rknn_multicam_demo \
    --model model/yolov5s_rknn3.rknn \
    --weight model/yolov5s_rknn3.weight \
    --image model/bus.jpg \
    --boards 4 --channels 8 --frames 200
```

参数说明：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | 必填 | RKNN 模型路径（`_rknn3` 模式 C 模型） |
| `--weight` | 必填 | RKNN 权重路径 |
| `--image` | 必填 | 输入图片（每路复用同一张；真实视频场景请替换为解码器输入） |
| `--boards` | 4 | 使用的板卡数（≤4） |
| `--channels` | 8 | 每板通道数（≤8，即核数） |
| `--frames` | 200 | 每路推理帧数；0 = 一直运行直到 Ctrl-C |
| `--fps` | 0 | 每路节流帧率（模拟实时视频）；0 = 全速跑满 |
| `--device-id` | 自动 | 显式指定板卡设备 ID，用 `#` 分隔，如 `id0#id1#id2#id3` |

## 5. 输出报告解读

运行结束后打印 32 路各自的统计 + 汇总：

```
 Channel | Board | Core |  Frames |  Avg Loop(ms) | Min(ms) | Max(ms) | Infer(ms) |  FPS
----------------------------------------------------------------------------------------
  ch00    |  b0   |  0   |     200 |         12.34 |   10.1  |   15.8  |      8.90 |  81.0
  ...
Summary:
  Aggregate throughput: 2592.0 FPS (all channels combined)
```

- **Avg Loop(ms)**：单帧端到端时延（预处理 + 推理 + 后处理），是决定单路能否实时（≤40ms @25fps）的关键。
- **Infer(ms)**：NPU 推理耗时。
- **FPS**：单路吞吐。
- **Aggregate throughput**：32 路合计吞吐（≈ 单路 FPS × 32）。
- 若各核 FPS 明显低于单核独立测试值，说明存在跨核资源竞争（如 DDR 带宽），可对比 `--channels 1` 与 `--channels 8` 的结果。

## 6. 关键代码位置

| 文件 | 作用 |
|------|------|
| [multi_camera_demo.cc](cpp/multi_camera_demo.cc) | 主程序：发现板卡 → 初始化 4×8 路 → 并行推理 → 汇总报告 |
| [channel_worker.cc](cpp/channel_worker.cc) | 单路推理线程：绑定设备/核、循环推理、计时统计 |
| [channel_worker.h](cpp/channel_worker.h) | 通道/板卡数据结构 |
| [yolov5.cc](cpp/yolov5.cc) | 新增 `init_yolov5_model_ex`（设备绑定 + 静默）与推理计时 |
| [yolov5.h](cpp/yolov5.h) | 新增 `quiet` 字段、`rknn_yolov5_times` 计时结构 |

## 7. 注意事项

1. **每路独立 context**：32 路 = 32 个 `rknn3_context`，每路加载一次模型文件。
   同一板卡上 8 个 context 的 `device_id` 相同、`core_mask` 各不相同。
2. **核掩码必须单 bit**：`0x01, 0x02, 0x04, ..., 0x80`，与 `core_num=1` 转换的模型匹配。
3. **设备发现**：程序通过 `rknn3_find_devices` 自动发现，按顺序取前 N 个；
   若板卡接入顺序不确定，用 `--device-id` 显式指定。
4. **主控线程数**：32 个 worker 线程 + 主线程，建议主控 CPU 核心充足（或配合 `taskset`）。
5. **真实视频输入**：当前用静态图片打满循环；接入 V4L2/RTSP 时，替换
   `channel_worker_thread` 中 `inference_yolov5_model` 的输入源即可，帧节流可用 `--fps` 模拟。
