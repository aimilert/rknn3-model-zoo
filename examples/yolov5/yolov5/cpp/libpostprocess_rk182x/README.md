# RKNN3 YOLOv5 YOLOv6 YOLOv8 后处理插件使用说明

## 概述

本插件为 RKNN3 平台提供 YOLOv5、YOLOv6、YOLOv8 目标检测模型的后处理功能。它实现了 `rknn3_custom_op` 接口，可作为自定义算子动态加载到 RKNN3 运行时中，在 RK182x 芯片的 C908 RISC-V 核心上执行后处理计算（包括边界框解码、置信度过滤、NMS 等）。

**关键优势：**
- 无需在主 CPU 上执行后处理，减轻主 CPU 负担
- 统一的插件接口，支持多个 YOLO 版本
- 包含预编译库，开箱即用

**限制：**
- ⚠️ **插件必须使用纯 C 语言实现**，不支持 C++ 特性
- 插件接口和实现必须符合 C 标准（不能使用 C++ 类、模板、命名空间等）

## 功能特性

- ✅ 支持 YOLOv5 YOLOv6 YOLOv8 目标检测后处理
- ✅ 支持 INT8 量化、FP16 和 FP32 浮点数据类型
- ✅ 内置 NMS（非极大值抑制）处理
- ✅ 在 RK182x 芯片的 CPU 核心上执行

## 文件结构

```
libpostprocess_rk182x/
├── rknn3_custom_op.c       # 插件主入口，实现 rknn3_custom_op 接口
├── yolov5_postprocess.c    # YOLOv5 后处理核心算法（使用 Anchor）
├── yolov8_postprocess.c    # YOLOv8/v6 后处理核心算法（使用 DFL）
├── postprocess.h           # 公共头文件和数据结构定义
├── build.sh                # 编译脚本
├── build_all.sh            # 编译所有版本的脚本
├── clean.sh                # 清理编译产物
├── prebuilt/               # 预编译的动态库
│   ├── libpostprocess_yolov5_rk182x.so
│   ├── libpostprocess_yolov6_rk182x.so
│   └── libpostprocess_yolov8_rk182x.so
└── README.md               # 本说明文档
```

## 编译方法

### 环境要求

- RISC-V 交叉编译工具链：`riscv64-unknown-elf-gcc`
- **编译器要求**：必须使用 C 编译器（gcc），不支持 C++ 编译器（g++）
- **语言标准**：插件代码必须符合 C 语言标准（C99/C11），不能使用 C++ 特性
- **编译器下载地址**： RK182X-GCC https://console.zbox.filez.com/l/103Dro 提取码: rknn


### 编译步骤

```bash
# 进入插件目录
cd examples/yolov5/cpp/libpostprocess_rk182x

# 编译单个版本
./build.sh yolov5   # 编译 YOLOv5 版本
./build.sh yolov8   # 编译 YOLOv8 版本
./build.sh yolov6   # 编译 YOLOv6 版本

# 或一次性编译所有版本
./build_all.sh
```

编译成功后会生成对应的动态库文件：
- `libpostprocess_yolov5_rk182x.so`
- `libpostprocess_yolov8_rk182x.so`
- `libpostprocess_yolov6_rk182x.so`

**注意**：
- 如果不想自行编译，可直接使用 `prebuilt/` 目录下的预编译库。
- `build.sh` 会根据参数设置相应的编译宏：
  - `yolov5` → 定义 `ENABLE_YOLOV5_POSTPROCESS`
  - `yolov8` 或 `yolov6` → 定义 `ENABLE_YOLOV8_POSTPROCESS`

## 工作原理

插件采用 RKNN3 的自定义算子机制，通过以下流程工作：

1. **加载阶段**：应用调用 `rknn3_load_custom_ops()` 加载插件动态库
2. **注册阶段**：RKNN3 运行时调用 `rknn3_register_custom_ops_plugin()` 获取算子列表
3. **初始化阶段**：调用 `init` 回调，初始化后处理参数（阈值、模型尺寸等）
4. **推理阶段**：每次推理后，RKNN3 自动调用 `compute` 回调执行后处理
5. **清理阶段**：调用 `deinit` 回调释放资源

插件从模型输出（原始特征图）接收数据，执行以下操作：
- **解码边界框**：YOLOv5 使用 Anchor，YOLOv8/v6 使用 DFL
- **类别预测**：提取最高分数的类别
- **置信度过滤**：过滤低于阈值的检测
- **NMS 去重**：对每个类别应用非极大值抑制
- **输出格式化**：生成统一的检测结果格式

## 插件接口说明

### 注册函数

```c
rknn3_custom_op* rknn3_register_custom_ops_plugin(int op_index);
```

RKNN3 运行时通过此函数获取插件中注册的自定义算子。

**参数:**
- `op_index`: 算子索引，从 0 开始

**返回值:**
- 返回 `rknn3_custom_op` 结构体指针
- 当 `op_index` 超出范围时返回 `NULL`

### 插件结构体

```c
static rknn3_custom_op yolo_postprocess_op = {
    .op_type = "YoloPostprocess",           // 算子类型名称
    .plugin_type = RKNN3_OP_PLUGIN_TYPE_POSTPROCESS,  // 后处理插件
    .target = RKNN3_OP_TARGET_TYPE_CPU,     // 在 CPU 上执行
    .version = 1,
    .author = "Rockchip",
    .description = "YOLO Postprocess Plugin for RKNN3",
    
    // 回调函数
    .init = yolo_postprocess_plugin_init,
    .prepare = NULL,
    .compute = yolo_postprocess_plugin_compute,
    .deinit = yolo_postprocess_plugin_deinit,
    
    // 后处理专用接口
    .get_output_num = yolo_postprocess_plugin_get_output_num,
    .get_attrs = yolo_postprocess_plugin_get_attrs,
};
```

## 输入输出规格

### 输出张量

| 输出 | Shape | 说明 |
|------|-------|------|
| output0 | [N, 256, 6] | 检测结果 |

输出数据格式（每个检测框 6 个 float 值）：

```c
typedef struct {
    float score;    // 置信度分数
    float class_id; // 类别 ID (0-79)
    box_t box;      // 边界框坐标
} object_detect_result;

typedef struct {
    float x1;       // 边界框左上角 x 坐标
    float y1;       // 边界框左上角 y 坐标
    float x2;       // 边界框右下角 x 坐标
    float y2;       // 边界框右下角 y 坐标
} box_t;
```

**注意**：输出坐标为 (x1, y1, x2, y2) 格式，不是 (x, y, w, h) 格式。

## 默认参数

```c
#define MAX_OBJ_NUM     256     // 最大检测目标数
#define OBJ_CLASS_NUM   80      // COCO 数据集类别数
#define NMS_THRESH      0.45    // NMS 阈值
#define BOX_THRESH      0.25    // 置信度阈值
```

### YOLOv5 Anchor 配置

YOLOv5 使用固定的 Anchor 配置，YOLOv8/v6 不使用 Anchor（采用 Anchor-free 设计）：

```c
const int anchor[3][6] = {
    {10, 13, 16, 30, 33, 23},     // stride=8
    {30, 61, 62, 45, 59, 119},    // stride=16
    {116, 90, 156, 198, 373, 326} // stride=32
};
```

### YOLOv8 DFL (Distribution Focal Loss) 参数

YOLOv8 使用 DFL 进行边界框预测，默认 `dfl_len=16`，即每个边界框坐标由 16 个分布值表示。

## 使用示例

### 1. 加载插件

在 RKNN3 应用中加载后处理插件：

```c
#include "rknn3_api.h"

// 初始化 RKNN 上下文
rknn3_context ctx;
rknn3_init(&ctx, model_path, 0, RKNN3_FLAG_PRIOR_HIGH, NULL);

// 根据模型类型加载相应的插件
// YOLOv5 模型使用:
rknn3_load_custom_ops(ctx, "libpostprocess_yolov5_rk182x.so");

// YOLOv8 模型使用:
// rknn3_load_custom_ops(ctx, "libpostprocess_yolov8_rk182x.so");

// YOLOv6 模型使用:
// rknn3_load_custom_ops(ctx, "libpostprocess_yolov6_rk182x.so");

// 正常运行推理，后处理会自动执行
rknn3_run(ctx, NULL);
```

**重要**：必须根据实际使用的模型类型选择对应的插件库，因为不同版本的输入输出格式不同。

### 2. 解析输出结果

```c
// 获取输出
rknn3_output outputs[1];
rknn3_outputs_get(ctx, 1, outputs, NULL);

// 解析检测结果
object_detect_result *results = (object_detect_result *)outputs[0].buf;

for (int i = 0; i < MAX_OBJ_NUM; i++) {
    if (results[i].score <= 0) break;  // 无效结果
    
    printf("检测到目标: class=%d, score=%.2f, box=(%.1f, %.1f, %.1f, %.1f)\n",
           (int)results[i].class_id,
           results[i].score,
           results[i].box.x1,
           results[i].box.y1,
           results[i].box.x2,
           results[i].box.y2);
}

// 释放输出
rknn3_outputs_release(ctx, 1, outputs);
```

## 扩展与自定义

### 添加新的自定义算子

1. 创建新的算子结构体（纯 C 实现）：

```c
static rknn3_custom_op my_new_op = {
    .op_type = "MyNewOp",
    .plugin_type = RKNN3_OP_PLUGIN_TYPE_POSTPROCESS,
    // ... 其他配置
};
```

2. 在 `registered_ops` 数组中注册：

```c
static rknn3_custom_op* registered_ops[] = {
    &yolo_postprocess_op,
    &my_new_op,           // 添加新算子
    NULL  // 数组结尾标记
};
```

### 修改检测参数

可以通过修改 `postprocess.h` 中的宏定义来调整检测参数：

```c
#define MAX_OBJ_NUM   512    // 增加最大检测数
#define OBJ_CLASS_NUM 91     // 自定义类别数
#define NMS_THRESH    0.50   // 调整 NMS 阈值
#define BOX_THRESH    0.30   // 调整置信度阈值
```

## 调试技巧

### 启用调试输出

取消注释 `yolov5_postprocess.c` 或 `yolov8_postprocess.c` 中的 `printf` 语句可获取详细的调试信息：

```c
// 取消注释以下行启用调试
// printf("[PostProcess] Total objects before NMS: %d\n", validCount);
// printf("[PostProcess] Result[%d]: box=(%.1f, %.1f, %.1f, %.1f), score=%.3f, class=%d\n", 
//        result_count, x, y, w, h, score, cls_id);
// printf("[PostProcess] Top 5 scores after sorting: ");
// printf("[PostProcess] Found %d unique classes, applying NMS with threshold=%.2f\n", 
//        unique_count, nms_threshold);
```

插件的 `rknn3_custom_op.c` 会在初始化时自动打印模型输入输出信息，有助于排查问题。

### 常见问题排查

1. **无检测结果**
   - 检查置信度阈值 `BOX_THRESH` 是否过高（默认 0.25）
   - 确认模型输出数据类型与插件支持的类型匹配
   - 查看插件初始化日志，确认模型输入尺寸正确识别

2. **检测框重叠**
   - 调低 NMS 阈值 `NMS_THRESH`（默认 0.45）
   - 检查是否正确加载了对应的插件版本

3. **类别错误**
   - 确认 `OBJ_CLASS_NUM` 与模型训练时一致（默认 80，COCO 数据集）
   - 如使用自定义类别数，需要修改 `postprocess.h` 并重新编译

4. **插件加载失败**
   - 检查动态库路径是否正确
   - 确认使用了正确的 RISC-V 交叉编译工具链
   - 查看是否有依赖库缺失
   - **确认插件是使用 C 语言编译的，不是 C++**（检查 .c 文件是否被 g++ 编译）

5. **输入张量不匹配**
   - YOLOv5 期望 3 个输入（255 通道）
   - YOLOv8/v6 期望 6-9 个输入（box, score, score_sum）
   - 确保模型输出格式与插件期望一致

6. **坐标异常**
   - 注意输出格式是 (x1, y1, x2, y2)，不是 (x, y, w, h)
   - 检查是否需要进行 letterbox 坐标逆变换

7. **性能问题**
   - 可以减少 `MAX_OBJ_NUM` 降低内存占用
   - 提高 `BOX_THRESH` 减少需要处理的候选框
   - 确保编译时使用了优化选项 `-O2`

## 快速参考

### 配置参数速查

| 参数 | 默认值 | 说明 | 位置 |
|------|--------|------|------|
| `MAX_OBJ_NUM` | 256 | 最大检测目标数 | `postprocess.h` |
| `OBJ_CLASS_NUM` | 80 | 类别数（COCO） | `postprocess.h` |
| `NMS_THRESH` | 0.45 | NMS 阈值 | `postprocess.h` |
| `BOX_THRESH` | 0.25 | 置信度阈值 | `postprocess.h` |

### 输出格式速查

```c
// 访问第 i 个检测结果
object_detect_result *result = results + i;
float score = result->score;           // 置信度
int class_id = (int)result->class_id;  // 类别 ID
float x1 = result->box.x1;             // 左上角 x
float y1 = result->box.y1;             // 左上角 y
float x2 = result->box.x2;             // 右下角 x
float y2 = result->box.y2;             // 右下角 y
```

