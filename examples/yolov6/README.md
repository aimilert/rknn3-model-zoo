## ONNX模型说明

本示例中涉及两种不同的onnx模型，共有三种使用模式：

### 1. 不带 "_rknn3" 后缀的模型（推荐）
文件名示例：`yolov6s.onnx`

**特点：**
- 标准的 YOLOv6 模型，无需对原始模型做特别修改
- 输出为 3 个特征图（stride=8/16/32 的检测头输出）
- 支持两种后处理方式（见下文）

#### 模式 A：配合后处理插件使用（⭐ 推荐单核场景）
**使用方式：**
- 需要配合后处理插件 `libpostprocess_yolov6_rk182x.so` 使用
- 通过 `rknn3_register_custom_ops_plugins()` API 加载插件
- 后处理在 RK182x 协处理器中的 CPU 执行

**优点：**
- ✅ 不需要修改原始 YOLOv6 模型，使用简单
- ✅ 单核性能有较大优势
- ✅ 减少主控端与协处理器之间的数据传输

**缺点：**
- ⚠️后处理在协处理器 CPU 上执行，多核时可能存在CPU资源竞争

**输出格式：** `[N, 256, 6]`，每个检测框包含 6 个值（score, class_id, x1, y1, x2, y2）

#### 模式 B：主控端后处理（不推荐）
**使用方式：**
- 不使用后处理插件
- 输出 3 个特征图直接传输到主控端
- 在主控端 CPU 上进行解码和 NMS

**缺点：**
- ❌ 输出数据量大（3 个特征图）
- ❌ 主控端与协处理器之间数据传输耗时较大
- ❌ 整体性能较差，**不建议使用**

### 2. 带 "_rknn3" 后缀的模型（多核场景优化）
文件名示例：`yolov6s_rknn3.onnx`

**特点：**
- 专门为 RK182x 平台优化的模型
- 将 yolo 后处理（解码、候选框筛选排序以及 NMS 等）作为自定义算子内置到模型中
- 后处理作为模型的一部分，在推理时一并执行

**使用方式：**
- 不需要后处理插件 `libpostprocess_yolov6_rk182x.so`
- 模型后处理已内置，推理时自动执行
- 直接输出检测结果

**优点：**
- ✅ 多核（8核）场景下有性能收益
- ✅ 后处理与推理流水化执行

**缺点：**
- ⚠️ 需要对原始模型进行修改（添加后处理算子）
- ⚠️ 模型导出和转换相对复杂
- ⚠️ 单核场景下性能不如模式 A

**输出格式：** `[N, 512, 6]`，每个检测框包含 6 个值（score, class_id, x1, y1, x2, y2）

### 三种模式对比总结

| 模式 | 模型类型 | 后处理插件 | 后处理位置 | 单核性能 | 多核性能 | 易用性 | 推荐场景 |
|------|---------|-----------|----------|---------|---------|--------|---------|
| **模式 A** | 不带_rknn3 | ✅ 需要 | 协处理器 CPU | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 单核/少核应用 |
| **模式 B** | 不带_rknn3 | ❌ 不用 | 主控端 CPU | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ 不推荐 |
| **模式 C** | 带_rknn3 | ❌ 不用 | 模型内置 | ⭐⭐⭐⭐| ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 8核并行应用 |

**选择建议：**
- 单核或少核场景：优先选择 **模式 A**（不带_rknn3 + 后处理插件）
- 8核并行场景：选择 **模式 C**（带_rknn3 模型）
- 避免使用 **模式 B**（数据传输开销大）

### 模型下载链接
- yolov6n.onnx: `https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov6/yolov6n.onnx`
- yolov6n_rknn3.onnx: `https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov6/yolov6n_rknn3.onnx`
- yolov6s.onnx: `https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov6/yolov6s.onnx`
- yolov6s_rknn3.onnx: `https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov6/yolov6s_rknn3.onnx`

### ONNX 模型导出
完整的 onnx 导出方法可以从以下链接下载完整的 PyTorch 示例工程：
`https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov6/yolov6-postprocess.tar.gz`

下载完成后，参考其中的导出文档，按照步骤进行操作，即可生成适配 RK182x 的优化 ONNX 模型。

## 模型转换

### 1. 标准模型（不带 `_rknn3` 后缀，配合模式 A/B 使用）
```bash
cd python
python convert.py ../model/yolov6n.onnx RK1820 i8
```

### 2. `_rknn3` 优化模型（配合模式 C 使用）
```bash
cd python
python convert.py ../model/yolov6n_rknn3.onnx RK1820 i8
```

注意：
- 由于坐标解码部分不适合量化，因此`convert.py`采用了混合量化模式，具体配置在`rknn.config()`的`subgraph_dtype_target`参数，详细定义见《Rockchip_RKNPU3_API_Reference_RKNN3_Toolkit》
- `convert.py` 中 `core_num` 默认为 `1`，即 `_rknn3` 模型默认按**单核**转换，运行时需使用 `0x01` 与之匹配（详见下文模式 C 说明）。

## C++ 示例编译和运行

### 编译准备

#### 1. 编译后处理插件（仅模式 A 需要）

如果使用**模式 A**（不带_rknn3模型 + 后处理插件），需要编译后处理插件：

```bash
cd examples/yolov5/cpp/libpostprocess_rk182x

# 编译 YOLOv6 后处理插件
./build.sh yolov6
```

编译成功后会生成 `libpostprocess_yolov6_rk182x.so`

**环境要求：**
- RISC-V 交叉编译工具链：`riscv64-unknown-elf-gcc`

**注意：** 模式 C（带_rknn3模型）**不需要**编译此插件

#### 2. 编译主程序

```bash
# 在项目根目录执行
./build-linux.sh -t rk3588 -a aarch64 -d yolov6 -b Release
```

编译完成后，会在安装目录 `install/rk3588_linux_aarch64/rknn_yolov6_demo/`（需推送到板端的即此目录）下生成完整的运行包：
- `rknn_yolov6_demo`：单张图片推理程序
- `dataset_eval`：数据集批量测试程序
- `model/`：模型和测试图片
- `lib/`：依赖库和后处理插件

### 推理运行示例

#### 单张图片推理

**通用命令格式：**
```bash
./rknn_yolov6_demo <model_path> <weight_path> <image_path> <core_mask> [postprocess_plugin_path]
```

**参数说明：**
- `model_path`：RKNN 模型路径
- `weight_path`：模型权重文件路径
- `image_path`：输入图片路径
- `core_mask`：NPU 核心掩码（十六进制）
  - `0x1`：使用 NPU 核心 0（单核）
  - `0x2`：使用 NPU 核心 1（单核）
- `postprocess_plugin_path`：（可选）后处理插件库路径

**模式 A：不带_rknn3模型 + 后处理插件（⭐ 推荐单核）**
```bash
# 使用后处理插件，后处理在协处理器 CPU 执行
./rknn_yolov6_demo model/yolov6n.rknn model/yolov6n.weight model/bus.jpg 0x1 lib/libpostprocess_yolov6_rk182x.so
```

**模式 B：不带_rknn3模型，主控端后处理（❌ 不推荐）**
```bash
# 不使用插件，后处理在主控端 CPU 执行
./rknn_yolov6_demo model/yolov6n.rknn model/yolov6n.weight model/bus.jpg 0x1
```

**模式 C：带_rknn3模型（⭐ 推荐多核）**
```bash
# 使用内置后处理的模型，不需要插件
./rknn_yolov6_demo model/yolov6n_rknn3.rknn model/yolov6n_rknn3.weight model/bus.jpg 0x1
```

**核心代码说明（模式 A）：**

在 `yolov6.cc` 的 `init_yolov6_model()` 函数中，通过传入 `postprocess_plugin_path` 参数来启用后处理插件：

```cpp
if (postprocess_plugin_path != NULL) {
    app_ctx->use_postprocess_plugin = true;

    // 注册后处理插件到 RKNN 运行时
    // 插件将在协处理器的 CPU 上执行后处理
    ret = rknn3_register_custom_ops_plugins(ctx, postprocess_plugin_path,
                                            strlen(postprocess_plugin_path));
    if (ret != RKNN3_SUCCESS) {
        printf("rknn3_register_custom_ops_plugins failed! ret=%d\n", ret);
        rknn3_destroy(ctx);
        return -1;
    }
    printf("rknn3_register_custom_ops_plugins success\n");
}
```

**重要说明：**
- 后处理插件 `libpostprocess_yolov6_rk182x.so` 配合**不带_rknn3后缀**的模型使用
- 使用后处理插件后，需要使用专用的查询命令：
  - 查询输入输出数量：`RKNN3_QUERY_POSTPROCESS_IN_OUT_NUM`
  - 查询输出属性：`RKNN3_QUERY_POSTPROCESS_OUTPUT_ATTR`
- 后处理在 RK182x 协处理器的 CPU 上执行，无需主控端参与

#### 输出结果

程序会在当前目录生成 `out.png`，其中包含检测框和类别标签。

控制台输出示例：
```
person @ (211 241 285 510) 0.947
bus @ (98 136 553 433) 0.944
person @ (476 231 560 521) 0.925
person @ (107 235 225 536) 0.921
person @ (79 328 120 516) 0.534
tie @ (160 282 168 299) 0.350
stop sign @ (79 149 100 192) 0.298
handbag @ (173 356 195 410) 0.257
```

## 数据集精度测试说明(可选)

测试数据集：COCO val2017
测试集数量：5000 张图片

### 图片预处理

为了排除前处理不一致导致的精度变化，对测试数据集先做预处理后再进行精度测试。
执行下列指令后，将在datasets/COCO目录下得到:

- val2017: 原始图片集
- annotations: 标注文件
- val2017_preprocess: 预处理后图片集
- coco_dataset_test_path.txt: 板端测试图片集路径列表文件

```bash
cd datasets/COCO/
python download_eval_dataset.py
python data_preprocess.py
```

### ONNX 数据集测试

进入demo目录运行推理脚本，结束后将得到result_onnx.json结果文件
```bash
cd examples/yolov6/python
python onnx_infer.py
```

### 板端数据集测试

#### 1. 准备测试数据

将预处理后图片集和相关文件推至板端：
```bash
adb push datasets/COCO/val2017_preprocess /userdata/
adb push datasets/COCO/coco_dataset_test_path.txt /userdata/
```

#### 2. 编译和部署

**编译主程序：**
```bash
./build-linux.sh -t rk3588 -a aarch64 -d yolov6 -b Release
```

**编译后处理插件（仅模式 A 需要）：**
```bash
# 如果使用模式 A（不带_rknn3模型 + 后处理插件），需要编译插件
cd examples/yolov5/cpp/libpostprocess_rk182x
./build.sh yolov6
cd -
```

**将可执行文件和模型文件推入板端：**

**推送整个安装目录**
```bash
# 推送整个安装包到板端
adb push install/rk3588_linux_aarch64/rknn_yolov6_demo /userdata/

# 推送测试数据集路径文件
adb push datasets/COCO/coco_dataset_test_path.txt /userdata/
```

#### 3. 运行推理

进入板端执行推理：
```bash
adb shell
cd /userdata/rknn_yolov6_demo
```

**设置库路径（如果使用方式1推送整个目录）：**
```bash
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
```

**模式 A：不带_rknn3模型 + 后处理插件（推荐单核）**
```bash
# 使用后处理插件（在协处理器端执行后处理，减少数据传输）
./dataset_eval model/yolov6n.rknn model/yolov6n.weight 0x1 lib/libpostprocess_yolov6_rk182x.so
```

**模式 B：不带_rknn3模型，主控端后处理（不推荐）**
```bash
# 不使用插件，主控端后处理（数据传输开销大）
./dataset_eval model/yolov6n.rknn model/yolov6n.weight 0x1
```

**模式 C：带_rknn3模型（推荐多核）**
```bash
# 使用内置后处理的模型（与默认 core_num=1 转换匹配，使用单核 0x01 运行）
./dataset_eval model/yolov6n_rknn3.rknn model/yolov6n_rknn3.weight 0x01
```
> ⚠️ `convert.py` 中 `core_num` 默认为 `1`，即 `_rknn3` 模型按单核转换，因此运行时使用 `0x01` 与之匹配。若需以 8 核（`0xFF`）运行，请先将 `convert.py` 中的 `core_num` 改为 `8` 重新转换模型。

推理结束后会在当前目录下生成 `results_rknn.json` 结果文件。

**参数说明：**
- 第一个参数：RKNN 模型文件路径
- 第二个参数：权重文件路径
- 第三个参数：NPU 核心掩码（十六进制），例如：
  - `0x1`：使用 NPU 核心 0（单核）
  - `0x2`：使用 NPU 核心 1（单核）
- 第四个参数（可选）：后处理插件路径，用于模式 A

#### 4. 获取测试结果

```bash
# 从板端拉取结果文件
adb pull /userdata/rknn_yolov6_demo/results_rknn.json ./
```

### 数据集精度评估

```bash
cd datasets/COCO
python eval_dataset.py --result_json /path/to/result.json
```

## 后处理插件详细说明（RK182x 专用，模式 A）

### 什么是后处理插件？

后处理插件 `libpostprocess_yolov6_rk182x.so` 是 RKNN3 平台的自定义算子扩展机制，用于**模式 A**（不带_rknn3模型 + 后处理插件）。

**工作原理：**
- 配合**不带_rknn3后缀**的标准 YOLOv6 模型使用
- 通过 `rknn3_register_custom_ops_plugins()` API 动态加载到 RKNN 运行时
- 后处理在 RK182x 协处理器中的 CPU 上执行（而非主控端 CPU）
- 接收模型输出的 3 个特征图，在协处理器端完成解码和 NMS

**优势：**
- ✅ 无需修改原始 YOLOv6 模型结构
- ✅ 减少主控端与协处理器之间的大量数据传输
- ✅ 单核场景下性能最优

### 插件功能特性

- ✅ 支持 YOLOv6 目标检测后处理
- ✅ 支持 INT8 量化和 FP16 浮点两种数据类型
- ✅ 内置 NMS（非极大值抑制）处理

### 插件目录结构

```
examples/yolov5/cpp/libpostprocess_rk182x/
├── rknn3_custom_op.c          # 插件主入口，实现 rknn3_custom_op 接口
├── yolov8_postprocess.c       # YOLOv6/YOLOv8 后处理核心算法
├── postprocess.h              # 公共头文件和数据结构定义
├── build.sh                   # 编译脚本
├── clean.sh                   # 清理脚本
└── README.md                  # 详细使用说明
```

### 编译环境要求

- RISC-V 交叉编译工具链：`riscv64-unknown-elf-gcc`
- 编译下载链接参考：`examples/yolov5/cpp/libpostprocess_rk182x/README.md`


### 编译步骤

```bash
cd examples/yolov5/cpp/libpostprocess_rk182x

# 编译 YOLOv6 后处理插件
./build.sh yolov6

# 清理编译产物
./clean.sh
```

编译成功后会生成 `libpostprocess_yolov6_rk182x.so` 动态库文件。

### 插件参数配置

后处理插件的默认参数定义在 `postprocess.h` 中：

```c
#define MAX_OBJ_NUM     256     // 最大检测目标数
#define OBJ_CLASS_NUM   80      // COCO 数据集类别数
#define NMS_THRESH      0.45    // NMS 阈值
#define BOX_THRESH      0.25    // 置信度阈值
```

如需调整参数，修改头文件后重新编译即可。

### 插件输入输出格式

**输入张量（来自 YOLOv6 的三个检测头）：**


**输出张量：**

| 输出 | Shape | 说明 |
|------|-------|------|
| output0 | [N, 256, 6] | 检测结果 |

输出数据格式（每个检测框 6 个值）：
- `[0]`：置信度分数
- `[1]`：类别 ID (0-79)
- `[2]`：边界框左上角 x1 坐标
- `[3]`：边界框左上角 y1 坐标
- `[4]`：边界框右下角 x2 坐标
- `[5]`：边界框右下角 y2 坐标


### API 使用说明

**核心 API：`rknn3_register_custom_ops_plugins()`**

```c
int rknn3_register_custom_ops_plugins(
    rknn3_context ctx,              // RKNN 上下文
    const char* plugin_path,        // 插件库文件路径
    size_t plugin_path_len          // 路径长度
);
```

**返回值：**
- `RKNN3_SUCCESS (0)`：注册成功
- 其他值：注册失败

**使用示例（模式 A）：**

```cpp
#include "rknn3_api.h"

// 步骤 1: 初始化 RKNN 上下文
rknn3_context ctx;
rknn3_init(&ctx, NULL);

// 步骤 2: 加载不带_rknn3后缀的标准模型
rknn3_load_model_from_path(ctx, "yolov6s.rknn", "yolov6s.weight");

// 步骤 3: 初始化模型
rknn3_config config;
memset(&config, 0, sizeof(config));
config.run_core_mask = 0x1;  // 单核
rknn3_model_init(ctx, &config);

// 步骤 4: 注册后处理插件（关键步骤）
const char* plugin_path = "libpostprocess_yolov6_rk182x.so";
int ret = rknn3_register_custom_ops_plugins(ctx, plugin_path, strlen(plugin_path));
if (ret != RKNN3_SUCCESS) {
    printf("Failed to register custom ops plugins: %d\n", ret);
    return -1;
}
printf("Plugin registered, postprocess will run on coprocessor CPU\n");

// 步骤 5: 使用专用查询命令获取输入输出信息
rknn3_input_output_num io_num;
ret = rknn3_query(ctx, RKNN3_QUERY_POSTPROCESS_IN_OUT_NUM, &io_num, sizeof(io_num));

// 查询输出属性（插件处理后的输出）
rknn3_tensor_attr output_attr;
output_attr.index = 0;
ret = rknn3_query(ctx, RKNN3_QUERY_POSTPROCESS_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
// 输出格式: [N, 256, 6] - 已完成后处理的检测结果
```

**注意事项：**
1. ⚠️ **插件配合不带_rknn3后缀的模型使用**（如 `yolov6s.rknn`，而非 `yolov6s_rknn3.rknn`）
2. 必须在 `rknn3_model_init()` 之后调用
3. 必须在第一次 `rknn3_run()` 之前调用
4. 使用插件后，查询命令需要使用专用的 `RKNN3_QUERY_POSTPROCESS_*` 系列命令
5. 插件文件需要确保可访问且与 RISC-V 平台架构匹配
6. 后处理在协处理器 CPU 上执行，主控端无需参与后处理计算


### 三种模式性能对比

| 性能指标 | 模式 A<br>不带_rknn3+插件 | 模式 B<br>主控端后处理 | 模式 C<br>带_rknn3模型 |
|---------|------------------------|---------------------|-------------------|
| 单核推理速度 | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ |
| 多核(8核)速度 | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| 数据传输开销 | 极小 | 较大 | 极小 |
| 主控端 CPU 负载 | 无 | 较高 | 无 |
| 模型准备难度 | 简单 | 简单 | 中等 |

### 调试技巧

**启用调试输出：**

在插件源码中取消注释 `printf` 语句可获取详细的调试信息：

```c
// 在 yolov8_postprocess.c 中取消注释
printf("[PostProcess] Total objects before NMS: %d\n", validCount);
printf("[PostProcess] Result[%d]: box=(%.1f, %.1f, %.1f, %.1f), score=%.3f, class=%d\n", ...);
```

**常见问题排查：**

1. **插件加载失败 (`rknn3_register_custom_ops_plugins` 返回错误)**：
   - 检查插件文件路径是否正确
   - 验证插件是否为 RISC-V 架构编译（使用 `file` 命令检查）
   - 确认是否使用了**不带_rknn3后缀**的模型

2. **模型与插件不匹配**：
   - ✅ 正确：`yolov6s.rknn` + `libpostprocess_yolov6_rk182x.so`
   - ❌ 错误：`yolov6s_rknn3.rknn` + 插件（带_rknn3模型不需要插件）

3. **无检测结果**：
   - 检查置信度阈值 `BOX_THRESH` 是否过高（默认 0.25）
   - 确认输入图片预处理正确（letterbox + RGB格式）
   - 打印输出张量形状，确认是否为 `[N, 256, 6]`

4. **检测框重叠**：
   - 调整 NMS 阈值 `NMS_THRESH`（默认 0.45）
   - 减小阈值可增加抑制强度（如改为 0.35）

5. **性能不达预期**：
   - 单核场景：确认使用**模式 A**（不带_rknn3 + 插件）
   - 多核(8核)场景：建议使用**模式 C**（带_rknn3模型）
   - 避免使用**模式 B**（主控端后处理，数据传输开销大）

6. **推理速度慢**：
   - 检查是否误用了模式 B（主控端后处理）
   - 验证插件是否正确加载（查看日志中的 "rknn3_register_custom_ops_plugins success"）

### 进一步参考

详细的插件开发和使用说明，请参考：
- `examples/yolov5/cpp/libpostprocess_rk182x/README.md`
- RKNN3 API 文档

