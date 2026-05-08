# jpy-ml CLI 使用指南

## 概述

jpy-ml 提供了命令行界面（CLI），让您可以通过终端直接使用各种 ML 功能，无需编写 Java 代码。

## 安装与构建

### 1. 构建 CLI JAR

```bash
mvn package -Pcli -DskipTests
```

构建完成后，CLI JAR 文件位于 `target/jpy-ml-cli.jar`。

### 2. 使用 CLI 脚本

项目提供了 `bin/jpy-ml` 脚本，可以直接使用：

```bash
./bin/jpy-ml --help
```

或者直接使用 JAR 文件：

```bash
java -jar target/jpy-ml-cli.jar --help
```

## 命令列表

### 基础推理命令

#### 1. predict - 目标检测/分割/分类

```bash
# 目标检测
./bin/jpy-ml predict -m yolov8n.pt -s photo.jpg

# 批量检测
./bin/jpy-ml predict -m yolov8n.pt -s photo1.jpg photo2.jpg photo3.jpg

# 指定置信度阈值
./bin/jpy-ml predict -m yolov8n.pt -s photo.jpg --conf 0.5

# 保存标注图像
./bin/jpy-ml predict -m yolov8n.pt -s photo.jpg -o output/
```

#### 2. segment - SAM 分割

```bash
# SAM2 点提示分割
./bin/jpy-ml segment -m sam2.1_t.pt -i photo.jpg -p 320,240

# SAM2 框提示分割
./bin/jpy-ml segment -m sam2.1_t.pt -i photo.jpg -b 100,100,400,400

# SAM3 文本提示分割
./bin/jpy-ml segment -m sam3.pt -i photo.jpg --text "person" --sam3
```

#### 3. classify - 图像分类

```bash
# 图像分类
./bin/jpy-ml classify -m yolov8n-cls.pt -s photo.jpg

# 显示前 10 个结果
./bin/jpy-ml classify -m yolov8n-cls.pt -s photo.jpg --top-k 10
```

#### 4. pose - 姿态估计

```bash
# 人体姿态估计
./bin/jpy-ml pose -m yolov8n-pose.pt -s person.jpg
```

#### 5. obb - 旋转框检测

```bash
# 旋转框目标检测
./bin/jpy-ml obb -m yolov8n-obb.pt -s satellite.jpg
```

#### 6. video - 视频/摄像头推理

```bash
# 视频文件推理
./bin/jpy-ml video -m yolov8n.pt -s video.mp4

# 摄像头推理（摄像头 0）
./bin/jpy-ml video -m yolov8n.pt -s 0

# 显示 FPS
./bin/jpy-ml video -m yolov8n.pt -s 0 --show-fps

# 限制最大帧数
./bin/jpy-ml video -m yolov8n.pt -s video.mp4 --max-frames 100
```

#### 7. batch - 批量推理

```bash
# 批量处理目录中的图像
./bin/jpy-ml batch -m yolov8n.pt -d ./images

# 指定输出目录
./bin/jpy-ml batch -m yolov8n.pt -d ./images -o ./results

# 指定图像扩展名
./bin/jpy-ml batch -m yolov8n.pt -d ./images --ext jpg,png
```

### 训练命令

#### 8. yolo-train - YOLO 模型训练

```bash
# 基础训练
./bin/jpy-ml yolo-train -m yolov8n.pt -d coco128.yaml

# 指定训练参数
./bin/jpy-ml yolo-train -m yolov8n.pt -d coco128.yaml \
    --epochs 50 \
    --batch-size 16 \
    --img-size 640 \
    --device 0

# 使用 GPU 训练
./bin/jpy-ml yolo-train -m yolov8n.pt -d coco128.yaml --device 0,1

# 恢复训练
./bin/jpy-ml yolo-train -m yolov8n.pt -d coco128.yaml --resume
```

#### 9. train - LLM 微调

```bash
# 基础微调
./bin/jpy-ml train -m Qwen/Qwen2.5-0.5B-Instruct -d training_data.jsonl

# 指定微调参数
./bin/jpy-ml train -m Qwen/Qwen2.5-0.5B-Instruct -d training_data.jsonl \
    --epochs 3 \
    --rank 16 \
    --alpha 32 \
    --batch-size 4 \
    --lr 2e-4

# 指定输出目录
./bin/jpy-ml train -m Qwen/Qwen2.5-0.5B-Instruct -d training_data.jsonl --output ./my_adapter
```

### 验证与导出命令

#### 10. validate - 模型验证

```bash
# 验证模型性能
./bin/jpy-ml validate -m yolov8n.pt -d coco128.yaml

# 指定验证参数
./bin/jpy-ml validate -m yolov8n.pt -d coco128.yaml \
    --batch-size 32 \
    --img-size 640 \
    --device 0
```

#### 11. export - 模型导出

```bash
# 导出为 ONNX
./bin/jpy-ml export -m yolov8n.pt -f onnx

# 导出为 TensorRT
./bin/jpy-ml export -m yolov8n.pt -f engine

# 导出为 CoreML
./bin/jpy-ml export -m yolov8n.pt -f coreml

# FP16 半精度导出
./bin/jpy-ml export -m yolov8n.pt -f onnx --half

# 动态输入尺寸
./bin/jpy-ml export -m yolov8n.pt -f onnx --dynamic
```

### 工具命令

#### 12. info - 模型信息

```bash
# 显示模型信息
./bin/jpy-ml info -m yolov8n.pt
```

#### 13. chat - LLM 对话

```bash
# 单次对话
./bin/jpy-ml chat -m Qwen/Qwen2.5-0.5B-Instruct --message "你好"

# 交互式对话
./bin/jpy-ml chat -m Qwen/Qwen2.5-0.5B-Instruct

# 使用系统提示
./bin/jpy-ml chat -m Qwen/Qwen2.5-0.5B-Instruct --system "你是一个有用的助手"

# 使用适配器
./bin/jpy-ml chat -m Qwen/Qwen2.5-0.5B-Instruct --adapter ./my_adapter

# 指定生成参数
./bin/jpy-ml chat -m Qwen/Qwen2.5-0.5B-Instruct --message "你好" \
    --max-tokens 256 \
    --temperature 0.8 \
    --top-p 0.95
```

#### 14. merge-adapter - 合并 LoRA 适配器

```bash
# 将 LoRA 适配器合并到基座模型
./bin/jpy-ml merge-adapter \
    -m Qwen/Qwen2.5-0.5B-Instruct \
    -a ./my_adapter \
    -o ./merged_model
```

#### 15. download - 下载模型

```bash
# 下载模型
./bin/jpy-ml download -n yolov8n.pt

# 下载 LLM 模型
./bin/jpy-ml download -n Qwen/Qwen2.5-0.5B-Instruct
```

#### 16. list-models - 列出可用模型

```bash
# 列出所有可用模型
./bin/jpy-ml list-models
```

## 通用选项

所有命令都支持以下通用选项：

- `--help` - 显示帮助信息
- `--version` - 显示版本信息

## 示例场景

### 场景 1：快速目标检测

```bash
# 下载模型并检测
./bin/jpy-ml predict -m yolov8n.pt -s photo.jpg --conf 0.5
```

### 场景 2：训练自定义模型

```bash
# 1. 准备数据集（参考 Ultralytics 数据格式）
# 2. 训练模型
./bin/jpy-ml yolo-train -m yolov8n.pt -d my_dataset.yaml --epochs 100

# 3. 验证模型
./bin/jpy-ml validate -m runs/train/yolo_train/weights/best.pt -d my_dataset.yaml

# 4. 导出模型
./bin/jpy-ml export -m runs/train/yolo_train/weights/best.pt -f onnx
```

### 场景 3：LLM 微调

```bash
# 1. 准备训练数据（JSONL 格式）
# 2. 微调模型
./bin/jpy-ml train -m Qwen/Qwen2.5-0.5B-Instruct -d training_data.jsonl --epochs 3

# 3. 使用微调后的模型对话
./bin/jpy-ml chat -m Qwen/Qwen2.5-0.5B-Instruct --adapter ./output/adapter

# 4. 合并适配器到基座模型（可选）
./bin/jpy-ml merge-adapter \
    -m Qwen/Qwen2.5-0.5B-Instruct \
    -a ./output/adapter \
    -o ./merged_model
```

### 场景 4：视频监控

```bash
# 摄像头实时检测
./bin/jpy-ml video -m yolov8n.pt -s 0 --show-fps

# 视频文件处理
./bin/jpy-ml video -m yolov8n.pt -s surveillance.mp4 --max-frames 1000
```

## 故障排除

### 1. 模型未找到

```
错误: 模型文件不存在: yolov8n.pt
```

解决方案：模型会自动下载，或者手动下载模型文件。

### 2. Python 环境问题

```
错误: Python 运行时未初始化
```

解决方案：确保已安装 Python 依赖：

```bash
.venv/bin/pip install -r src/main/resources/requirements.txt
```

### 3. GPU 内存不足

```
错误: CUDA 内存不足
```

解决方案：
- 使用更小的模型（如 yolov8n.pt 而不是 yolov8x.pt）
- 减小批次大小：`--batch-size 8`
- 减小图像尺寸：`--img-size 320`

### 4. 权限问题

```
权限被拒绝: ./bin/jpy-ml
```

解决方案：

```bash
chmod +x bin/jpy-ml
```

## 最佳实践

1. **模型选择**：
   - 快速原型：使用 `yolov8n.pt`（最小）
   - 平衡性能：使用 `yolov8s.pt` 或 `yolov8m.pt`
   - 最高精度：使用 `yolov8l.pt` 或 `yolov8x.pt`

2. **训练建议**：
   - 从小模型开始测试
   - 使用 GPU 加速训练
   - 监控训练日志，避免过拟合

3. **推理优化**：
   - 使用 FP16 半精度（`--half`）
   - 调整置信度阈值
   - 批量处理提高效率

4. **内存管理**：
   - 大模型需要更多内存
   - 批量处理时注意内存限制
   - 使用 `--batch-size` 控制内存使用

## 更多信息

- [项目主页](https://github.com/javpower/jpy-ml)
- [API 文档](README_zh.md)
- [问题反馈](https://github.com/javpower/jpy-ml/issues)
