<div align="center">

# jpy-ml

**有史以来最强大的 Java AI/ML 框架。**

*一个依赖。零 Python 安装。完整 PyTorch。生产即用。*

检测 &middot; 分割 &middot; 跟踪 &middot; 分类 &middot; 姿态 &middot; 训练 &middot; 验证 &middot; 导出 &middot; LLM 微调 — **纯 Java 完成。**

[![Maven Central](https://img.shields.io/maven-central/v/io.github.javpower/jpy-ml.svg)](https://central.sonatype.com/artifact/io.github.javpower/jpy-ml)
[![CI](https://github.com/javpower/jpy-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/javpower/jpy-ml/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-110%20passed-brightgreen)]()
[![Java](https://img.shields.io/badge/JDK-17-orange)]()
[![Python](https://img.shields.io/badge/CPython-3.12-blue)]()
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-green)]()

<p align="center">
  <a href="README.md">English</a> &middot;
  <a href="#快速开始">快速开始</a> &middot;
  <a href="#api-用法">API 文档</a> &middot;
  <a href="#路线图">路线图</a>
</p>

![index.png](docs%2Findex.png)

</div>

---

## 为什么选择 jpy-ml？

jpy-ml 将**整个 Python ML 生态直接嵌入 JVM** — YOLO、SAM、MediaPipe、OpenCV、HuggingFace LLM — 全部封装为简洁、类型安全的 Java API，**零 Python 安装、零 REST 服务、零网络延迟**。

```java
// 3 行代码。自动下载模型，运行推理，返回类型化结果。
try (Model model = Model.preset("yolov8n")) {
    DetectionResult result = model.predict("photo.jpg");
    System.out.println(result.toJson());   // {"task":"detect","count":6,"boxes":[...]}
}
```

无需安装 Python。无需手动下载模型。无需配置文件。无需 `Map<String, Object>` 强转。无需微服务。
**生产级 ML，进程内运行，纯 Java 体验。**

### 唯一一个能同时做到这些的 Java 框架：

<table>
<tr><td>

**计算机视觉**
- YOLOv8 / YOLO11 / YOLO26 / RT-DETR
- 检测、分割、分类、姿态、旋转框（OBB）
- 训练、验证、导出（ONNX/TensorRT/CoreML）
- 零拷贝 GPU 推理（`TensorBufferPool`）

</td><td>

**交互式分割**
- SAM 2 — 点/框提示 + 视频目标跟踪
- SAM 3 — 自然语言分割（"找到所有车辆"）
- CLIP 驱动的语义理解

</td></tr>
<tr><td>

**人体与人脸**
- MediaPipe — 手部追踪（21 关键点）
- 面部网格（478 关键点）、姿态估计（33 关键点）

</td><td>

**图像处理**
- OpenCV — 模糊、边缘检测、轮廓、形态学
- 颜色转换、阈值处理、缩放

</td></tr>
<tr><td>

**大语言模型**
- HuggingFace 模型下载与缓存
- 对话推理（`ChatMessage` API）
- LoRA/QLoRA 微调 + 实时回调
- 异步训练、适配器合并、量化

</td></tr><td>

**生产级架构**
- 完整 PyTorch — CPU / Apple MPS / NVIDIA CUDA / 多卡
- 单 JVM 进程 — 无需 Python 服务
- 线程安全引擎（`ReadWriteLock`）
- 自动下载 Python、依赖、模型权重

</td></tr>
</table>

### 传统方案 vs jpy-ml

| 传统 Java ML 方案 | jpy-ml |
|---|---|
| 通过 REST 调用 Python 服务 | ML **进程内运行**（JNI）— 零网络延迟 |
| 手动安装 Python + pip + torch | **自动下载** Python、所有依赖、模型权重 |
| 解析无类型的 JSON 结果 | **强类型**结果：`DetectionResult`、`PoseResult`... |
| 部署两个服务（Java + Python API） | **单 JVM 进程** — 运维更简单，成本更低 |
| 只能用 ONNX Runtime（仅 CPU） | **完整 PyTorch** — CPU、Apple MPS、NVIDIA CUDA、多卡 |
| 只能推理 | **推理 + 训练 + 验证 + 导出 + LLM 微调** — 全生命周期覆盖 |

---

## 特性

### 核心框架
- **嵌入式 Python 运行时** — 通过 Jep（JNI）将完整 CPython 嵌入 JVM，自动管理生命周期
- **零配置环境** — 生产模式自动下载 Python，开发模式使用本地 venv
- **线程安全引擎** — 单例 PythonEngine + ReadWriteLock，支持并发调用
- **类型安全 API** — 强类型配置、结果、回调 —— 用户代码无需 `Map` 强转
- **透明 Python 桥接** — 需要时可通过 `PythonEngine` 直接调用任意 Python/NumPy
- **SLF4J 日志** — 集成 Logback 日志框架
- **异常层次** — `JpyMlException` 基类 + 类型化异常

### 计算机视觉（Ultralytics YOLO）
- **统一 Model API** — 单一 `Model` 类支持所有架构和任务
- **6 个模型家族** — YOLOv8、YOLO11、YOLO26、RT-DETR、SAM，以及 ONNX Runtime 推理
- **5 种任务类型** — 检测、分割、分类、姿态估计、旋转框（OBB）
- **完整生命周期** — 推理、训练、验证、导出（ONNX/TensorRT/CoreML/TFLite/...）
- **丰富结果类型** — BoundingBox、Mask、Keypoint、RotatedBoundingBox，带过滤/查询辅助方法
- **设备抽象** — CPU / MPS（Apple Silicon）/ CUDA GPU / 多 GPU，通过 `Device` 类统一管理
- **Epoch 回调** — 实时训练进度，逐 epoch 的 loss/fitness 指标
- **逐类验证** — mAP50、mAP50-95、precision、recall 按类别细分
- **图像标注** — 通过 PIL 绘制推理结果，支持所有任务类型
- **零拷贝桥接** — `TensorBufferPool` + `RawDetectionResult` 高性能推理
- **GPU 内存管理** — `warmup()`、`unload()`、`reload(device)` API
- **直接图像输入** — `predict(byte[])`、`predict(BufferedImage)`，无需临时文件
- **异步 API** — `predictAsync()` 返回 `CompletableFuture<InferenceResult>`
- **结果序列化** — `toJson()`、`toMap()`，所有结果类型，无外部依赖
- **模型中心** — `Model.preset("yolov8n")` 自动下载并缓存模型
- **Java 可视化** — `ImageVisualizer` 纯 Java2D 绘制框/mask/关键点

### SAM 2 — 交互式分割
- **点/框提示** — 通过点击或绘制边界框分割对象
- **多提示组合** — 组合正向和负向提示
- **视频跟踪** — 利用时序记忆在视频帧间跟踪目标
- **SAM2Model** — 专用 SAM 2 推理模型类
- **SAM2Result** — 带 mask 和置信度的类型化结果
- **SAM2VideoTracker** — 视频目标跟踪，支持逐帧提示

### SAM 3 — 概念级分割
- **文本提示** — 使用自然语言分割目标（"person"、"red car"）
- **图像样本** — 使用参考图像区域查找相似目标
- **SAM3Model** — 专用 SAM 3 推理模型类
- **SAM3Result** — 带 mask、置信度和类别的类型化结果
- **CLIP 集成** — 使用 CLIP 实现文本到 mask 的语义理解

### OpenCV 集成
- **OpenCVEngine** — OpenCV 操作的 Java API
- **图像 I/O** — imread、imwrite，支持格式检测
- **颜色转换** — BGR2GRAY、BGR2RGB 等
- **滤波** — 高斯模糊、Canny 边缘检测、阈值处理
- **轮廓** — 查找和分析轮廓
- **形态学** — 膨胀、腐蚀、开运算、闭运算

### MediaPipe 集成
- **MediaPipeEngine** — MediaPipe 任务的 Java API
- **手部追踪** — 检测手部 21 个关键点
- **面部网格** — 检测面部 478 个关键点
- **姿态估计** — 检测身体 33 个关键点

### LLM — 大语言模型
- **LLMModel** — 统一入口：模型下载、推理、微调
- **HuggingFace Hub** — `LLMModel.download("Qwen/Qwen2.5-0.5B-Instruct")` 自动缓存
- **对话推理** — 类型化 `ChatResponse`，含 token 计数，支持 `ChatMessage` 角色化 API
- **LoRA/QLoRA 微调** — 基于 PEFT + TRL SFTTrainer 的参数高效训练
- **实时回调** — 逐 step 训练进度，通过 `TrainingCallback` 回调
- **异步训练** — `runAsync()` 返回 `CompletableFuture<LLMTrainingResult>`
- **量化支持** — NF4/INT8（CUDA），根据平台自动选择
- **自动设备检测** — CPU / Apple MPS / NVIDIA CUDA 自动检测
- **LoRA 合并** — `LLMModel.mergeAdapter()` 将适配器合并回基座模型
- **生成配置** — temperature、top-p、max tokens、repetition penalty
- **自动依赖安装** — transformers、peft、trl、accelerate 首次使用时自动安装

### FLUX.1 — AI 图像生成
- **FluxModel** — 文本生成图像的统一入口
- **FLUX.1 Dev** — 高质量版本，20 步推理，需要 HuggingFace 授权
- **FLUX.1 Schnell** — 快速版本，4 步推理，Apache 2.0 许可
- **文本到图像** — `flux.generate("A cat in space", "output.png")`
- **图像到图像** — `flux.img2img(prompt, inputPath, outputPath, config)`
- **自定义配置** — 宽度、高度、步数、引导强度、种子
- **自动设备检测** — CPU / Apple MPS / NVIDIA CUDA
- **按需依赖** — 首次使用时自动安装 diffusers、transformers 等

---

## Maven

```xml
<dependency>
    <groupId>io.github.javpower</groupId>
    <artifactId>jpy-ml</artifactId>
    <version>1.4.0</version>
</dependency>
```

---

## 环境要求

| 组件 | 版本 | 备注 |
|------|------|------|
| JDK | Temurin 17 | 通过 sdkman，**不要用 GraalVM**（JNI 崩溃） |
| Python | 3.12（自动下载） | python-build-standalone，用户无需安装 |
| Jep (pip + Maven) | 4.3.1 | Java-Python JNI 桥接，Maven groupId: `org.ninia` |
| Ultralytics | 8.4.45 | YOLOv8, YOLO11, YOLO26, RT-DETR, SAM |
| OpenCV | 4.6.0+ | 图像处理（可选） |
| MediaPipe | 0.10.0+ | 手部/面部/姿态检测（可选） |
| PyTorch | 2.11.0 | CPU-only macOS ARM64 |
| OS | Linux、macOS、Windows | 三个平台均通过 CI 测试 |

### 依赖管理

**两种初始化模式：**

| 模式 | 方法 | 自动安装 | 使用场景 |
|------|------|----------|----------|
| **自动下载** | `PythonRuntime.init()` | ✅ 是 | 生产环境，零配置 |
| **本地venv** | `PythonRuntime.init(pythonPath, jepLibPath)` | ❌ 否 | 开发环境，已有Python |

**自动下载模式** (`PythonRuntime.init()`):
- 自动下载 python-build-standalone
- 自动安装 `requirements.txt` 中的所有依赖

**本地venv模式** (`PythonRuntime.init(pythonPath, jepLibPath)`):
- 使用现有Python环境
- 需手动安装依赖：
  ```bash
  .venv/bin/pip install -r src/main/resources/requirements.txt
  ```

### 代理与镜像配置

对于服务器无法直接访问 GitHub 或 PyPI 的情况，支持以下配置：

**快捷代理模式**（最简单）：
```bash
# 使用默认代理 (127.0.0.1:7890) + GitHub 镜像 + 清华 PyPI 源
java -Djpy.proxy=true -jar your-app.jar

# 或使用 CLI 脚本
jpy-ml --proxy train ...
```

**JVM 系统属性**（高级）：
```bash
# HTTP/HTTPS 代理
java -Djpy.download.proxy=http://127.0.0.1:7890 -jar your-app.jar

# 自定义 GitHub 镜像源（下载 Python 运行时）
java -Djpy.download.base-url=https://mirror.ghproxy.com/https://github.com/astral-sh/python-build-standalone/releases/download/ -jar your-app.jar

# 自定义 PyPI 镜像源（pip 安装依赖）
java -Djpy.pip.index-url=https://pypi.tuna.tsinghua.edu.cn/simple -jar your-app.jar

# 自定义模型下载镜像源
java -Djpy.model.base-url=https://mirror.ghproxy.com/https://github.com/ultralytics/assets/releases/download -jar your-app.jar
```

**环境变量**：
```bash
export JPY_DOWNLOAD_PROXY=http://127.0.0.1:7890
export JPY_DOWNLOAD_BASE_URL=https://mirror.ghproxy.com/https://github.com/astral-sh/python-build-standalone/releases/download/
export JPY_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export JPY_MODEL_BASE_URL=https://mirror.ghproxy.com/https://github.com/ultralytics/assets/releases/download
```

| 属性 | 环境变量 | 说明 |
|------|----------|------|
| `jpy.proxy=true` | - | 快捷模式：代理 + 镜像（默认 127.0.0.1:7890） |
| `jpy.download.proxy` | `JPY_DOWNLOAD_PROXY` | HTTP 代理地址（如 `http://host:port`） |
| `jpy.download.base-url` | `JPY_DOWNLOAD_BASE_URL` | Python 运行时下载的自定义基础 URL |
| `jpy.pip.index-url` | `JPY_PIP_INDEX_URL` | pip 使用的自定义 PyPI 源 |
| `jpy.model.base-url` | `JPY_MODEL_BASE_URL` | 模型下载的自定义基础 URL |

---

## 快速开始

### 1. 切换 JDK 17

```bash
sdk install java 17.0.19-tem   # 首次安装
sdk use java 17.0.19-tem
java -version                   # 确认: openjdk 17.0.19 Temurin
```

### 2. 创建 Python 虚拟环境

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
```

### 3. 安装 Python 依赖

```bash
# 基础依赖（必须）
.venv/bin/pip install jep numpy ultralytics

# 可选：OpenCV 图像处理
.venv/bin/pip install opencv-python

# 可选：MediaPipe 手部/面部/姿态检测
.venv/bin/pip install mediapipe

# 或一次性安装所有依赖
.venv/bin/pip install -r src/main/resources/requirements.txt
```

### 4. 构建与测试

```bash
mvn clean test

# 预期: Tests run: 72, Failures: 0, Errors: 0, Skipped: 6
```

---

## CLI 使用

jpy-ml 提供了命令行界面，让您无需编写 Java 代码即可使用各种 ML 功能。

### 构建 CLI

```bash
mvn package -Pcli -DskipTests
```

### 使用 CLI

```bash
# 查看帮助
./bin/jpy-ml --help

# 目标检测
./bin/jpy-ml predict -m yolov8n.pt -s photo.jpg

# 图像分类
./bin/jpy-ml classify -m yolov8n-cls.pt -s photo.jpg

# 姿态估计
./bin/jpy-ml pose -m yolov8n-pose.pt -s person.jpg

# SAM 分割
./bin/jpy-ml segment -m sam2.1_t.pt -i photo.jpg -p 320,240

# YOLO 模型训练
./bin/jpy-ml yolo-train -m yolov8n.pt -d coco128.yaml --epochs 50

# 模型验证
./bin/jpy-ml validate -m yolov8n.pt -d coco128.yaml

# 模型导出
./bin/jpy-ml export -m yolov8n.pt -f onnx

# LLM 对话
./bin/jpy-ml chat -m Qwen/Qwen2.5-0.5B-Instruct --message "你好"

# LLM 微调
./bin/jpy-ml train -m Qwen/Qwen2.5-0.5B-Instruct -d training_data.jsonl

# 列出可用模型
./bin/jpy-ml list-models

# 下载模型
./bin/jpy-ml download -n yolov8n.pt

# FLUX.1 图像生成
./bin/jpy-ml generate -p "A cat in space" -o cat.png

# FLUX.1 高质量生成
./bin/jpy-ml generate -m dev -p "A beautiful sunset" -o sunset.png --steps 30
```

更多 CLI 用法请参考 [CLI 使用指南](CLI_USAGE_zh.md)。

---

## API 用法

### 模型加载 — 指定任务类型和设备

```java
// 自动检测任务类型
try (Model model = new Model("yolov8n.pt")) { ... }

// 显式指定任务类型（适用于自定义命名的模型）
try (Model model = new Model("my_model.pt", TaskType.DETECT)) { ... }

// 指定设备
ModelConfig config = new ModelConfig()
    .device(Device.cpu());       // CPU
    .device(Device.mps());       // Apple Silicon GPU
    .device(Device.gpu(0));      // GPU 0
    .device(Device.cuda(0));     // CUDA:0
```

### 目标检测

```java
try (Model model = new Model("yolov8n.pt")) {
    DetectionResult result = model.predict("photo.jpg");

    for (ClassPrediction pred : result.getBoxes()) {
        System.out.println(pred);
        // person 92.3% BoundingBox[x1=100.5, y1=50.2, x2=300.1, y2=400.8]
    }

    // 过滤结果
    List<ClassPrediction> persons = result.filterByClass("person");
    List<ClassPrediction> confident = result.filterByConfidence(0.8f);
}
```

### 实例分割

```java
try (Model model = new Model("yolov8n-seg.pt")) {
    SegmentationResult result = model.predict("photo.jpg");

    for (int i = 0; i < result.getBoxes().size(); i++) {
        ClassPrediction pred = result.getBoxes().get(i);
        Mask mask = result.getMasks().get(i);
        System.out.println(pred.className() + " 掩码点数: " + mask.getPointCount());
    }
}
```

### 图像分类

```java
try (Model model = new Model("yolov8n-cls.pt")) {
    ClassificationResult result = model.predict("photo.jpg");
    System.out.println("预测: " + result.getTop1ClassName());
    System.out.println("置信度: " + result.getTop1Confidence());
}
```

### 姿态估计

```java
try (Model model = new Model("yolov8n-pose.pt")) {
    PoseResult result = model.predict("photo.jpg");

    for (int i = 0; i < result.personCount(); i++) {
        KeypointCollection kpts = result.getKeypoints().get(i);
        Keypoint nose = kpts.getNose();
        System.out.printf("人物 %d: 鼻子=(%.1f, %.1f)%n", i, nose.x(), nose.y());
    }
}
```

### 旋转框检测 (OBB)

```java
try (Model model = new Model("yolov8n-obb.pt")) {
    OBBResult result = model.predict("satellite_image.jpg");

    for (OBBPrediction pred : result.getPredictions()) {
        RotatedBoundingBox box = pred.box();
        System.out.printf("%s %.1f%% 角度=%.1f°%n",
            pred.className(), pred.confidence() * 100, box.angleDegrees());
    }
}
```

### 推理配置

```java
ModelConfig config = new ModelConfig()
    .confidence(0.5f)        // 置信度阈值
    .iouThreshold(0.7f)      // NMS IoU 阈值
    .imageSize(640)           // 输入图像尺寸
    .maxDetections(100)       // 最大检测数
    .device(Device.gpu(0))   // 设备选择
    .augment(true)            // 测试时增强
    .filterClasses(0, 2, 5)   // 只检测指定类别
    .half(true)               // FP16 推理（GPU）
    .verbose(false)           // 静默模式
    .save(true)               // 保存结果
    .embed(0, 1, 2);          // 提取特征嵌入

InferenceResult result = model.predict("photo.jpg", config);
```

### ONNX 推理

```java
// 加载 ONNX 模型（推荐显式指定任务类型）
try (Model model = new Model("yolov8n.onnx", TaskType.DETECT)) {
    DetectionResult result = model.predict("photo.jpg");
}

// 导出为 ONNX
ExportResult exported = model.export(ExportFormat.ONNX);
```

### 模型训练

```java
TrainingConfig config = new TrainingConfig()
    .dataConfig("coco128.yaml")
    .epochs(50)
    .batchSize(16)
    .device(Device.gpu(0))
    .optimizer(OptimizerType.ADAMW)
    .learningRate(0.001f)
    .augmentation(new AugmentationConfig()
        .mosaic(true)
        .fliplr(0.5f)
        .hsvH(0.015f));

// 带 epoch 回调
TrainingResult result = model.train(config, (epoch, log) -> {
    System.out.println("Epoch " + epoch + ": " + log);
});

// 查看 epoch 级别指标
for (EpochMetric m : result.getEpochMetrics()) {
    System.out.printf("Epoch %d: box_loss=%.4f cls_loss=%.4f%n",
        m.epoch(), m.boxLoss(), m.clsLoss());
}
```

### 模型验证

```java
ValidationResult val = model.validate("coco128.yaml");
System.out.printf("mAP50=%.3f, mAP50-95=%.3f, P=%.3f, R=%.3f%n",
    val.getMAP50(), val.getMAP5095(), val.getPrecision(), val.getRecall());

// 每个类别的指标
for (PerClassMetric pc : val.getPerClassMetrics()) {
    System.out.println(pc);  // "person (id=0): mAP50-95=0.7234"
}
```

### 批量推理

```java
try (Model model = new Model("yolov8n.pt")) {
    List<String> images = List.of("photo1.jpg", "photo2.jpg", "photo3.jpg");
    List<InferenceResult> results = model.predict(images);

    for (int i = 0; i < results.size(); i++) {
        System.out.println("图像 " + i + ": " + results.get(i).count() + " 个目标");
    }
}
```

### 视频流式推理

```java
try (Model model = new Model("yolov8n.pt")) {
    // 视频文件 — 逐帧分块流式处理
    model.predictVideo("video.mp4", frame -> {
        System.out.println("帧: " + frame.count() + " 个目标");
    });

    // 摄像头 — 实时推理（阻塞当前线程）
    // 在另一个线程调用 stopStream() 停止
    new Thread(() -> {
        Thread.sleep(10000);
        model.stopStream();  // 10 秒后停止
    }).start();

    model.predictStream(0, frame -> {
        // 0 = 默认摄像头
        if (frame instanceof DetectionResult dr) {
            System.out.println("摄像头: " + dr.getBoxes().size() + " 个目标");
        }
    });
}
```

### LLM — 下载与对话推理

```java
// 从 HuggingFace Hub 下载模型（缓存到 ~/.jpy-ml/llm-models/）
LLMModel model = LLMModel.download("Qwen/Qwen2.5-0.5B-Instruct");

// 或从本地路径加载
LLMModel model = LLMModel.load("/path/to/local/model");

// 对话推理
ChatResponse response = model.chat(
    ChatMessage.system("你是一个有用的助手"),
    ChatMessage.user("你好，请用一句话介绍你自己")
);

System.out.println(response.getContent());
System.out.println("Tokens: prompt=" + response.getPromptTokens()
    + " completion=" + response.getCompletionTokens());

// 带生成配置
ChatResponse response = model.chat(
    List.of(
        ChatMessage.system("你是一个有用的助手"),
        ChatMessage.user("解释量子计算")
    ),
    GenerationConfig.create()
        .maxNewTokens(256)
        .temperature(0.7)
        .topP(0.9)
        .repetitionPenalty(1.1)
);
```

### LLM — LoRA 微调

```java
LLMModel model = LLMModel.load("Qwen/Qwen2.5-0.5B-Instruct")
    .quantize(Quantization.NONE); // macOS

// 同步微调，带实时回调
LLMTrainingResult result = model.fineTune()
    .lora(LoRAConfig.create().rank(8).alpha(16))
    .dataset("training_data.jsonl")
    .config(LLMTrainConfig.create()
        .epochs(3)
        .batchSize(4)
        .gradientAccumulation(4)
        .learningRate(2e-4)
        .maxSeqLength(2048)
        .gradientCheckpointing(true))
    .run((step, log) -> {
        System.out.println("Step " + step + ": " + log);
    });

System.out.println("适配器保存到: " + result.getAdapterPath());
System.out.println("最终 loss: " + result.getFinalLoss());
```

### LLM — 加载适配器推理

```java
// 加载基座模型 + 训练好的 LoRA 适配器
LLMModel finetuned = LLMModel.load("Qwen/Qwen2.5-0.5B-Instruct")
    .adapter("/path/to/adapter");

ChatResponse response = finetuned.chat(
    ChatMessage.user("你叫什么名字？")
);
System.out.println(response.getContent());
```

### LLM — 异步微调

```java
CompletableFuture<LLMTrainingResult> future = model.fineTune()
    .lora(LoRAConfig.create().rank(4).alpha(8))
    .dataset("data.jsonl")
    .config(LLMTrainConfig.create().epochs(2))
    .runAsync((step, log) -> {
        System.out.println("[异步] " + log);
    });

// 做其他事情...

LLMTrainingResult result = future.get(10, TimeUnit.MINUTES);
```

### LLM — 合并适配器到基座模型

```java
LLMTrainingResult result = model.fineTune()
    .dataset("data.jsonl")
    .config(LLMTrainConfig.create().epochs(3))
    .run();

// 将 LoRA 适配器合并到基座模型，用于独立部署
String mergedPath = LLMModel.mergeAdapter(
    model.getModelPath(),
    result.getAdapterPath(),
    "/path/to/merged-model"
);
```

### LLM — 训练数据格式（JSONL）

```json
{"messages": [{"role": "user", "content": "1+1=?"}, {"role": "assistant", "content": "1+1=2"}]}
{"messages": [{"role": "user", "content": "什么是Java？"}, {"role": "assistant", "content": "Java是一种编程语言。"}]}
```

也支持 instruction 格式：
```json
{"instruction": "翻译成英文", "input": "你好", "output": "Hello"}
```

### FLUX.1 — 文本生成图像

```java
// 快速生成（Schnell，4 步）
try (FluxModel flux = new FluxModel("black-forest-labs/FLUX.1-schnell")) {
    FluxResult result = flux.generate("A cat in space", "cat.png");
    System.out.println("输出: " + result.getFirstOutputPath());
    System.out.println("耗时: " + result.getElapsedSeconds() + "s");
}

// 高质量生成（Dev，20 步）
FluxConfig config = FluxConfig.dev()
    .width(1920)
    .height(1080)
    .steps(30)
    .guidance(7.5f)
    .seed(42);

try (FluxModel flux = new FluxModel("black-forest-labs/FLUX.1-dev")) {
    FluxResult result = flux.generate("A beautiful sunset", "sunset.png", config);
}

// 图像到图像
try (FluxModel flux = new FluxModel("black-forest-labs/FLUX.1-schnell")) {
    FluxResult result = flux.img2img(
        "Oil painting style", "input.png", "output.png",
        new FluxConfig().steps(10)
    );
}
```

---

## 架构

```
┌──────────────────────────────────────────┐
│            用户 Java 代码                 │
│  Model m = new Model("yolov8n.pt");      │
│  DetectionResult r = m.predict(img);     │
├──────────────────────────────────────────┤
│  Model / ModelConfig / Result 类型       │
│  (58 个 Java 源文件)                     │
├──────────────────────────────────────────┤
│  PythonEngine (单例, ReadWriteLock)      │
│  SharedInterpreter + sys.path 过滤       │
├──────────────────────────────────────────┤
│  Jep 4.3.1 (JNI 桥接)                   │
│  libjep.jnilib 来自 pip install jep      │
├──────────────────────────────────────────┤
│  Python 辅助脚本 (12 个 .py 文件)        │
│  jpy_load_model / jpy_extract_result /   │
│  jpy_train / jpy_export / jpy_validate / │
│  jpy_sam2 / jpy_sam3 / jpy_opencv /      │
│  jpy_mediapipe / jpy_streaming / ...     │
├──────────────────────────────────────────┤
│  CPython 3.13 (.venv)                    │
│  Ultralytics 8.4.45 + PyTorch 2.11       │
└──────────────────────────────────────────┘
```

---

## 测试覆盖

全部 110 个测试通过（0 跳过）：

| 测试套件 | 数量 | 说明 |
|---------|------|------|
| QuickVerifyTest | 10 | 基础 Python 桥接（eval、put/get、numpy、列表、字典） |
| PythonEngineTest | 11 | 引擎特性（多线程、回调、模块） |
| PythonRuntimeTest | 3 | 平台检测 |
| ModelIntegrationTest | 18 | 完整 YOLO 集成（推理 + 批量 + 视频 + 训练 + 导出） |
| SAMIntegrationTest | 9 | SAM 2/3 集成（点/框/视频/文本/样本） |
| LLMIntegrationTest | 4 | LLM 下载、对话、LoRA 微调、异步训练 |
| NewFeaturesTest | 17 | 序列化、byte[]输入、异步、模型中心、可视化 |
| TensorBufferPoolTest | 6 | 零拷贝缓冲池 |
| BoundingBoxTest | 10 | BoundingBox 记录类 |
| KeypointTest | 5 | Keypoint 记录类 |
| InferenceSpeedTest | 5 | InferenceSpeed 记录类 |
| MediaPipeEngineTest | 4 | 手部/面部/姿态检测 |
| OpenCVEngineTest | 8 | 图像处理操作 |

覆盖范围：
- 5 种任务类型：检测、分割、分类、姿态、OBB
- PyTorch 和 ONNX 模型推理
- YOLOv8 和 YOLO26 模型
- SAM 2 交互式分割 + 视频跟踪
- SAM 3 文本提示 + 图像样本分割
- LLM 模型下载、对话推理、LoRA/QLoRA 微调（含异步训练）
- MediaPipe 手部（21 关键点）、面部（478 关键点）、姿态（33 关键点）
- OpenCV 图像处理（颜色转换、滤波、边缘检测、轮廓、形态学）
- 训练（含 epoch 回调和指标）
- 验证（含每类别 mAP）
- ONNX 导出
- 训练后加载模型推理

---

## 关键设计

1. **单例 PythonEngine** — Jep 限制每个 JVM 只能有一个 SharedInterpreter。所有 Model 实例通过唯一变量名前缀共享。

2. **Python 脚本做重活** — 复杂的 tensor→dict 转换在 Python 端完成，Java 只接收 `Map<String, Object>`。

3. **sys.path 过滤** — 自动过滤 Homebrew 系统包路径冲突，优先使用 venv 的 site-packages。

4. **AutoCloseable Model** — `Model implements AutoCloseable`，close 时将 Python 变量设为 `None` 便于 GC。

5. **Builder 模式** — 所有配置类（`ModelConfig`、`TrainingConfig`、`ExportConfig`、`AugmentationConfig`）使用链式调用。

---

## 常见问题

### JVM 崩溃 (Abort trap: 6)
**原因：** 使用了 GraalVM CE。
**解决：** 切换到标准 OpenJDK（Temurin）。

### "Jep native library not found"
**原因：** venv 中未安装 jep。
**解决：** `.venv/bin/pip install jep`

### "ultralytics not installed" 或 opencv dlopen 错误
**原因：** Homebrew site-packages 路径与 venv 冲突。
**解决：** 使用 `PythonRuntime.init(pythonPath, jepLibPath)` 传入 venv 路径，PythonEngine 会自动过滤冲突路径。

---

## 路线图

jpy-ml 的定位是通用 Java-Python ML 桥接框架。YOLO 是第一个引擎 —— 更多引擎正在路上。

### 已完成
- [x] 批量推理 API（`model.predict(List<String>)`）
- [x] 摄像头/视频流实时推理
- [x] python-build-standalone 自动下载（终端用户无需安装 Python）
- [x] PyTorch tensor 零拷贝桥接（`TensorBufferPool` + `RawDetectionResult`）
- [x] GPU 显存管理与模型预热 API
- [x] SAM 2 交互式分割（点/框提示）
- [x] SAM 2 视频跟踪（逐帧提示 + 时序记忆）
- [x] SAM 3 概念级分割（文本提示 + 图像样本）
- [x] OpenCV 集成（图像处理）
- [x] MediaPipe 集成（手部/面部/姿态）
- [x] SLF4J 日志框架
- [x] 异常层次（`JpyMlException` 基类）
- [x] CI/CD（GitHub Actions）
- [x] 结果序列化（toJson / toMap）
- [x] 直接图像输入（byte[] / BufferedImage）
- [x] 模型中心自动下载（Model.preset）
- [x] Java2D 结果可视化（ImageVisualizer）
- [x] 异步推理 API（predictAsync）
- [x] LLM 对话推理（HuggingFace Transformers）
- [x] LoRA/QLoRA 微调，带实时 step 回调
- [x] 异步微调 API（runAsync）
- [x] LoRA 适配器合并到基座模型
- [x] LLM 依赖自动安装（transformers、peft、trl、accelerate）

### 近期
- [ ] 所有值类型的单元测试

### 计划中的 ML 引擎

#### 其他引擎
- [ ] **Whisper** — 语音识别、语音转文字
- [ ] **Stable Diffusion / FLUX** — 图像生成、局部重绘、ControlNet

### 基础设施
- [ ] 模型仓库 / Hub 集成（从 URL 下载）
- [ ] Spring Boot Starter 自动配置
- [ ] GraalVM Native Image 支持
