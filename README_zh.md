# jpy-ml

[English](README.md)

**jpy-ml** 是一个生产级 Java 框架，将 Python ML 生态无缝桥接到 Java。
通过 JNI 嵌入完整的 Python 运行时，提供零配置环境管理，并将强大的 ML 库封装为
简洁、类型安全的 Java API。

愿景：**将整个 Python AI/ML 生态带给 Java 开发者 —— 无需任何 Python 知识。**

当前已实现 **Ultralytics YOLO**（v8/v11/v26、RT-DETR、SAM）的一等支持，
覆盖目标检测、实例分割、图像分类、姿态估计、旋转框检测 ——
推理、训练、验证、导出、ONNX Runtime 全部打通。

---

## 特性

### 核心框架
- **嵌入式 Python 运行时** — 通过 Jep（JNI）将完整 CPython 嵌入 JVM，自动管理生命周期
- **零配置环境** — 生产模式自动下载 Python，开发模式使用本地 venv
- **线程安全引擎** — 单例 PythonEngine + ReadWriteLock，支持并发调用
- **类型安全 API** — 强类型配置、结果、回调 —— 用户代码无需 `Map` 强转
- **透明 Python 桥接** — 需要时可通过 `PythonEngine` 直接调用任意 Python/NumPy

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

---

## Maven

```xml
<dependency>
    <groupId>io.github.javpower</groupId>
    <artifactId>jpy-ml</artifactId>
    <version>1.0.0-SNAPSHOT</version>
</dependency>
```

---

## 环境要求

| 组件 | 版本 | 备注 |
|------|------|------|
| JDK | Temurin 17 | 通过 sdkman，**不要用 GraalVM**（JNI 崩溃） |
| Python | 3.13 (venv) | 项目本地 `.venv/` |
| Jep (pip + Maven) | 4.3.1 | Java-Python JNI 桥接，Maven groupId: `org.ninia` |
| Ultralytics | 8.4.45 | YOLOv8, YOLO11, YOLO26, RT-DETR, SAM |
| PyTorch | 2.11.0 | CPU-only macOS ARM64 |
| OS | macOS ARM64 (Apple Silicon) | |

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
.venv/bin/pip install jep numpy ultralytics
```

### 4. 构建与测试

```bash
mvn clean test

# 预期: Tests run: 39, Failures: 0, Errors: 0, Skipped: 0
```

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

---

## 架构

```
┌──────────────────────────────────────────┐
│            用户 Java 代码                 │
│  Model m = new Model("yolov8n.pt");      │
│  DetectionResult r = m.predict(img);     │
├──────────────────────────────────────────┤
│  Model / ModelConfig / Result 类型       │
│  (41 个 Java 源文件)                     │
├──────────────────────────────────────────┤
│  PythonEngine (单例, ReadWriteLock)      │
│  SharedInterpreter + sys.path 过滤       │
├──────────────────────────────────────────┤
│  Jep 4.3.1 (JNI 桥接)                   │
│  libjep.jnilib 来自 pip install jep      │
├──────────────────────────────────────────┤
│  Python 辅助脚本 (6 个 .py 文件)         │
│  jpy_load_model / jpy_extract_result /   │
│  jpy_train / jpy_export / jpy_validate   │
├──────────────────────────────────────────┤
│  CPython 3.13 (.venv)                    │
│  Ultralytics 8.4.45 + PyTorch 2.11       │
└──────────────────────────────────────────┘
```

---

## 测试覆盖

全部 41 个测试通过：

| 测试套件 | 数量 | 说明 |
|---------|------|------|
| QuickVerifyTest | 10 | 基础 Python 桥接（eval、put/get、numpy、列表、字典） |
| PythonEngineTest | 11 | 引擎特性（多线程、回调、模块） |
| PythonRuntimeTest | 3 | 平台检测 |
| ModelIntegrationTest | 17 | 完整 YOLO 集成（推理 + 批量 + 视频 + 训练 + 导出） |

覆盖范围：
- 5 种任务类型：检测、分割、分类、姿态、OBB
- PyTorch 和 ONNX 模型推理
- YOLOv8 和 YOLO26 模型
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

### 近期
- [ ] PyTorch tensor 零拷贝桥接，实现高性能数据传输
- [x] 批量推理 API（`model.predict(List<String>)`）
- [x] 摄像头/视频流实时推理
- [x] python-build-standalone 自动下载（终端用户无需安装 Python）
- [ ] Windows / Linux CI 测试

### 计划中的 ML 引擎

#### SAM 2 — 交互式提示分割

[SAM 2](https://ai.meta.com/sam2/)（Segment Anything Model 2）由 Meta 开发，是与 YOLO 完全不同的分割范式。YOLO 自动检测所有目标，而 SAM 2 则根据你的**交互式提示**精确分割你指定的目标：

| 提示类型 | 说明 | 示例 |
|---------|------|------|
| 点提示 | 点击某个位置，分割该位置的物体 | "分割 (100, 200) 处的物体" |
| 框提示 | 绘制边界框，分割框内的物体 | "分割 [50,50,300,400] 内的区域" |
| 掩码提示 | 基于已有掩码进行精细化分割 | "将此掩码扩展以包含尾部" |

**视频目标跟踪** — SAM 2 的核心能力：在第 1 帧中标记一个物体，模型会利用时序记忆在整个视频中持续跟踪该目标。支持跨帧添加/移除提示进行修正。

**计划 API：**
```java
// 图像提示分割
SAM2Model sam = new SAM2Model("sam2_b.pt");
SAM2Result result = sam.predict("photo.jpg",
    Prompt.point(100, 200),          // 分割该位置的物体
    Prompt.point(300, 400, Label.NEGATIVE)  // 排除该区域
);

// 视频目标跟踪
SAM2VideoTracker tracker = sam.trackVideo("video.mp4");
tracker.addPrompt(0, Prompt.box(50, 50, 300, 400));  // 在第 0 帧标记目标
SAM2VideoMask mask = tracker.propagate();              // 跟踪到所有帧
```

**模型：** `sam2_t.pt`、`sam2_s.pt`、`sam2_b.pt`、`sam2_l.pt`、`sam2.1_t/s/b/l.pt`
**限制：** 仅推理，不支持训练和导出。需要包含 SAM2 模块的 ultralytics 版本。

#### SAM 3 — 概念级分割

[SAM 3](https://ai.meta.com/blog/segment-anything-announcements/) 将分割能力提升到概念层面。不需要手动点击标注，直接用**自然语言**或**视觉样本**描述你想要分割的目标：

| 模式 | 输入 | 说明 |
|------|------|------|
| 文本提示 | "person"、"red car"、"bus" | 找到并分割所有匹配该概念的目标实例 |
| 图像样本 | 提供参考物体的边界框 | 在目标图像中找到相似物体 |
| 语义分割 | 无需提示 | 自动分割所有目标并进行语义标注 |

这解锁了 SAM 2 无法实现的应用场景：无需逐一点击就能"找到图中所有的车"，或者跨图像"找到和这个物体相似的目标"。

**计划 API：**
```java
SAM3Model sam = new SAM3Model("sam3.pt");

// 基于文本的概念分割
SAM3Result result = sam.predict("street.jpg",
    Prompt.text("person")     // 分割所有行人
);

// 基于图像样本："找到类似的目标"
SAM3Result result = sam.predict("target.jpg",
    Prompt.exemplar("reference.jpg", BoundingBox.of(10, 20, 200, 300))
);
```

**模型：** `sam3.pt`（3.45 GB，需申请 HuggingFace 权重访问）
**要求：** ultralytics 8.3.237+、PyTorch 2.1+
**限制：** 仅推理，权重文件不会自动下载。

#### 其他引擎
- [ ] **Transformers (HuggingFace)** — NLP、文本生成、翻译、向量嵌入
- [ ] **Whisper** — 语音识别、语音转文字
- [ ] **Stable Diffusion / FLUX** — 图像生成、局部重绘、ControlNet
- [ ] **DeepSeek / LLM** — 大语言模型推理
- [ ] **OpenCV** — 传统计算机视觉流水线
- [ ] **MediaPipe** — 手部追踪、人脸检测、手势识别

### 基础设施
- [ ] 异步训练实时流式回调
- [ ] GPU 显存管理与模型预热 API
- [ ] 模型仓库 / Hub 集成（从 URL 下载）
- [ ] Spring Boot Starter 自动配置
- [ ] GraalVM Native Image 支持
