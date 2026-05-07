<div align="center">

# jpy-ml

**The most powerful Java AI/ML framework ever built.**

*One dependency. Zero Python. Full PyTorch. Production-ready.*

Detect &middot; Segment &middot; Track &middot; Classify &middot; Pose &middot; Train &middot; Validate &middot; Export &middot; Fine-tune LLMs — **all in pure Java.**

[![Maven Central](https://img.shields.io/maven-central/v/io.github.javpower/jpy-ml.svg)](https://central.sonatype.com/artifact/io.github.javpower/jpy-ml)
[![CI](https://github.com/javpower/jpy-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/javpower/jpy-ml/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-110%20passed-brightgreen)]()
[![Java](https://img.shields.io/badge/JDK-17-orange)]()
[![Python](https://img.shields.io/badge/CPython-3.12-blue)]()
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-green)]()

<p align="center">
  <a href="README_zh.md">中文文档</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#api-usage">API Docs</a> &middot;
  <a href="#roadmap">Roadmap</a>
</p>

![index.png](docs%2Findex.png)

</div>

---

## Why jpy-ml?

jpy-ml embeds the **entire Python ML ecosystem directly into the JVM** — YOLO, SAM, MediaPipe, OpenCV, HuggingFace LLMs — all behind clean, type-safe Java APIs with **zero Python setup, zero REST servers, zero network latency**.

```java
// 3 lines. Auto-downloads model, runs inference, returns typed results.
try (Model model = Model.preset("yolov8n")) {
    DetectionResult result = model.predict("photo.jpg");
    System.out.println(result.toJson());   // {"task":"detect","count":6,"boxes":[...]}
}
```

No Python installation. No model downloads. No config files. No `Map<String, Object>` casting. No microservices.
**Just production-ready ML, running in-process, in Java.**

### The only Java framework that does ALL of this:

<table>
<tr><td>

**Computer Vision**
- YOLOv8 / YOLO11 / YOLO26 / RT-DETR
- Detect, Segment, Classify, Pose, OBB
- Train, Validate, Export (ONNX/TensorRT/CoreML)
- Zero-copy GPU inference with `TensorBufferPool`

</td><td>

**Interactive Segmentation**
- SAM 2 — point/box prompts + video tracking
- SAM 3 — natural language ("find all cars")
- CLIP-powered semantic understanding

</td></tr>
<tr><td>

**Body & Face**
- MediaPipe — hand tracking (21 pts)
- Face mesh (478 pts), pose estimation (33 pts)

</td><td>

**Image Processing**
- OpenCV — blur, edges, contours, morphology
- Color conversion, thresholding, resize

</td></tr>
<tr><td>

**Large Language Models**
- HuggingFace model download & caching
- Chat inference with `ChatMessage` API
- LoRA/QLoRA fine-tuning + real-time callbacks
- Async training, adapter merge, quantization

</td></tr><td>

**Production-Grade**
- Full PyTorch — CPU / Apple MPS / NVIDIA CUDA / Multi-GPU
- Single JVM process — no Python server needed
- Thread-safe engine with `ReadWriteLock`
- Auto-download Python, deps, and model weights

</td></tr>
</table>

### Traditional Java ML vs jpy-ml

| Traditional Java ML | jpy-ml |
|---|---|
| Wrap REST calls to a Python server | ML runs **in-process** via JNI — zero network latency |
| Manually install Python + pip + torch | **Auto-downloads** Python, all deps, and model weights |
| Parse untyped JSON from model APIs | **Strongly typed** results: `DetectionResult`, `PoseResult`, ... |
| Deploy 2 services (Java app + Python API) | **Single JVM process** — simpler ops, lower cost |
| Limited to ONNX Runtime (CPU only) | **Full PyTorch** — CPU, Apple MPS, NVIDIA CUDA, Multi-GPU |
| Only inference | **Inference + Training + Validation + Export + LLM Fine-tuning** — full lifecycle |

---

## Features

### Core Framework
- **Embedded Python Runtime** — full CPython embedded in JVM via Jep (JNI), auto-managed lifecycle
- **Zero-Config Environment** — auto-downloads Python (production) or uses local venv (dev)
- **Thread-Safe Engine** — singleton PythonEngine with ReadWriteLock, safe for concurrent use
- **Type-Safe Java APIs** — strongly typed configs, results, and callbacks — no `Map` casting in user code
- **Transparent Python Bridge** — `PythonEngine` for arbitrary Python/NumPy when you need it
- **SLF4J Logging** — proper logging framework integration (Logback)
- **Exception Hierarchy** — `JpyMlException` base class with typed exceptions

### Computer Vision (Ultralytics YOLO)
- **Unified Model API** — single `Model` class for all architectures and tasks
- **6 Model Families** — YOLOv8, YOLO11, YOLO26, RT-DETR, SAM, plus ONNX Runtime inference
- **5 Task Types** — Detect, Segment, Classify, Pose Estimation, OBB
- **Full Lifecycle** — predict, train, validate, export (ONNX/TensorRT/CoreML/TFLite/...)
- **Rich Result Types** — BoundingBox, Mask, Keypoint, RotatedBoundingBox with filter/query helpers
- **Device Abstraction** — CPU / MPS (Apple Silicon) / CUDA GPU / Multi-GPU via `Device` class
- **Epoch Callbacks** — real-time training progress with per-epoch loss/fitness metrics
- **Per-Class Validation** — mAP50, mAP50-95, precision, recall broken down by class
- **Image Annotation** — draw results on images via PIL, supports all task types
- **Zero-Copy Bridge** — `TensorBufferPool` + `RawDetectionResult` for high-performance inference
- **GPU Memory Management** — `warmup()`, `unload()`, `reload(device)` APIs
- **Direct Image Input** — `predict(byte[])`, `predict(BufferedImage)` — no temp files needed
- **Async API** — `predictAsync()` returning `CompletableFuture<InferenceResult>`
- **Result Serialization** — `toJson()`, `toMap()` on all result types, no external deps
- **Model Hub** — `Model.preset("yolov8n")` auto-downloads and caches models
- **Java Visualization** — `ImageVisualizer` draws boxes/masks/keypoints in pure Java2D

### SAM 2 — Interactive Segmentation
- **Point/Box Prompts** — segment objects by clicking or drawing bounding boxes
- **Multiple Prompts** — combine positive and negative prompts
- **Video Tracking** — track objects across video frames with temporal memory
- **SAM2Model** — dedicated model class for SAM 2 inference
- **SAM2Result** — typed result with masks and confidence scores
- **SAM2VideoTracker** — video object tracking with per-frame prompts

### SAM 3 — Concept-Level Segmentation
- **Text Prompts** — segment objects using natural language ("person", "red car")
- **Image Exemplars** — find similar objects using a reference image region
- **SAM3Model** — dedicated model class for SAM 3 inference
- **SAM3Result** — typed result with masks, scores, and class IDs
- **CLIP Integration** — uses CLIP for text-to-mask semantic understanding

### OpenCV Integration
- **OpenCVEngine** — Java API for common OpenCV operations
- **Image I/O** — imread, imwrite with format detection
- **Color Conversion** — BGR2GRAY, BGR2RGB, etc.
- **Filtering** — Gaussian blur, Canny edge detection, thresholding
- **Contours** — find and analyze contours
- **Morphology** — erode, dilate, open, close operations

### MediaPipe Integration
- **MediaPipeEngine** — Java API for MediaPipe tasks
- **Hand Tracking** — detect hands with 21 keypoints
- **Face Mesh** — detect face landmarks (478 points)
- **Pose Estimation** — detect body pose with 33 keypoints

### LLM — Large Language Models
- **LLMModel** — unified entry for model download, inference, and fine-tuning
- **HuggingFace Hub** — `LLMModel.download("Qwen/Qwen2.5-0.5B-Instruct")` with local cache
- **Chat Inference** — typed `ChatResponse` with token counts, supports `ChatMessage` role-based API
- **LoRA/QLoRA Fine-Tuning** — parameter-efficient training via PEFT + TRL SFTTrainer
- **Real-Time Callbacks** — per-step training progress via `TrainingCallback`
- **Async Training** — `runAsync()` returning `CompletableFuture<LLMTrainingResult>`
- **Quantization** — NF4/INT8 (CUDA), auto-detect based on platform
- **Auto Device** — CPU / Apple MPS / NVIDIA CUDA auto-detection
- **LoRA Adapter Merge** — `LLMModel.mergeAdapter()` to merge adapter into base model
- **GenerationConfig** — temperature, top-p, max tokens, repetition penalty
- **Auto Dependency Install** — transformers, peft, trl, accelerate installed on first use

---

## Environment

| Component | Version | Notes |
|-----------|---------|-------|
| JDK | Temurin 17 | via sdkman, **NOT GraalVM** (JNI crashes) |
| Python | 3.12 (auto-downloaded) | python-build-standalone, zero user setup |
| Jep (pip + Maven) | 4.3.1 | Java-Python JNI bridge, Maven groupId: `org.ninia` |
| Ultralytics | 8.4.45 | YOLOv8, YOLO11, YOLO26, RT-DETR, SAM |
| PyTorch | 2.11.0 | CPU-only macOS ARM64 |
| OS | Linux, macOS, Windows | CI tested on all three platforms |

**Important:** GraalVM CE's JNI support is incomplete and will crash when loading Jep's native library. Always use standard OpenJDK (Temurin, Zulu, etc.).

### Dependency Management

**Two initialization modes:**

| Mode | Method | Auto-install | Use case |
|------|--------|--------------|----------|
| **Auto-download** | `PythonRuntime.init()` | ✅ Yes | Production, zero Python setup |
| **Local venv** | `PythonRuntime.init(pythonPath, jepLibPath)` | ❌ No | Development, existing Python |

**Auto-download mode** (`PythonRuntime.init()`):
- Downloads python-build-standalone automatically
- Installs all dependencies from `requirements.txt`:
  - `jep>=4.3.1` — JNI bridge
  - `numpy>=1.26` — Numerical computing
  - `ultralytics>=8.4.45` — YOLO/SAM models
  - `opencv-python>=4.6.0` — Image processing
  - `mediapipe>=0.10.0` — Hand/face/pose detection
  - `git+https://github.com/ultralytics/CLIP.git` — CLIP model for SAM 3 text prompts

**Local venv mode** (`PythonRuntime.init(pythonPath, jepLibPath)`):
- Uses existing Python installation
- User must install dependencies manually:
  ```bash
  .venv/bin/pip install -r src/main/resources/requirements.txt
  ```

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

## Quick Start

### 1. Switch to JDK 17

```bash
sdk install java 17.0.19-tem   # First time only
sdk use java 17.0.19-tem
java -version                   # Confirm: openjdk 17.0.19 Temurin
```

### 2. Create Python virtual environment

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
```

### 3. Install Python dependencies

```bash
# Basic dependencies (required)
.venv/bin/pip install jep numpy ultralytics

# Optional: OpenCV for image processing
.venv/bin/pip install opencv-python

# Optional: MediaPipe for hand/face/pose detection
.venv/bin/pip install mediapipe
```

**Or install all at once:**
```bash
.venv/bin/pip install -r src/main/resources/requirements.txt
```

### 4. Build & Test

```bash
mvn clean test

# Expected: Tests run: 110, Failures: 0, Errors: 0, Skipped: 0
```

### 5. Run Demo

```bash
# Basic Python demo
mvn compile exec:java

# YOLO detection
mvn compile exec:java -Dexec.args="/path/to/image.jpg yolov8n.pt"
```

---

## Project Structure

```
jpy-ml/
├── pom.xml
├── .venv/                              # Python venv (gitignored)
│   ├── bin/python3
│   └── lib/python3.13/site-packages/
│       ├── jep/libjep.jnilib           # JNI native library (critical!)
│       ├── ultralytics/
│       └── torch/
│
├── src/main/resources/python/          # Python helper scripts
│   ├── _jpy_init.py                    # Bootstrap, version check
│   ├── _jpy_inference.py               # Load model, predict, extract results
│   ├── _jpy_training.py                # Training helper
│   ├── _jpy_export.py                  # Model export (ONNX, etc.)
│   ├── _jpy_validation.py              # Model validation
│   ├── _jpy_annotation.py              # Image annotation
│   ├── _jpy_streaming.py               # Video/webcam streaming inference
│   ├── _jpy_sam2.py                    # SAM 2 interactive segmentation
│   ├── _jpy_sam2_video.py              # SAM 2 video tracking
│   ├── _jpy_sam3.py                    # SAM 3 concept segmentation
│   ├── _jpy_opencv.py                  # OpenCV operations
│   ├── _jpy_mediapipe.py               # MediaPipe hand/face/pose
│   ├── _jpy_llm_inference.py           # LLM chat inference
│   ├── _jpy_llm_training.py            # LLM LoRA/QLoRA fine-tuning
│   ├── _jpy_llm_download.py            # HuggingFace model download
│   └── requirements.txt                # Python deps for production auto-install
│
├── src/main/java/io/github/javpower/jpyml/
│   ├── Demo.java                       # Quick demo entry point
│   ├── core/                           # Engine layer
│   │   ├── PythonRuntime.java          # Python environment manager
│   │   ├── PythonEngine.java           # Jep bridge (singleton, ReadWriteLock)
│   │   └── PythonScriptLoader.java     # Thread-safe script loader
│   ├── exception/                      # Custom exceptions
│   │   ├── JpyMlException.java         # Base exception
│   │   ├── InferenceException.java
│   │   ├── ModelException.java
│   │   ├── PythonException.java
│   │   ├── TrainingException.java
│   │   └── ValidationException.java
│   ├── ml/
│   │   ├── model/                      # Core model API
│   │   │   ├── Model.java              # **Unified entry point**
│   │   │   ├── ModelConfig.java        # Inference config (conf, iou, imgsz, device, ...)
│   │   │   ├── ModelInfo.java          # Model metadata
│   │   │   ├── TaskType.java           # DETECT, SEGMENT, CLASSIFY, POSE, OBB
│   │   │   ├── Device.java             # CPU / MPS / GPU device selection
│   │   │   ├── SAM2Model.java          # SAM 2 interactive segmentation
│   │   │   ├── SAM2VideoTracker.java   # SAM 2 video tracking
│   │   │   ├── SAM3Model.java          # SAM 3 concept segmentation
│   │   │   └── Prompt.java             # Point/Box/Mask/Text prompts
│   │   │   └── ModelHub.java            # Model registry + auto-download
│   ├── llm/                              # LLM module
│   │   ├── LLMModel.java                 # Model download, load, chat, fine-tune entry
│   │   ├── LLMFineTuner.java             # Fine-tuning builder (LoRA/QLoRA)
│   │   ├── LLMTrainingResult.java        # Training result + merge adapter
│   │   ├── DependencyManager.java        # Auto pip install for LLM deps
│   │   ├── config/                       # LLM configs
│   │   │   ├── LoRAConfig.java           # LoRA rank, alpha, target modules
│   │   │   ├── LLMTrainConfig.java       # Training hyperparameters
│   │   │   ├── GenerationConfig.java     # Inference generation params
│   │   │   └── Quantization.java         # NF4, INT8, NONE, AUTO
│   │   └── data/                         # LLM data types
│   │       ├── ChatMessage.java          # role-based chat message
│   │       └── ChatResponse.java         # Inference response with tokens
│   │   ├── result/                     # Strongly typed results
│   │   │   ├── InferenceResult.java    # Base interface
│   │   │   ├── InferenceSpeed.java     # Pre/inference/postprocess timing
│   │   │   ├── BoundingBox.java        # Axis-aligned box (x1,y1,x2,y2)
│   │   │   ├── RotatedBoundingBox.java # Rotated box (cx,cy,w,h,angle)
│   │   │   ├── ClassPrediction.java    # Class + confidence + box
│   │   │   ├── Mask.java               # Polygon mask
│   │   │   ├── Keypoint.java           # Single keypoint (x,y,conf)
│   │   │   ├── KeypointCollection.java # COCO 17 keypoints
│   │   │   ├── DetectionResult.java
│   │   │   ├── SegmentationResult.java
│   │   │   ├── ClassificationResult.java
│   │   │   ├── PoseResult.java
│   │   │   ├── OBBResult.java
│   │   │   ├── OBBPrediction.java
│   │   │   ├── SAM2Result.java         # SAM 2 segmentation result
│   │   │   ├── SAM2VideoResult.java    # SAM 2 video tracking result
│   │   │   ├── SAM3Result.java         # SAM 3 concept segmentation result
│   │   │   ├── ResultSerializer.java   # JSON/Map serialization
│   │   │   ├── RawDetectionResult.java # Zero-copy detection result
│   │   │   ├── RawInferenceResult.java # Zero-copy result interface
│   │   │   ├── TensorBufferPool.java   # DirectByteBuffer pool
│   │   │   └── StreamFrame.java        # Video frame with result
│   │   ├── training/                   # Training API
│   │   │   ├── TrainingConfig.java     # Full parameter builder
│   │   │   ├── TrainingResult.java     # Training result + epoch metrics
│   │   │   ├── TrainingCallback.java   # Epoch callback interface
│   │   │   ├── EpochMetric.java        # Per-epoch loss/fitness record
│   │   │   ├── AugmentationConfig.java # Data augmentation settings
│   │   │   └── OptimizerType.java      # AUTO, SGD, ADAM, ADAMW
│   │   ├── export/                     # Export API
│   │   │   ├── ExportConfig.java
│   │   │   ├── ExportResult.java
│   │   │   └── ExportFormat.java       # ONNX, TORCHSCRIPT, COREML, ...
│   │   ├── validation/                 # Validation API
│   │   │   ├── ValidationResult.java   # mAP50, mAP50-95, precision, recall, per-class
│   │   │   └── PerClassMetric.java     # Per-class mAP record
│   │   └── annotation/                 # Image annotation
│   │       ├── ImageAnnotator.java     # Draw results via Python PIL
│   │       └── ImageVisualizer.java   # Java2D result visualization
│   ├── cv/                             # OpenCV integration
│   │   └── OpenCVEngine.java           # OpenCV operations
│   ├── mp/                             # MediaPipe integration
│   │   └── MediaPipeEngine.java        # Hand/face/pose detection
│   └── util/
│       └── ImageUtils.java             # Image resize/crop/convert via PIL
│
├── .github/workflows/                  # CI/CD
│   ├── ci.yml                          # Multi-platform CI
│   └── release.yml                     # Maven Central release
│
└── src/test/java/io/github/javpower/jpyml/
    ├── QuickVerifyTest.java             # Core bridge tests (10 cases)
    ├── PythonEngineTest.java            # Engine tests (11 cases)
    ├── PythonRuntimeTest.java           # Platform detection (3 cases)
    ├── ModelIntegrationTest.java        # Full YOLO integration (18 cases)
    ├── SAMIntegrationTest.java          # SAM 2/3 integration (9 cases)
    ├── LLMIntegrationTest.java          # LLM download, chat, fine-tune, async (4 cases)
    ├── NewFeaturesTest.java             # Serialization, byte[], async, hub, viz (17 cases)
    ├── OpenCVEngineTest.java            # OpenCV operations (8 cases)
    ├── MediaPipeEngineTest.java         # Hand/face/pose detection (4 cases)
    ├── StreamRealtimeTest.java          # Real-time stream tests
    └── ml/result/                       # Unit tests
        ├── TensorBufferPoolTest.java    # Buffer pool tests (6 cases)
        ├── BoundingBoxTest.java         # BoundingBox tests (10 cases)
        ├── KeypointTest.java            # Keypoint tests (5 cases)
        └── InferenceSpeedTest.java      # Speed tests (5 cases)
```

---

## API Usage

### Model Loading with Task Override and Device

```java
// Auto-detect task from model file
try (Model model = new Model("yolov8n.pt")) { ... }

// Explicit task type (for custom-named models)
try (Model model = new Model("my_custom_detector.pt", TaskType.DETECT)) { ... }

// Specify device: CPU, MPS (Apple Silicon), GPU
ModelConfig config = new ModelConfig()
    .device(Device.cpu());
ModelConfig config = new ModelConfig()
    .device(Device.mps());
ModelConfig config = new ModelConfig()
    .device(Device.gpu(0));
ModelConfig config = new ModelConfig()
    .device(Device.cuda(0));
ModelConfig config = new ModelConfig()
    .device(Device.fromString("cuda:0"));
```

### Object Detection

```java
PythonRuntime.init(pythonPath, jepLibPath);

try (Model model = new Model("yolov8n.pt")) {
    System.out.println(model.getModelInfo());  // ModelInfo{task=DETECT, classes=80, ...}

    DetectionResult result = model.predict("photo.jpg");

    for (ClassPrediction pred : result.getBoxes()) {
        System.out.println(pred);
        // person 92.3% BoundingBox[x1=100.5, y1=50.2, x2=300.1, y2=400.8]
    }

    // Filter results
    List<ClassPrediction> persons = result.filterByClass("person");
    List<ClassPrediction> confident = result.filterByConfidence(0.8f);

    // Timing
    InferenceSpeed speed = result.getSpeed();
    System.out.printf("Inference: %.1fms%n", speed.inferenceMs());
}
```

### Instance Segmentation

```java
try (Model model = new Model("yolov8n-seg.pt")) {
    SegmentationResult result = model.predict("photo.jpg");

    for (int i = 0; i < result.getBoxes().size(); i++) {
        ClassPrediction pred = result.getBoxes().get(i);
        Mask mask = result.getMasks().get(i);
        System.out.println(pred.className() + " mask points: " + mask.getPointCount());
    }
}
```

### Image Classification

```java
try (Model model = new Model("yolov8n-cls.pt")) {
    ClassificationResult result = model.predict("photo.jpg");

    System.out.println("Top prediction: " + result.getTop1ClassName());
    System.out.println("Confidence: " + result.getTop1Confidence());
}
```

### Pose Estimation

```java
try (Model model = new Model("yolov8n-pose.pt")) {
    PoseResult result = model.predict("photo.jpg");

    for (int i = 0; i < result.personCount(); i++) {
        KeypointCollection kpts = result.getKeypoints().get(i);
        Keypoint nose = kpts.getNose();       // COCO keypoint #0
        Keypoint lShoulder = kpts.get(5);     // COCO keypoint #5
        System.out.printf("Person %d: nose=(%.1f,%.1f)%n", i, nose.x(), nose.y());
    }
}
```

### Oriented Bounding Boxes (OBB)

```java
try (Model model = new Model("yolov8n-obb.pt")) {
    OBBResult result = model.predict("satellite_image.jpg");

    for (OBBPrediction pred : result.getPredictions()) {
        RotatedBoundingBox box = pred.box();
        System.out.printf("%s %.1f%% at (%.1f,%.1f) %.1fx%.1f angle=%.1f%n",
            pred.className(), pred.confidence() * 100,
            box.centerX(), box.centerY(), box.width(), box.height(),
            box.angleDegrees());
    }
}
```

### Inference Configuration

```java
ModelConfig config = new ModelConfig()
    .confidence(0.5f)        // Confidence threshold
    .iouThreshold(0.7f)      // NMS IoU threshold
    .imageSize(640)           // Input image size
    .maxDetections(100)       // Max detections per image
    .device(Device.gpu(0))   // CPU / MPS / GPU device
    .augment(true)            // Test-time augmentation
    .agnosticNms(true)        // Class-agnostic NMS
    .filterClasses(0, 2, 5)   // Only detect persons, cars, buses
    .retinaMasks(true)        // High-quality segmentation masks
    .half(true)               // FP16 inference (GPU)
    .verbose(false)           // Suppress output
    .save(true)               // Save results to disk
    .saveTxt(true)            // Save as .txt
    .saveCrop(true)           // Save cropped predictions
    .embed(0, 1, 2);          // Extract feature embeddings

InferenceResult result = model.predict("photo.jpg", config);
```

### Model Training

```java
TrainingConfig config = new TrainingConfig()
    .dataConfig("coco128.yaml")
    .epochs(50)
    .batchSize(16)
    .device(Device.gpu(0))       // Train on GPU
    .optimizer(OptimizerType.ADAMW)
    .learningRate(0.001f)
    .augmentation(new AugmentationConfig()
        .mosaic(true)
        .fliplr(0.5f)
        .hsvH(0.015f));

// With epoch callback
TrainingResult result = model.train(config, (epoch, log) -> {
    System.out.println("Epoch " + epoch + ": " + log);
});
System.out.println("Best model: " + result.getBestModelPath());
System.out.println("Best fitness: " + result.getBestFitness());

// Epoch metrics
for (EpochMetric m : result.getEpochMetrics()) {
    System.out.printf("Epoch %d: box=%.4f cls=%.4f dfl=%.4f%n",
        m.epoch(), m.boxLoss(), m.clsLoss(), m.dflLoss());
}
```

### Model Validation

```java
ValidationResult val = model.validate("coco128.yaml");
System.out.printf("mAP50=%.3f, mAP50-95=%.3f, P=%.3f, R=%.3f%n",
    val.getMAP50(), val.getMAP5095(), val.getPrecision(), val.getRecall());

// Per-class metrics
for (PerClassMetric pc : val.getPerClassMetrics()) {
    System.out.println(pc);  // "person (id=0): mAP50-95=0.7234"
}
```

### Model Export

```java
ExportResult exported = model.export(ExportFormat.ONNX);
System.out.println("Exported to: " + exported.getOutputPath());
System.out.println("File size: " + exported.getFileSizeMB());
```

### Image Annotation

```java
ImageAnnotator annotator = new ImageAnnotator();

try (Model model = new Model("yolov8n.pt")) {
    InferenceResult result = model.predict("photo.jpg");
    String annotated = annotator.annotate(result, "output.jpg");
}
```

### Batch Inference

```java
try (Model model = new Model("yolov8n.pt")) {
    List<String> images = List.of("photo1.jpg", "photo2.jpg", "photo3.jpg");
    List<InferenceResult> results = model.predict(images);

    for (int i = 0; i < results.size(); i++) {
        System.out.println("Image " + i + ": " + results.get(i).count() + " objects");
    }
}
```

### Video Stream Inference

```java
try (Model model = new Model("yolov8n.pt")) {
    // Video file — processes frame-by-frame with chunked streaming
    model.predictVideo("video.mp4", frame -> {
        System.out.println("Frame: " + frame.count() + " objects");
    });

    // Webcam — real-time inference (blocks current thread)
    // Call stopStream() from another thread to stop
    new Thread(() -> {
        Thread.sleep(10000);
        model.stopStream();  // Stop after 10 seconds
    }).start();

    model.predictStream(0, frame -> {
        // frame 0 = default webcam
        if (frame instanceof DetectionResult dr) {
            System.out.println("Camera: " + dr.getBoxes().size() + " objects");
        }
    });
}
```

### Zero-Copy Inference (High Performance)

```java
try (Model model = new Model("yolov8n.pt")) {
    // Zero-copy prediction for detection tasks
    RawDetectionResult result = model.predictRaw("photo.jpg");

    // Option 1: Strongly-typed access (lazy-loaded)
    for (ClassPrediction pred : result.getBoxes()) {
        System.out.println(pred);
    }

    // Option 2: Raw buffer access (zero allocation)
    FloatBuffer xyxy = result.getRawBoxesXYXY();  // (N, 4) float buffer
    FloatBuffer conf = result.getRawConfidences();  // (N,) float buffer
    IntBuffer cls = result.getRawClassIds();        // (N,) int buffer

    for (int i = 0; i < result.getBoxCount(); i++) {
        int offset = i * 4;
        System.out.printf("Box: (%.1f,%.1f,%.1f,%.1f) conf=%.2f cls=%d%n",
            xyxy.get(offset), xyxy.get(offset+1),
            xyxy.get(offset+2), xyxy.get(offset+3),
            conf.get(i), cls.get(i));
    }

    // Release buffers back to pool for reuse
    result.release();
}
```

### GPU Memory Management

```java
try (Model model = new Model("yolov8n.pt")) {
    // Warmup: run dummy inference to trigger CUDA kernel compilation
    model.warmup();

    // Inference...
    model.predict("photo.jpg");

    // Unload model from GPU to free memory
    model.unload();

    // Reload to GPU
    model.reload("cuda:0");
}
```

### SAM 2 — Interactive Segmentation

```java
// Point prompt: segment what's at (320, 240)
try (SAM2Model sam = new SAM2Model("sam2.1_t.pt")) {
    SAM2Result result = sam.predict("photo.jpg",
        Prompt.point(320, 240)
    );

    for (Mask mask : result.getMasks()) {
        System.out.println("Mask points: " + mask.getPointCount());
    }
    System.out.println("Best score: " + result.bestScore());
}

// Box prompt: segment inside bounding box
SAM2Result result = sam.predict("photo.jpg",
    Prompt.box(100, 100, 400, 400)
);

// Multiple prompts with negative points
SAM2Result result = sam.predict("photo.jpg",
    Prompt.point(200, 200),                          // positive
    Prompt.point(400, 300, Prompt.Label.NEGATIVE)    // negative
);
```

### SAM 2 — Video Tracking

```java
try (SAM2Model sam = new SAM2Model("sam2.1_t.pt")) {
    // Start tracking with initial bounding box
    try (SAM2VideoTracker tracker = sam.trackVideo("video.mp4",
            Prompt.box(100, 100, 400, 400))) {

        // Add more prompts at specific frames
        tracker.addPrompt(10, Prompt.point(300, 200));

        // Propagate through all frames
        SAM2VideoResult result = tracker.propagate();
        System.out.println("Tracked " + result.trackedFrameCount() + " frames");
    }
}
```

### SAM 3 — Concept-Level Segmentation

```java
// Text-based concept segmentation
try (SAM3Model sam = new SAM3Model("sam3.pt")) {
    SAM3Result result = sam.predictText("street.jpg", "person", "bus");

    for (Mask mask : result.getMasks()) {
        System.out.println("Mask: " + mask.getPointCount() + " points");
    }

    // Filter by confidence score
    SAM3Result filtered = result.filterByScore(0.5f);
}

// Image exemplar: "find things like this"
try (SAM3Model sam = new SAM3Model("sam3.pt")) {
    BoundingBox exemplarBox = new BoundingBox(10, 20, 200, 300);
    SAM3Result result = sam.predictExemplar("target.jpg", "reference.jpg",
        exemplarBox);
}
```

### OpenCV — Image Processing

```java
OpenCVEngine cv = new OpenCVEngine();

// Read image info
OpenCVEngine.ImageInfo info = cv.imread("photo.jpg");
System.out.println(info.width() + "x" + info.height());

// Color conversion
cv.cvtColor("photo.jpg", "gray.jpg", "BGR2GRAY");

// Edge detection
cv.canny("photo.jpg", "edges.jpg", 100, 200);

// Gaussian blur
cv.blur("photo.jpg", "blurred.jpg", 15);

// Find contours
OpenCVEngine.ContourResult contours = cv.findContours("binary.jpg", "contours.jpg");
System.out.println("Found " + contours.count() + " contours");
```

### MediaPipe — Hand/Face/Pose Detection

```java
MediaPipeEngine mp = new MediaPipeEngine();

// Hand detection
MediaPipeEngine.HandResult hands = mp.detectHands("hand.jpg");
for (MediaPipeEngine.HandResult.Hand hand : hands.hands()) {
    for (MediaPipeEngine.HandResult.Landmark lm : hand.landmarks()) {
        System.out.printf("Landmark: (%.3f, %.3f, %.3f)%n", lm.x(), lm.y(), lm.z());
    }
}

// Face mesh
MediaPipeEngine.FaceResult faces = mp.detectFace("face.jpg");
System.out.println("Faces found: " + faces.count());

// Pose estimation
MediaPipeEngine.PoseResult pose = mp.detectPose("pose.jpg");
System.out.println("Pose landmarks: " + pose.count());
```

### Model Preset — Auto-Download

```java
// Auto-download + load in one line
try (Model model = Model.preset("yolov8n")) {
    DetectionResult result = model.predict("photo.jpg");
}

// List available models
for (ModelHub.ModelEntry entry : ModelHub.listAvailable()) {
    System.out.println(entry);  // "yolov8n (6.2 MB, DETECT)"
}
```

### Direct Image Input — byte[] / BufferedImage

```java
try (Model model = new Model("yolov8n.pt")) {
    // From byte[] (e.g., HTTP upload, file read)
    byte[] imageData = Files.readAllBytes(Path.of("photo.jpg"));
    InferenceResult result = model.predict(imageData);

    // From BufferedImage (e.g., Java image processing)
    BufferedImage image = ImageIO.read(new File("photo.jpg"));
    InferenceResult result = model.predict(image);
}
```

### Async Prediction

```java
try (Model model = new Model("yolov8n.pt")) {
    // Returns CompletableFuture
    CompletableFuture<InferenceResult> future = model.predictAsync("photo.jpg");
    future.thenAccept(result -> {
        System.out.println("Detected " + result.count() + " objects");
    });
}
```

### Result Serialization — JSON / Map

```java
DetectionResult result = model.predict("photo.jpg");

// JSON output (no external dependencies)
String json = result.toJson();
// {"task":"detect","source":"photo.jpg","count":3,"boxes":[...]}

// Map output (for Jackson/Gson integration)
Map<String, Object> map = result.toMap();

// Works on SAM results too
SAM2Result samResult = sam.predict("photo.jpg", Prompt.point(100, 200));
String samJson = samResult.toJson();
```

### Java Visualization — ImageVisualizer

```java
ImageVisualizer viz = new ImageVisualizer()
    .lineWidth(2.5f)
    .fontSize(14.0f)
    .maskAlpha(0.4f);

BufferedImage image = ImageIO.read(new File("photo.jpg"));
InferenceResult result = model.predict("photo.jpg");

// Draw boxes/masks/keypoints on image
BufferedImage annotated = viz.visualize(image, result);
ImageIO.write(annotated, "jpg", new File("output.jpg"));

// Or from byte[]
byte[] annotatedBytes = viz.visualizeToBytes(imageBytes, result);
```

### LLM — Download & Chat Inference

```java
// Download model from HuggingFace Hub (cached at ~/.jpy-ml/llm-models/)
LLMModel model = LLMModel.download("Qwen/Qwen2.5-0.5B-Instruct");

// Or load from local path
LLMModel model = LLMModel.load("/path/to/local/model");

// Chat inference
ChatResponse response = model.chat(
    ChatMessage.system("You are a helpful assistant"),
    ChatMessage.user("Hello, introduce yourself in one sentence")
);

System.out.println(response.getContent());
System.out.println("Tokens: prompt=" + response.getPromptTokens()
    + " completion=" + response.getCompletionTokens());

// With generation config
ChatResponse response = model.chat(
    List.of(
        ChatMessage.system("You are a helpful assistant"),
        ChatMessage.user("Explain quantum computing")
    ),
    GenerationConfig.create()
        .maxNewTokens(256)
        .temperature(0.7)
        .topP(0.9)
        .repetitionPenalty(1.1)
);
```

### LLM — LoRA Fine-Tuning

```java
LLMModel model = LLMModel.load("Qwen/Qwen2.5-0.5B-Instruct")
    .quantize(Quantization.NONE); // macOS

// Synchronous fine-tuning with real-time callbacks
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

System.out.println("Adapter saved to: " + result.getAdapterPath());
System.out.println("Final loss: " + result.getFinalLoss());
```

### LLM — Load Adapter & Inference

```java
// Load base model with trained LoRA adapter
LLMModel finetuned = LLMModel.load("Qwen/Qwen2.5-0.5B-Instruct")
    .adapter("/path/to/adapter");

ChatResponse response = finetuned.chat(
    ChatMessage.user("What is your name?")
);
System.out.println(response.getContent());
```

### LLM — Async Fine-Tuning

```java
CompletableFuture<LLMTrainingResult> future = model.fineTune()
    .lora(LoRAConfig.create().rank(4).alpha(8))
    .dataset("data.jsonl")
    .config(LLMTrainConfig.create().epochs(2))
    .runAsync((step, log) -> {
        System.out.println("[async] " + log);
    });

// Do other work...

LLMTrainingResult result = future.get(10, TimeUnit.MINUTES);
```

### LLM — Merge Adapter into Base Model

```java
LLMTrainingResult result = model.fineTune()
    .dataset("data.jsonl")
    .config(LLMTrainConfig.create().epochs(3))
    .run();

// Merge LoRA adapter into base model for standalone deployment
String mergedPath = LLMModel.mergeAdapter(
    model.getModelPath(),
    result.getAdapterPath(),
    "/path/to/merged-model"
);
```

### LLM — Training Data Format (JSONL)

```json
{"messages": [{"role": "user", "content": "1+1=?"}, {"role": "assistant", "content": "1+1=2"}]}
{"messages": [{"role": "user", "content": "What is Java?"}, {"role": "assistant", "content": "Java is a programming language."}]}
```

Also supports instruction format:
```json
{"instruction": "Translate to English", "input": "你好", "output": "Hello"}
```

### Basic Python Operations

```java
PythonRuntime.init(pythonPath, jepLibPath);
PythonEngine engine = PythonEngine.getInstance();

// Variables
engine.put("name", "World");
engine.exec("greeting = f'Hello, {name}!'");
String msg = engine.get("greeting");  // "Hello, World!"

// Functions
engine.exec("def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n");
long fib10 = engine.eval("fib(10)");  // 55

// NumPy
engine.exec("import numpy as np");
engine.exec("arr = np.arange(10).reshape(2, 5).tolist()");
List<?> data = engine.get("arr");

// Stdout capture
engine.exec("for i in range(3): print(f'item {i}')",
    line -> System.out.println("[py] " + line),
    null  // stderr callback
);
```

---

## Architecture

```
┌──────────────────────────────────────────┐
│            User Java Code                │
│  Model m = new Model("yolov8n.pt");      │
│  DetectionResult r = m.predict(img);     │
├──────────────────────────────────────────┤
│  Model / ModelConfig / Result types      │
│  (58 Java source files)                  │
├──────────────────────────────────────────┤
│  PythonEngine (singleton, ReadWriteLock) │
│  SharedInterpreter + sys.path filtering  │
├──────────────────────────────────────────┤
│  Jep 4.3.1 (JNI bridge)                 │
│  libjep.jnilib from pip install jep      │
├──────────────────────────────────────────┤
│  Python Helper Scripts (15 .py files)    │
│  jpy_load_model / jpy_extract_result /   │
│  jpy_train / jpy_export / jpy_validate / │
│  jpy_sam2 / jpy_sam3 / jpy_opencv /      │
│  jpy_mediapipe / jpy_llm_* / ...         │
├──────────────────────────────────────────┤
│  CPython 3.13 (.venv)                    │
│  Ultralytics 8.4.45 + PyTorch 2.11       │
└──────────────────────────────────────────┘
```

**Data flow for inference:**
1. Java `Model.predict(path)` → puts path into PythonEngine
2. Calls `model_var(image_path)` via Jep → Ultralytics runs inference
3. Python `jpy_extract_result()` converts tensors to plain dicts/lists
4. Java `buildResult()` creates typed result objects (DetectionResult, etc.)

---

## Test Results

All 110 tests passing (0 skipped):

| Test Suite | Tests | Pass | Skip | Description |
|-----------|-------|------|------|-------------|
| QuickVerifyTest | 10 | 10 | 0 | Basic Python bridge (eval, put/get, numpy, lists, dicts) |
| PythonEngineTest | 11 | 11 | 0 | Engine features (threads, callbacks, modules) |
| PythonRuntimeTest | 3 | 3 | 0 | Platform detection |
| ModelIntegrationTest | 18 | 18 | 0 | Full YOLO integration (inference + batch + video + training + export) |
| SAMIntegrationTest | 9 | 9 | 0 | SAM 2/3 integration (point/box/video/text/exemplar) |
| LLMIntegrationTest | 4 | 4 | 0 | LLM download, chat, LoRA fine-tuning, async training |
| NewFeaturesTest | 17 | 17 | 0 | Serialization, byte[] input, async, ModelHub, visualization |
| TensorBufferPoolTest | 6 | 6 | 0 | Zero-copy buffer pool |
| BoundingBoxTest | 10 | 10 | 0 | BoundingBox record |
| KeypointTest | 5 | 5 | 0 | Keypoint record |
| InferenceSpeedTest | 5 | 5 | 0 | InferenceSpeed record |
| MediaPipeEngineTest | 4 | 4 | 0 | Hand/face/pose detection |
| OpenCVEngineTest | 8 | 8 | 0 | Image processing operations |

### ModelIntegrationTest details:

| Test | Model | Task | Result |
|------|-------|------|--------|
| testModelLoad | yolov8n.pt | — | Model loads, info correct |
| testDetection | yolov8n.pt | detect | 6 objects (bus, persons, stop sign) |
| testDetectionWithConfig | yolov8n.pt | detect | 3 objects with conf>0.7, imgsz=320 |
| testSegmentation | yolov8n-seg.pt | segment | 6 objects + 6 masks |
| testClassification | yolov8n-cls.pt | classify | Top-1 prediction |
| testPose | yolov8n-pose.pt | pose | 4 persons, 17 keypoints each |
| testOBB | yolov8n-obb.pt | obb | Model loads and runs |
| testModelClose | yolov8n.pt | — | Close lifecycle works |
| testInferenceSpeed | yolov8n.pt | detect | Timing captured correctly |
| testBatchPrediction | yolov8n.pt | detect | 3 images batch, 6 objects each |
| testVideoPrediction | yolov8n.pt | detect | Video streaming with URL |
| testStreamWithAnnotatedImage | yolov8n.pt | detect | Stream with annotated frames |
| testYOLO26Detection | yolo26n.pt | detect | 5 objects (YOLO26 next-gen model) |
| testOnnxInference | yolov8n.onnx | detect | ONNX Runtime inference, 5 objects |
| testExportOnnx | yolov8n.pt | — | Export to ONNX |
| testTrainWithCallback | yolov8n.pt | train | 2 epochs, callback + epoch metrics |
| testValidate | yolov8n.pt | val | mAP50, per-class metrics |
| testTrainThenPredict | best.pt | detect | Train then predict end-to-end |

### SAMIntegrationTest details:

| Test | Model | Prompt | Result |
|------|-------|--------|--------|
| testSAM2PointPrompt | sam2.1_t.pt | Point(320,240) | 1 mask, score=0.52 |
| testSAM2BoxPrompt | sam2.1_t.pt | Box(100,100,400,400) | 1 mask |
| testSAM2MultiplePrompts | sam2.1_t.pt | Point + Negative Point | 1 mask |
| testSAM2ModelClose | sam2.1_t.pt | — | Close lifecycle works |
| testSAM2VideoTracker | sam2.1_t.pt | Box + Point | Video tracking, frame-by-frame |
| testSAM3TextPrompt | sam3.pt | Text("person","bus") | Concept segmentation with CLIP |
| testSAM3ExemplarPrompt | sam3.pt | Image exemplar | Exemplar-based segmentation |
| testSAM3FilterByScore | sam3.pt | Text("person") | Score filtering works |
| testSAM3ModelClose | sam3.pt | — | Close lifecycle works |

### LLMIntegrationTest details:

| Test | Model | Task | Result |
|------|-------|------|--------|
| testDownloadModel | Qwen2.5-0.5B-Instruct | download | Model cached locally |
| testChatInference | Qwen2.5-0.5B-Instruct | chat | Response with token counts |
| testFineTuneWithCallback | Qwen2.5-0.5B-Instruct | LoRA fine-tune | Adapter saved, callback events received |
| testAsyncFineTune | Qwen2.5-0.5B-Instruct | async LoRA | CompletableFuture completes |

### MediaPipeEngineTest details:

| Test | Task | Result |
|------|------|--------|
| testDetectHands | Hand detection | 21 keypoints per hand |
| testDetectFace | Face mesh | 478 face landmarks |
| testDetectPose | Pose estimation | 33 pose landmarks |
| testClose | Lifecycle | Close works |

### OpenCVEngineTest details:

| Test | Operation | Result |
|------|-----------|--------|
| testImread | Image read | Width/height/channels |
| testImwrite | Image write | Output file created |
| testCvtColor | BGR2GRAY | Grayscale image |
| testResize | Resize | Resized image |
| testBlur | Gaussian blur | Blurred image |
| testCanny | Edge detection | Edge image |
| testThreshold | Threshold | Binary image |
| testFindContours | Contours | Contour count |

---

## Key Design Decisions

1. **Singleton PythonEngine** — Jep limits one SharedInterpreter per JVM. All Model instances share it via unique variable name prefixes (`_jpy_mv0`, `_jpy_mv1`, etc.).

2. **Python scripts do the heavy lifting** — Complex tensor-to-dict conversion happens in Python (`_jpy_inference.py`), Java receives only plain `Map<String, Object>`.

3. **sys.path filtering** — When Jep starts via Homebrew Python, it inherits system site-packages paths that conflict with venv packages. PythonEngine automatically filters these and injects venv site-packages at priority.

4. **AutoCloseable Model** — `Model implements AutoCloseable`. On close, Python variables are set to `None` for garbage collection.

5. **Builder pattern** — All config classes (`ModelConfig`, `TrainingConfig`, `ExportConfig`, `AugmentationConfig`) use chainable setters.

---

## Troubleshooting

### JVM Crash (Abort trap: 6)
**Cause:** Using GraalVM CE.
**Fix:** Switch to standard OpenJDK (Temurin).
```bash
sdk use java 17.0.19-tem
```

### "Jep native library not found"
**Cause:** jep not installed in the venv.
**Fix:**
```bash
.venv/bin/pip install jep
```

### "ultralytics not installed" or opencv dlopen errors
**Cause:** Homebrew site-packages shadowing venv packages in sys.path.
**Fix:** Ensure you're using `PythonRuntime.init(pythonPath, jepLibPath)` with venv paths. The PythonEngine automatically filters conflicting paths.

### "JepConfig must be set before creating any SharedInterpreters"
**Cause:** SharedInterpreter.setConfig() called after interpreter already exists.
**Fix:** Use `PythonEngine.getInstance()` instead of `create()`. The singleton pattern handles this.

### "externally-managed-environment" pip error
**Cause:** Using Homebrew's system Python directly.
**Fix:** Always use the project venv: `.venv/bin/pip install ...`

### "PythonEngine is closed"
**Cause:** Calling methods after engine.close() or after create() replaced the singleton.
**Fix:** Get a fresh instance with `PythonEngine.getInstance()`.

---

## Roadmap

jpy-ml is designed as a universal Java-Python ML bridge. YOLO is the first engine — many more are coming.

### Completed
- [x] Batch inference API (`model.predict(List<String>)`)
- [x] Webcam / video stream real-time inference
- [x] python-build-standalone auto-download (zero Python install for end users)
- [x] PyTorch tensor zero-copy bridge (`TensorBufferPool` + `RawDetectionResult`)
- [x] GPU memory management and model warmup APIs
- [x] SAM 2 interactive segmentation (point/box prompts)
- [x] SAM 2 video tracking (per-frame prompts with temporal memory)
- [x] SAM 3 concept-level segmentation (text prompts + image exemplars)
- [x] OpenCV integration (image processing)
- [x] MediaPipe integration (hand/face/pose)
- [x] SLF4J logging framework
- [x] Exception hierarchy (`JpyMlException` base)
- [x] CI/CD (GitHub Actions)
- [x] Result serialization (toJson / toMap)
- [x] Direct image input (byte[] / BufferedImage)
- [x] Model Hub auto-download (Model.preset)
- [x] Java2D result visualization (ImageVisualizer)
- [x] Async prediction API (predictAsync)
- [x] LLM chat inference (HuggingFace Transformers)
- [x] LoRA/QLoRA fine-tuning with real-time step callbacks
- [x] Async fine-tuning API (runAsync)
- [x] LoRA adapter merge into base model
- [x] Auto dependency installation for LLM (transformers, peft, trl, accelerate)

### Next
- [ ] Unit tests for all value types

### Planned ML Engines

#### Other Engines
- [ ] **Whisper** — speech-to-text, automatic speech recognition
- [ ] **Stable Diffusion / FLUX** — image generation, inpainting, controlnet

### Infrastructure
- [ ] Model registry / hub integration (download from URL)
- [ ] Spring Boot starter auto-configuration
- [ ] GraalVM native image support
