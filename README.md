# jpy-ml

**jpy-ml** is a production-grade Java framework that seamlessly bridges the Python ML ecosystem into Java.
It embeds a full Python runtime via JNI, providing zero-config environment management and wrapping
powerful ML libraries behind clean, type-safe Java APIs.

The vision: **bring the entire Python AI/ML world to Java developers — no Python knowledge required.**

Currently implements first-class support for **Ultralytics YOLO** (v8/v11/v26, RT-DETR, SAM) covering
object detection, segmentation, classification, pose estimation, oriented bounding boxes — with
inference, training, validation, export, and ONNX Runtime all fully wired.

---

## Features

### Core Framework
- **Embedded Python Runtime** — full CPython embedded in JVM via Jep (JNI), auto-managed lifecycle
- **Zero-Config Environment** — auto-downloads Python (production) or uses local venv (dev)
- **Thread-Safe Engine** — singleton PythonEngine with ReadWriteLock, safe for concurrent use
- **Type-Safe Java APIs** — strongly typed configs, results, and callbacks — no `Map` casting in user code
- **Transparent Python Bridge** — `PythonEngine` for arbitrary Python/NumPy when you need it

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

---

## Environment

| Component | Version | Notes |
|-----------|---------|-------|
| JDK | Temurin 17 | via sdkman, **NOT GraalVM** (JNI crashes) |
| Python | 3.13 (venv) | Project-local `.venv/` |
| Jep (pip + Maven) | 4.3.1 | Java-Python JNI bridge, Maven groupId: `org.ninia` |
| Ultralytics | 8.4.45 | YOLOv8, YOLO11, YOLO26, RT-DETR, SAM |
| PyTorch | 2.11.0 | CPU-only macOS ARM64 |
| OS | macOS ARM64 (Apple Silicon) | |

**Important:** GraalVM CE's JNI support is incomplete and will crash when loading Jep's native library. Always use standard OpenJDK (Temurin, Zulu, etc.).

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
.venv/bin/pip install jep numpy ultralytics
```

### 4. Build & Test

```bash
mvn clean test

# Expected: Tests run: 39, Failures: 0, Errors: 0, Skipped: 0
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
│   └── _jpy_annotation.py              # Image annotation
│
├── src/main/java/io/github/javpower/jpyml/
│   ├── Demo.java                       # Quick demo entry point
│   ├── core/                           # Engine layer
│   │   ├── PythonRuntime.java          # Python environment manager
│   │   ├── PythonEngine.java           # Jep bridge (singleton, ReadWriteLock)
│   │   ├── PythonModule.java           # Isolated module context
│   │   └── PythonScriptLoader.java     # Thread-safe script loader
│   ├── exception/                      # Custom exceptions
│   │   ├── InferenceException.java
│   │   ├── ModelException.java
│   │   ├── PythonException.java
│   │   └── TrainingException.java
│   ├── ml/
│   │   ├── model/                      # Core model API
│   │   │   ├── Model.java              # **Unified entry point**
│   │   │   ├── ModelConfig.java        # Inference config (conf, iou, imgsz, device, ...)
│   │   │   ├── ModelInfo.java          # Model metadata
│   │   │   ├── TaskType.java           # DETECT, SEGMENT, CLASSIFY, POSE, OBB
│   │   │   └── Device.java             # CPU / MPS / GPU device selection
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
│   │   │   └── OBBPrediction.java
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
│   │       └── ImageAnnotator.java     # Draw results on images
│   └── util/
│       └── ImageUtils.java             # Image resize/crop/convert via PIL
│
└── src/test/java/io/github/javpower/jpyml/
    ├── QuickVerifyTest.java             # Core bridge tests (10 cases)
    ├── PythonEngineTest.java            # Engine tests (11 cases)
    ├── PythonRuntimeTest.java           # Platform detection (3 cases)
    └── ModelIntegrationTest.java        # Full YOLO integration (10 cases)
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
│  (41 Java source files)                  │
├──────────────────────────────────────────┤
│  PythonEngine (singleton, ReadWriteLock) │
│  SharedInterpreter + sys.path filtering  │
├──────────────────────────────────────────┤
│  Jep 4.3.1 (JNI bridge)                 │
│  libjep.jnilib from pip install jep      │
├──────────────────────────────────────────┤
│  Python Helper Scripts (6 .py files)     │
│  jpy_load_model / jpy_extract_result /   │
│  jpy_train / jpy_export / jpy_validate   │
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

All 39 tests passing:

| Test Suite | Tests | Description |
|-----------|-------|-------------|
| QuickVerifyTest | 10 | Basic Python bridge (eval, put/get, numpy, lists, dicts) |
| PythonEngineTest | 11 | Engine features (multithreading, callbacks, modules) |
| PythonRuntimeTest | 3 | Platform detection |
| ModelIntegrationTest | 15 | Full YOLO integration (inference + training + export) |

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
| testYOLO26Detection | yolo26n.pt | detect | 5 objects (YOLO26 next-gen model) |
| testOnnxInference | yolov8n.onnx | detect | ONNX Runtime inference, 5 objects |
| testExportOnnx | yolov8n.pt | — | Export to ONNX (12.3 MB) |
| testTrainWithCallback | yolov8n.pt | train | 2 epochs, callback + epoch metrics |
| testValidate | yolov8n.pt | val | mAP50=0.60, 80 per-class metrics |
| testTrainThenPredict | best.pt | detect | Train then predict end-to-end |

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

### Next
- [ ] PyTorch tensor zero-copy bridge for high-performance data transfer
- [ ] Batch inference API (`model.predict(List<String>)`)
- [ ] Webcam / video stream real-time inference
- [ ] python-build-standalone auto-download (zero Python install for end users)
- [ ] Windows / Linux CI testing

### Planned ML Engines

#### SAM 2 — Interactive Promptable Segmentation

[SAM 2](https://ai.meta.com/sam2/) (Segment Anything Model 2) by Meta is a fundamentally different paradigm from YOLO. Instead of auto-detecting all objects, SAM 2 segments **what you point at** using interactive prompts:

| Prompt Type | Description | Example |
|------------|-------------|---------|
| Point | Click a location to segment the object at that point | "Segment what's at (100, 200)" |
| Box | Draw a bounding box to segment within it | "Segment inside [50,50,300,400]" |
| Mask | Refine an existing mask with additional prompts | "Expand this mask to include the tail" |

**Video Tracking** — SAM 2's killer feature: segment an object in frame 1, and it tracks that object through the entire video with temporal memory. Supports adding/removing prompts across frames for correction.

**Planned API:**
```java
// Image segmentation with prompts
SAM2Model sam = new SAM2Model("sam2_b.pt");
SAM2Result result = sam.predict("photo.jpg",
    Prompt.point(100, 200),          // segment at this point
    Prompt.point(300, 400, Label.NEGATIVE)  // exclude this area
);

// Video object tracking
SAM2VideoTracker tracker = sam.trackVideo("video.mp4");
tracker.addPrompt(0, Prompt.box(50, 50, 300, 400));  // segment in frame 0
SAM2VideoMask mask = tracker.propagate();              // track through all frames
```

**Models:** `sam2_t.pt`, `sam2_s.pt`, `sam2_b.pt`, `sam2_l.pt`, `sam2.1_t/s/b/l.pt`
**Limitations:** Inference only — no training or export. Requires ultralytics with SAM2 module.

#### SAM 3 — Concept-Level Segmentation

[SAM 3](https://ai.meta.com/blog/segment-anything-announcements/) takes segmentation to the concept level. Instead of pointing at objects, describe **what** you want segmented using natural language or visual exemplars:

| Mode | Input | Description |
|------|-------|-------------|
| Text Prompt | "person", "red car", "bus" | Find and segment all instances matching the concept |
| Image Exemplar | Provide bounding box of reference object | Find similar objects in the target image |
| Semantic | No prompt needed | Segment everything and label semantically |

This enables use cases impossible with SAM 2: "find all cars in this image" without manually clicking on each one, or "find objects similar to this one" across different images.

**Planned API:**
```java
SAM3Model sam = new SAM3Model("sam3.pt");

// Text-based concept segmentation
SAM3Result result = sam.predict("street.jpg",
    Prompt.text("person")     // segment all persons
);

// Image exemplar: "find things like this"
SAM3Result result = sam.predict("target.jpg",
    Prompt.exemplar("reference.jpg", BoundingBox.of(10, 20, 200, 300))
);
```

**Model:** `sam3.pt` (3.45 GB, requires HuggingFace access for weights)
**Requirements:** ultralytics 8.3.237+, PyTorch 2.1+
**Limitations:** Inference only. Weights are not auto-downloaded.

#### Other Engines
- [ ] **Transformers (HuggingFace)** — NLP, text generation, translation, embeddings
- [ ] **Whisper** — speech-to-text, automatic speech recognition
- [ ] **Stable Diffusion / FLUX** — image generation, inpainting, controlnet
- [ ] **DeepSeek / LLM** — large language model inference
- [ ] **OpenCV** — traditional computer vision pipelines
- [ ] **MediaPipe** — hand tracking, face detection, gesture recognition

### Infrastructure
- [ ] Async training with real-time streaming callbacks
- [ ] GPU memory management and model warmup APIs
- [ ] Model registry / hub integration (download from URL)
- [ ] Spring Boot starter auto-configuration
- [ ] GraalVM native image support
