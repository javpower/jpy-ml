package io.github.javpower.jpyml.ml.model;

import io.github.javpower.jpyml.core.DependencyManager;
import io.github.javpower.jpyml.core.PythonEngine;
import io.github.javpower.jpyml.core.PythonScriptLoader;
import io.github.javpower.jpyml.exception.InferenceException;
import io.github.javpower.jpyml.exception.ModelException;
import io.github.javpower.jpyml.exception.TrainingException;
import io.github.javpower.jpyml.exception.ValidationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import io.github.javpower.jpyml.ml.export.ExportConfig;
import io.github.javpower.jpyml.ml.export.ExportFormat;
import io.github.javpower.jpyml.ml.export.ExportResult;
import io.github.javpower.jpyml.ml.result.*;
import io.github.javpower.jpyml.ml.training.TrainingCallback;
import io.github.javpower.jpyml.ml.training.TrainingConfig;
import io.github.javpower.jpyml.ml.training.TrainingResult;
import io.github.javpower.jpyml.ml.validation.PerClassMetric;
import io.github.javpower.jpyml.ml.validation.ValidationResult;
import io.github.javpower.jpyml.util.PythonEscape;
import jep.JepException;

import io.github.javpower.jpyml.ml.training.ProgressMonitor;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import javax.imageio.ImageIO;

/**
 * Unified ML model entry point. Supports all Ultralytics models:
 * YOLOv8, YOLO11, RT-DETR, SAM. All tasks: detect, segment, classify, pose, obb.
 * <p>
 * Usage:
 * <pre>
 *   try (Model model = new Model("yolov8n.pt")) {
 *       DetectionResult result = model.predict("photo.jpg");
 *       for (ClassPrediction pred : result.getBoxes()) {
 *           System.out.println(pred);
 *       }
 *   }
 * </pre>
 */
public class Model implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(Model.class);
    private static final AtomicLong idCounter = new AtomicLong(0);

    private final long id;
    private final String varName;
    private final String modelPath;
    private final TaskType taskType;
    private final Map<Integer, String> classNames;
    private final ModelInfo modelInfo;
    private final PythonEngine engine;
    private volatile boolean closed = false;

    // Training state for cancellation and real-time progress
    private static final ExecutorService trainingExecutor =
            Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "jpy-training-executor");
                t.setDaemon(true);
                return t;
            });
    private volatile Path currentCancelFile;
    private volatile Path currentProgressFile;
    private volatile ProgressMonitor currentMonitor;

    public Model(String modelPath) throws ModelException {
        this(modelPath, null);
    }

    public Model(String modelPath, TaskType overrideTask) throws ModelException {
        this.id = idCounter.getAndIncrement();
        this.varName = "_jpy_m" + this.id;
        this.modelPath = modelPath;

        try {
            DependencyManager.ensure("cv");
            engine = PythonEngine.getInstance();
            log.info("Loading model: {} (task={})", modelPath, overrideTask);

            // Load Python helper scripts (idempotent)
            PythonScriptLoader.ensureLoaded(engine, "_jpy_init.py");
            PythonScriptLoader.ensureLoaded(engine, "_jpy_inference.py");

            // Load model in Python
            engine.put(varName + "_path", modelPath);
            if (overrideTask != null) {
                engine.put(varName + "_task_arg", overrideTask.getKey());
                engine.exec(varName + "_info = jpy_load_model(" + varName + "_path, task=" + varName + "_task_arg)");
            } else {
                engine.exec(varName + "_info = jpy_load_model(" + varName + "_path)");
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> info = engine.eval(varName + "_info");

            String detectedVar = (String) info.get("var");
            String detectedTask = (String) info.get("task");

            // Store the actual var name Python assigned
            this.taskType = overrideTask != null ? overrideTask : TaskType.fromString(detectedTask);

            @SuppressWarnings("unchecked")
            Map<Integer, String> names = new LinkedHashMap<>();
            Object rawNames = info.get("names");
            if (rawNames instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<Object, Object> namesMap = (Map<Object, Object>) rawNames;
                namesMap.forEach((k, v) -> names.put(((Number) k).intValue(), String.valueOf(v)));
            }
            this.classNames = Collections.unmodifiableMap(names);

            long params = 0;
            int layers = 0;
            if (info.get("parameters") != null) params = ((Number) info.get("parameters")).longValue();
            if (info.get("layers") != null) layers = ((Number) info.get("layers")).intValue();

            this.modelInfo = new ModelInfo(taskType, classNames, classNames.size(), params, layers);
            log.info("Model loaded: {} classes, {} params, {} layers", classNames.size(), params, layers);

            // Keep a reference to the actual Python var for this model
            engine.exec("_jpy_mv" + this.id + " = _jpy_models['" + escapePythonString(detectedVar) + "']");

        } catch (JepException e) {
            throw new ModelException("Failed to load model: " + modelPath, e);
        }
    }

    // ==================== Preset Factory ====================

    /**
     * Create a model by name, auto-downloading if needed.
     * Uses the ModelHub registry for known models.
     *
     * @param modelName registered model name (e.g., "yolov8n", "yolo11n-seg")
     * @return loaded Model instance
     */
    public static Model preset(String modelName) throws ModelException {
        try {
            Path path = ModelHub.ensure(modelName);
            return new Model(path.toString());
        } catch (IOException e) {
            throw new ModelException("Failed to download model: " + modelName, e);
        }
    }

    /**
     * Create a model by name with explicit task type, auto-downloading if needed.
     */
    public static Model preset(String modelName, TaskType task) throws ModelException {
        try {
            Path path = ModelHub.ensure(modelName);
            return new Model(path.toString(), task);
        } catch (IOException e) {
            throw new ModelException("Failed to download model: " + modelName, e);
        }
    }

    // ==================== Prediction (String path) ====================

    /**
     * Predict on an image file. Returns task-appropriate result.
     */
    public InferenceResult predict(String imagePath) throws InferenceException {
        return predict(imagePath, new ModelConfig());
    }

    public InferenceResult predict(String imagePath, ModelConfig config) throws InferenceException {
        ensureOpen();
        try {
            return predictClean(imagePath, config);
        } catch (Exception e) {
            throw new InferenceException("Prediction failed on: " + imagePath, e);
        }
    }

    // ==================== Prediction (byte[] / BufferedImage) ====================

    /**
     * Predict on raw image bytes (JPEG, PNG, etc.).
     */
    public InferenceResult predict(byte[] imageData) throws InferenceException {
        return predict(imageData, new ModelConfig());
    }

    public InferenceResult predict(byte[] imageData, ModelConfig config) throws InferenceException {
        ensureOpen();
        if (imageData == null || imageData.length == 0) {
            throw new InferenceException("Image data must not be empty");
        }
        try {
            return predictFromBytes(imageData, config);
        } catch (Exception e) {
            throw new InferenceException("Prediction failed on byte[] image", e);
        }
    }

    /**
     * Predict on a BufferedImage.
     */
    public InferenceResult predict(BufferedImage image) throws InferenceException {
        return predict(image, new ModelConfig());
    }

    public InferenceResult predict(BufferedImage image, ModelConfig config) throws InferenceException {
        ensureOpen();
        if (image == null) {
            throw new InferenceException("Image must not be null");
        }
        try {
            byte[] bytes = bufferedImageToBytes(image);
            return predictFromBytes(bytes, config);
        } catch (Exception e) {
            throw new InferenceException("Prediction failed on BufferedImage", e);
        }
    }

    private InferenceResult predictFromBytes(byte[] imageData, ModelConfig config) throws JepException {
        PythonScriptLoader.ensureLoaded(engine, "_jpy_inference.py");
        String mv = "_jpy_mv" + id;
        String rv = "_jpy_pr" + id;
        Map<String, Object> kwargs = config.toPythonKwargs();

        // Pass raw bytes to Python and decode to numpy array
        engine.put(rv + "_src_bytes", imageData);
        engine.exec(rv + "_src = jpy_decode_image(" + rv + "_src_bytes)");

        // Pass kwargs via Jep's safe Java-to-Python conversion
        engine.put(rv + "_kwargs", kwargs);
        engine.exec(rv + "_raw = " + mv + "(" + rv + "_src, **" + rv + "_kwargs)");

        engine.put(rv + "_task", taskType.getKey());
        engine.exec(rv + " = jpy_extract_result(" + rv + "_raw[0], " + rv + "_task)");
        @SuppressWarnings("unchecked")
        Map<String, Object> data = engine.eval(rv);

        return buildResult(data);
    }

    private static byte[] bufferedImageToBytes(BufferedImage image) throws IOException {
        // Convert ARGB to RGB to avoid JPEG alpha issues
        BufferedImage rgb = image;
        if (image.getType() != BufferedImage.TYPE_INT_RGB) {
            rgb = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_RGB);
            Graphics2D g = rgb.createGraphics();
            g.setColor(java.awt.Color.WHITE);
            g.fillRect(0, 0, rgb.getWidth(), rgb.getHeight());
            g.drawImage(image, 0, 0, null);
            g.dispose();
        }
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        if (!ImageIO.write(rgb, "jpg", baos)) {
            throw new IOException("No JPEG image writer available");
        }
        return baos.toByteArray();
    }

    // ==================== Async Prediction ====================

    /**
     * Async prediction on an image file.
     * <p>
     * Because Jep's SharedInterpreter is thread-bound to the creating thread,
     * the prediction executes synchronously on the calling thread and the result
     * is wrapped in a completed CompletableFuture.
     * <p>
     * To offload to a background thread, use
     * {@code CompletableFuture.supplyAsync(() -> model.predict(...), yourExecutor)}
     * with your own executor, but ensure all Model operations are serialized
     * on the same thread that called {@code PythonRuntime.init()}.
     */
    public CompletableFuture<InferenceResult> predictAsync(String imagePath) {
        ensureOpen();
        try {
            return CompletableFuture.completedFuture(predict(imagePath));
        } catch (InferenceException e) {
            return CompletableFuture.failedFuture(e);
        }
    }

    public CompletableFuture<InferenceResult> predictAsync(String imagePath, ModelConfig config) {
        ensureOpen();
        try {
            return CompletableFuture.completedFuture(predict(imagePath, config));
        } catch (InferenceException e) {
            return CompletableFuture.failedFuture(e);
        }
    }

    /**
     * Async prediction on raw image bytes.
     */
    public CompletableFuture<InferenceResult> predictAsync(byte[] imageData) {
        ensureOpen();
        try {
            return CompletableFuture.completedFuture(predict(imageData));
        } catch (InferenceException e) {
            return CompletableFuture.failedFuture(e);
        }
    }

    public CompletableFuture<InferenceResult> predictAsync(byte[] imageData, ModelConfig config) {
        ensureOpen();
        try {
            return CompletableFuture.completedFuture(predict(imageData, config));
        } catch (InferenceException e) {
            return CompletableFuture.failedFuture(e);
        }
    }

    /**
     * Async batch prediction.
     */
    public CompletableFuture<List<InferenceResult>> predictAsync(List<String> imagePaths) {
        ensureOpen();
        try {
            return CompletableFuture.completedFuture(predict(imagePaths));
        } catch (InferenceException e) {
            return CompletableFuture.failedFuture(e);
        }
    }

    public CompletableFuture<List<InferenceResult>> predictAsync(List<String> imagePaths, ModelConfig config) {
        ensureOpen();
        try {
            return CompletableFuture.completedFuture(predict(imagePaths, config));
        } catch (InferenceException e) {
            return CompletableFuture.failedFuture(e);
        }
    }

    /**
     * Async video prediction. Runs video processing synchronously and returns
     * a CompletableFuture that completes when all frames have been processed.
     */
    public CompletableFuture<Void> predictVideoAsync(String videoPath, Consumer<InferenceResult> frameConsumer) {
        ensureOpen();
        try {
            predictVideo(videoPath, frameConsumer);
            return CompletableFuture.completedFuture(null);
        } catch (Exception e) {
            log.warn("Video async prediction failed", e);
            return CompletableFuture.failedFuture(e);
        }
    }

    public CompletableFuture<Void> predictVideoAsync(String videoPath, ModelConfig config, Consumer<InferenceResult> frameConsumer) {
        ensureOpen();
        try {
            predictVideo(videoPath, config, frameConsumer);
            return CompletableFuture.completedFuture(null);
        } catch (Exception e) {
            log.warn("Video async prediction failed", e);
            return CompletableFuture.failedFuture(e);
        }
    }

    private InferenceResult predictClean(String imagePath, ModelConfig config) throws JepException {
        String mv = "_jpy_mv" + id;
        String rv = "_jpy_pr" + id;
        Map<String, Object> kwargs = config.toPythonKwargs();
        engine.put(rv + "_src", imagePath);

        // Pass kwargs via Jep's safe Java-to-Python conversion
        engine.put(rv + "_kwargs", kwargs);
        engine.exec(rv + "_raw = " + mv + "(" + rv + "_src, **" + rv + "_kwargs)");

        // Extract first result
        engine.put(rv + "_task", taskType.getKey());
        engine.exec(rv + " = jpy_extract_result(" + rv + "_raw[0], " + rv + "_task)");
        @SuppressWarnings("unchecked")
        Map<String, Object> data = engine.eval(rv);

        return buildResult(data);
    }

    @SuppressWarnings("unchecked")
    private InferenceResult buildResult(Map<String, Object> data) {
        String sourcePath = (String) data.getOrDefault("path", "");
        List<Number> origShape = (List<Number>) data.get("orig_shape");
        int origH = origShape != null && origShape.size() >= 1 ? origShape.get(0).intValue() : 0;
        int origW = origShape != null && origShape.size() >= 2 ? origShape.get(1).intValue() : 0;

        Map<String, Object> speedData = (Map<String, Object>) data.get("speed");
        InferenceSpeed speed = new InferenceSpeed(
                floatVal(speedData, "preprocess"),
                floatVal(speedData, "inference"),
                floatVal(speedData, "postprocess")
        );

        Map<Integer, String> names = new LinkedHashMap<>();
        Object rawNames = data.get("names");
        if (rawNames instanceof Map) {
            ((Map<Object, Object>) rawNames).forEach((k, v) -> names.put(((Number) k).intValue(), String.valueOf(v)));
        }

        return switch (taskType) {
            case DETECT -> buildDetectionResult(sourcePath, origW, origH, speed, names, data);
            case SEGMENT -> buildSegmentationResult(sourcePath, origW, origH, speed, names, data);
            case CLASSIFY -> buildClassificationResult(sourcePath, origW, origH, speed, names, data);
            case POSE -> buildPoseResult(sourcePath, origW, origH, speed, names, data);
            case OBB -> buildOBBResult(sourcePath, origW, origH, speed, names, data);
        };
    }

    @SuppressWarnings("unchecked")
    private DetectionResult buildDetectionResult(String sp, int w, int h, InferenceSpeed speed,
                                                  Map<Integer, String> names, Map<String, Object> data) {
        List<Map<String, Object>> rawBoxes = (List<Map<String, Object>>) data.get("boxes");
        List<ClassPrediction> boxes = new ArrayList<>();
        if (rawBoxes != null) {
            for (Map<String, Object> b : rawBoxes) {
                BoundingBox box = new BoundingBox(floatVal(b,"x1"), floatVal(b,"y1"), floatVal(b,"x2"), floatVal(b,"y2"));
                boxes.add(new ClassPrediction(box, floatVal(b,"confidence"),
                        intVal(b,"class_id"), names.getOrDefault(intVal(b,"class_id"), "?")));
            }
        }
        return new DetectionResult(sp, w, h, speed, names, boxes);
    }

    @SuppressWarnings("unchecked")
    private SegmentationResult buildSegmentationResult(String sp, int w, int h, InferenceSpeed speed,
                                                        Map<Integer, String> names, Map<String, Object> data) {
        List<Map<String, Object>> rawBoxes = (List<Map<String, Object>>) data.get("boxes");
        List<ClassPrediction> boxes = new ArrayList<>();
        if (rawBoxes != null) {
            for (Map<String, Object> b : rawBoxes) {
                BoundingBox box = new BoundingBox(floatVal(b,"x1"), floatVal(b,"y1"), floatVal(b,"x2"), floatVal(b,"y2"));
                boxes.add(new ClassPrediction(box, floatVal(b,"confidence"),
                        intVal(b,"class_id"), names.getOrDefault(intVal(b,"class_id"), "?")));
            }
        }
        List<Map<String, Object>> rawMasks = (List<Map<String, Object>>) data.get("masks");
        List<Mask> masks = new ArrayList<>();
        if (rawMasks != null) {
            for (Map<String, Object> m : rawMasks) {
                List<List<Number>> poly = (List<List<Number>>) m.get("polygon");
                masks.add(new Mask(ResultParseUtil.parsePolygon(poly)));
            }
        }
        return new SegmentationResult(sp, w, h, speed, names, boxes, masks);
    }

    @SuppressWarnings("unchecked")
    private ClassificationResult buildClassificationResult(String sp, int w, int h, InferenceSpeed speed,
                                                           Map<Integer, String> names, Map<String, Object> data) {
        List<Map<String, Object>> rawPreds = (List<Map<String, Object>>) data.get("classification");
        List<ClassPrediction> preds = new ArrayList<>();
        if (rawPreds != null) {
            for (Map<String, Object> p : rawPreds) {
                int cid = intVal(p, "class_id");
                BoundingBox dummy = new BoundingBox(0, 0, 0, 0);
                preds.add(new ClassPrediction(dummy, floatVal(p, "confidence"), cid, names.getOrDefault(cid, "?")));
            }
        }
        return new ClassificationResult(sp, w, h, speed, names, preds);
    }

    @SuppressWarnings("unchecked")
    private PoseResult buildPoseResult(String sp, int w, int h, InferenceSpeed speed,
                                       Map<Integer, String> names, Map<String, Object> data) {
        List<Map<String, Object>> rawBoxes = (List<Map<String, Object>>) data.get("boxes");
        List<ClassPrediction> boxes = new ArrayList<>();
        if (rawBoxes != null) {
            for (Map<String, Object> b : rawBoxes) {
                BoundingBox box = new BoundingBox(floatVal(b,"x1"), floatVal(b,"y1"), floatVal(b,"x2"), floatVal(b,"y2"));
                boxes.add(new ClassPrediction(box, floatVal(b,"confidence"),
                        intVal(b,"class_id"), names.getOrDefault(intVal(b,"class_id"), "?")));
            }
        }
        List<Map<String, Object>> rawKpts = (List<Map<String, Object>>) data.get("keypoints");
        List<KeypointCollection> kptCollections = new ArrayList<>();
        if (rawKpts != null) {
            for (Map<String, Object> k : rawKpts) {
                List<List<Number>> xy = (List<List<Number>>) k.get("xy");
                List<Number> conf = (List<Number>) k.get("conf");
                List<Keypoint> kpts = new ArrayList<>();
                if (xy != null) {
                    for (int i = 0; i < xy.size(); i++) {
                        float x = xy.get(i).get(0).floatValue();
                        float y = xy.get(i).get(1).floatValue();
                        float c = conf != null && i < conf.size() ? conf.get(i).floatValue() : 1.0f;
                        kpts.add(new Keypoint(x, y, c));
                    }
                }
                kptCollections.add(new KeypointCollection(kpts));
            }
        }
        return new PoseResult(sp, w, h, speed, names, boxes, kptCollections);
    }

    @SuppressWarnings("unchecked")
    private OBBResult buildOBBResult(String sp, int w, int h, InferenceSpeed speed,
                                      Map<Integer, String> names, Map<String, Object> data) {
        List<Map<String, Object>> rawObbs = (List<Map<String, Object>>) data.get("obb");
        List<OBBPrediction> preds = new ArrayList<>();
        if (rawObbs != null) {
            for (Map<String, Object> o : rawObbs) {
                RotatedBoundingBox rbox = new RotatedBoundingBox(
                        floatVal(o,"cx"), floatVal(o,"cy"), floatVal(o,"w"), floatVal(o,"h"), floatVal(o,"angle"));
                int cid = intVal(o, "class_id");
                preds.add(new OBBPrediction(rbox, floatVal(o,"confidence"), cid, names.getOrDefault(cid, "?")));
            }
        }
        return new OBBResult(sp, w, h, speed, names, preds);
    }

    // ==================== Batch Prediction ====================

    public List<InferenceResult> predict(List<String> imagePaths) throws InferenceException {
        return predict(imagePaths, new ModelConfig());
    }

    public List<InferenceResult> predict(List<String> imagePaths, ModelConfig config) throws InferenceException {
        ensureOpen();
        if (imagePaths == null || imagePaths.isEmpty()) {
            throw new InferenceException("Image paths must not be empty");
        }
        try {
            String mv = "_jpy_mv" + id;
            String rv = "_jpy_pr" + id;
            Map<String, Object> kwargs = config.toPythonKwargs();

            // Put image list into Python (Jep converts Java List to Python list)
            engine.put(rv + "_batch_src", imagePaths);

            // Pass kwargs via Jep's safe Java-to-Python conversion
            engine.put(rv + "_batch_kwargs", kwargs);

            // Run batch predict in Python using native batch inference
            engine.put(rv + "_task", taskType.getKey());
            engine.exec(
                    rv + "_batch = jpy_batch_predict(" + mv + ", " + rv + "_batch_src, " + rv + "_task, " + rv + "_batch_kwargs)"
            );

            @SuppressWarnings("unchecked")
            List<Map<String, Object>> batchData = engine.eval(rv + "_batch");

            List<InferenceResult> results = new ArrayList<>();
            if (batchData != null) {
                for (Map<String, Object> data : batchData) {
                    results.add(buildResult(data));
                }
            }
            return results;
        } catch (Exception e) {
            throw new InferenceException("Batch prediction failed", e);
        }
    }

    // ==================== Batch Prediction (byte[] / BufferedImage) ====================

    /**
     * Batch prediction on multiple raw image byte arrays.
     * Images are decoded and processed together for GPU-batched inference.
     *
     * @param imageDataList list of raw image bytes (JPEG, PNG, etc.)
     * @return list of inference results, one per image
     * @throws InferenceException if prediction fails
     */
    public List<InferenceResult> predictBytesBatch(List<byte[]> imageDataList) throws InferenceException {
        return predictBytesBatch(imageDataList, new ModelConfig());
    }

    public List<InferenceResult> predictBytesBatch(List<byte[]> imageDataList, ModelConfig config) throws InferenceException {
        ensureOpen();
        if (imageDataList == null || imageDataList.isEmpty()) {
            throw new InferenceException("Image data list must not be empty");
        }
        try {
            String mv = "_jpy_mv" + id;
            String rv = "_jpy_pr" + id;
            Map<String, Object> kwargs = config.toPythonKwargs();

            engine.put(rv + "_batch_bytes", imageDataList);
            engine.put(rv + "_batch_kwargs", kwargs);
            engine.put(rv + "_task", taskType.getKey());

            engine.exec(
                    rv + "_batch_imgs = [jpy_decode_image(b) for b in " + rv + "_batch_bytes]\n" +
                    rv + "_batch = jpy_batch_predict(" + mv + ", " + rv + "_batch_imgs, " + rv + "_task, " + rv + "_batch_kwargs)"
            );

            @SuppressWarnings("unchecked")
            List<Map<String, Object>> batchData = engine.eval(rv + "_batch");

            List<InferenceResult> results = new ArrayList<>();
            if (batchData != null) {
                for (Map<String, Object> data : batchData) {
                    results.add(buildResult(data));
                }
            }
            return results;
        } catch (Exception e) {
            throw new InferenceException("Batch prediction (bytes) failed", e);
        }
    }

    /**
     * Batch prediction on multiple BufferedImages.
     * Images are converted to byte arrays and processed together for GPU-batched inference.
     *
     * @param images list of BufferedImage objects
     * @return list of inference results, one per image
     * @throws InferenceException if prediction fails
     */
    public List<InferenceResult> predictImagesBatch(List<BufferedImage> images) throws InferenceException {
        return predictImagesBatch(images, new ModelConfig());
    }

    public List<InferenceResult> predictImagesBatch(List<BufferedImage> images, ModelConfig config) throws InferenceException {
        ensureOpen();
        if (images == null || images.isEmpty()) {
            throw new InferenceException("Image list must not be empty");
        }
        try {
            List<byte[]> byteList = new ArrayList<>();
            for (BufferedImage img : images) {
                byteList.add(bufferedImageToBytes(img));
            }
            return predictBytesBatch(byteList, config);
        } catch (IOException e) {
            throw new InferenceException("Failed to convert BufferedImages", e);
        }
    }

    // ==================== Raw Prediction (Zero-Copy) ====================

    /**
     * Zero-copy prediction for detection tasks. Returns RawDetectionResult
     * with direct buffer access for high-performance scenarios.
     * <p>
     * Only supports DETECT task type. For other task types, use predict().
     *
     * @param imagePath path to the image file
     * @return RawDetectionResult with direct buffer access
     * @throws InferenceException if prediction fails or task type is not DETECT
     */
    public RawDetectionResult predictRaw(String imagePath) throws InferenceException {
        return predictRaw(imagePath, new ModelConfig());
    }

    /**
     * Zero-copy prediction for detection tasks with custom config.
     *
     * @param imagePath path to the image file
     * @param config    inference configuration
     * @return RawDetectionResult with direct buffer access
     * @throws InferenceException if prediction fails or task type is not DETECT
     */
    public RawDetectionResult predictRaw(String imagePath, ModelConfig config) throws InferenceException {
        ensureOpen();
        if (taskType != TaskType.DETECT) {
            throw new InferenceException("predictRaw() only supports DETECT task, current task: " + taskType);
        }
        try {
            return predictRawClean(imagePath, config);
        } catch (Exception e) {
            throw new InferenceException("Raw prediction failed on: " + imagePath, e);
        }
    }

    private RawDetectionResult predictRawClean(String imagePath, ModelConfig config) throws JepException {
        String mv = "_jpy_mv" + id;
        String rv = "_jpy_prr" + id;
        Map<String, Object> kwargs = config.toPythonKwargs();

        // Execute prediction
        engine.put(rv + "_src", imagePath);
        engine.put(rv + "_kwargs", kwargs);
        engine.exec(rv + "_raw = " + mv + "(" + rv + "_src, **" + rv + "_kwargs)");

        // Get result count first to size buffers
        engine.exec(rv + "_count = len(" + rv + "_raw[0].boxes) if " + rv + "_raw[0].boxes is not None else 0");
        int boxCount = engine.eval(rv + "_count");

        // Acquire buffers from pool
        TensorBufferPool pool = TensorBufferPool.getInstance();
        FloatBuffer xyxyBuf = pool.acquireFloatBuffer(Math.max(boxCount, 1) * 4);
        FloatBuffer confBuf = pool.acquireFloatBuffer(Math.max(boxCount, 1));
        IntBuffer clsBuf = pool.acquireIntBuffer(Math.max(boxCount, 1));

        try {
            // Create DirectNDArray wrappers
            jep.DirectNDArray<FloatBuffer> xyxyNd = new jep.DirectNDArray<>(xyxyBuf, boxCount * 4);
            jep.DirectNDArray<FloatBuffer> confNd = new jep.DirectNDArray<>(confBuf, boxCount);
            jep.DirectNDArray<IntBuffer> clsNd = new jep.DirectNDArray<>(clsBuf, boxCount);

            // Pass buffers to Python
            engine.put(rv + "_xyxy_buf", xyxyNd);
            engine.put(rv + "_conf_buf", confNd);
            engine.put(rv + "_cls_buf", clsNd);

            // Extract result with zero-copy
            engine.put(rv + "_task", taskType.getKey());
            engine.exec(rv + " = jpy_extract_result_ndarray(" + rv + "_raw[0], " + rv + "_task, " +
                    "{'boxes_xyxy': " + rv + "_xyxy_buf, 'boxes_conf': " + rv + "_conf_buf, 'boxes_cls': " + rv + "_cls_buf})");

            @SuppressWarnings("unchecked")
            Map<String, Object> data = engine.eval(rv);

            // Parse common fields
            String sourcePath = (String) data.getOrDefault("path", "");
            List<Number> origShape = (List<Number>) data.get("orig_shape");
            int origH = origShape != null && origShape.size() >= 1 ? origShape.get(0).intValue() : 0;
            int origW = origShape != null && origShape.size() >= 2 ? origShape.get(1).intValue() : 0;

            Map<String, Object> speedData = (Map<String, Object>) data.get("speed");
            InferenceSpeed speed = new InferenceSpeed(
                    floatVal(speedData, "preprocess"),
                    floatVal(speedData, "inference"),
                    floatVal(speedData, "postprocess")
            );

            Map<Integer, String> names = new LinkedHashMap<>();
            Object rawNames = data.get("names");
            if (rawNames instanceof Map) {
                ((Map<Object, Object>) rawNames).forEach((k, v) -> names.put(((Number) k).intValue(), String.valueOf(v)));
            }

            // Reset buffer limits for reading
            xyxyBuf.limit(boxCount * 4);
            confBuf.limit(boxCount);
            clsBuf.limit(boxCount);

            // Save references before nulling for finally block
            FloatBuffer xyxyRef = xyxyBuf;
            FloatBuffer confRef = confBuf;
            IntBuffer clsRef = clsBuf;

            // Ownership transferred to result — null out so finally won't release
            xyxyBuf = null;
            confBuf = null;
            clsBuf = null;

            return new RawDetectionResult(sourcePath, origW, origH, speed, names,
                    xyxyRef, confRef, clsRef, boxCount);
        } finally {
            // Release buffers if ownership was not transferred (exception path)
            pool.release(xyxyBuf);
            pool.release(confBuf);
            pool.release(clsBuf);
        }
    }

    // ==================== GPU Memory Management ====================

    /**
     * Warmup the model by running a dummy inference. This triggers CUDA kernel
     * compilation and reduces first-inference latency.
     */
    public void warmup() throws InferenceException {
        ensureOpen();
        try {
            String mv = "_jpy_mv" + id;
            engine.exec(mv + ".warmup()");
        } catch (JepException e) {
            throw new InferenceException("Model warmup failed", e);
        }
    }

    /**
     * Move the model to CPU to free GPU memory.
     * After calling this, inference will run on CPU until reload() is called.
     */
    public void unload() throws InferenceException {
        ensureOpen();
        try {
            String mv = "_jpy_mv" + id;
            engine.exec(mv + ".cpu()");
        } catch (JepException e) {
            throw new InferenceException("Failed to unload model to CPU", e);
        }
    }

    /**
     * Move the model to the specified device.
     *
     * @param device target device (e.g., "cpu", "cuda:0", "mps")
     */
    public void reload(String device) throws InferenceException {
        ensureOpen();
        try {
            String mv = "_jpy_mv" + id;
            engine.put(mv + "_device", device);
            engine.exec(mv + ".to(" + mv + "_device)");
        } catch (JepException e) {
            throw new InferenceException("Failed to reload model to device: " + device, e);
        }
    }

    // ==================== Video (chunk-based streaming) ====================

    public void predictVideo(String videoPath, Consumer<InferenceResult> frameConsumer) {
        predictVideo(videoPath, new ModelConfig(), frameConsumer);
    }

    public void predictVideo(String videoPath, ModelConfig config, Consumer<InferenceResult> frameConsumer) {
        ensureOpen();
        try {
            PythonScriptLoader.ensureLoaded(engine, "_jpy_streaming.py");
            String mv = "_jpy_mv" + id;
            String sv = "_jpy_st" + id;
            Map<String, Object> kwargs = config.toPythonKwargs();

            engine.put(sv + "_src", videoPath);
            engine.put(sv + "_kwargs", kwargs);
            engine.exec("jpy_stream_start(" + mv + ", " + sv + "_src, " + sv + "_kwargs)");

            // Read frames in chunks
            engine.put(sv + "_task", taskType.getKey());
            while (true) {
                engine.exec(sv + "_chunk = jpy_stream_next(" + sv + "_task, chunk_size=10)");
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> chunk = engine.eval(sv + "_chunk");
                if (chunk == null || chunk.isEmpty()) break;
                for (Map<String, Object> frameData : chunk) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> resultMap = (Map<String, Object>) frameData.get("result");
                    frameConsumer.accept(buildResult(resultMap));
                }
            }

            engine.exec("jpy_stream_cleanup()");
        } catch (JepException e) {
            throw new InferenceException("Video prediction failed: " + videoPath, e);
        }
    }

    // ==================== Stream (webcam / RTSP / real-time) ====================

    private final java.util.concurrent.atomic.AtomicBoolean streaming = new java.util.concurrent.atomic.AtomicBoolean(false);

    public void predictStream(int cameraIndex, Consumer<StreamFrame> frameConsumer) {
        predictStream(String.valueOf(cameraIndex), new ModelConfig(), true, frameConsumer);
    }

    public void predictStream(String source, Consumer<StreamFrame> frameConsumer) {
        predictStream(source, new ModelConfig(), true, frameConsumer);
    }

    public void predictStream(int cameraIndex, ModelConfig config, Consumer<StreamFrame> frameConsumer) {
        predictStream(String.valueOf(cameraIndex), config, true, frameConsumer);
    }

    public void predictStream(String source, ModelConfig config, Consumer<StreamFrame> frameConsumer) {
        predictStream(source, config, true, frameConsumer);
    }

    public void predictStream(int cameraIndex, ModelConfig config, boolean annotate, Consumer<StreamFrame> frameConsumer) {
        predictStream(String.valueOf(cameraIndex), config, annotate, frameConsumer);
    }

    public void predictStream(String source, ModelConfig config, boolean annotate, Consumer<StreamFrame> frameConsumer) {
        ensureOpen();
        if (!streaming.compareAndSet(false, true)) {
            throw new IllegalStateException("Stream already active on this model instance");
        }
        try {
            PythonScriptLoader.ensureLoaded(engine, "_jpy_streaming.py");
            String mv = "_jpy_mv" + id;
            String sv = "_jpy_st" + id;
            Map<String, Object> kwargs = config.toPythonKwargs();

            engine.put(sv + "_src", source);
            engine.put(sv + "_kwargs", kwargs);
            engine.exec("jpy_stream_start(" + mv + ", " + sv + "_src, " + sv + "_kwargs)");

            // Read frames one at a time for low latency
            engine.put(sv + "_task", taskType.getKey());
            while (streaming.get()) {
                engine.exec(sv + "_chunk = jpy_stream_next(" + sv + "_task" +
                        ", chunk_size=1, annotate=" + (annotate ? "True" : "False") + ")");
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> chunk = engine.eval(sv + "_chunk");
                if (chunk == null || chunk.isEmpty()) break;
                for (Map<String, Object> frameData : chunk) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> resultMap = (Map<String, Object>) frameData.get("result");
                    InferenceResult result = buildResult(resultMap);

                    byte[] imageBytes = null;
                    if (annotate && frameData.get("image") != null) {
                        imageBytes = (byte[]) frameData.get("image");
                    }
                    int frameIndex = frameData.get("frame_index") instanceof Number ?
                            ((Number) frameData.get("frame_index")).intValue() : 0;

                    frameConsumer.accept(new StreamFrame(result, imageBytes, frameIndex));
                }
            }

            engine.exec("jpy_stream_cleanup()");
        } catch (JepException e) {
            throw new InferenceException("Stream prediction failed: " + source, e);
        } finally {
            streaming.set(false);
        }
    }

    public void stopStream() {
        streaming.set(false);
    }

    /**
     * Shut down the training executor. Call this when the application is done
     * with all training operations. After calling this, trainAsync() will throw
     * RejectedExecutionException. Typically called from a shutdown hook.
     */
    public static void shutdownTraining() {
        trainingExecutor.shutdown();
    }

    public boolean isStreaming() {
        return streaming.get();
    }

    // ==================== Training ====================

    public TrainingResult train(TrainingConfig config) throws TrainingException {
        return trainInternal(config, null);
    }

    public TrainingResult train(TrainingConfig config, TrainingCallback callback) throws TrainingException {
        return trainInternal(config, callback);
    }

    /**
     * Train the model asynchronously with real-time callback.
     * Returns a CompletableFuture that completes when training finishes.
     * The callback fires on a background monitor thread as each epoch completes.
     * <p>
     * No other Model operations (predict, validate, etc.) can run until
     * training completes or is cancelled via stopTraining().
     */
    public CompletableFuture<TrainingResult> trainAsync(TrainingConfig config, TrainingCallback callback) {
        ensureOpen();
        if (config.getDataConfig() == null) {
            throw new TrainingException("dataConfig is required");
        }
        return CompletableFuture.supplyAsync(() -> {
            try {
                return trainInternal(config, callback);
            } catch (TrainingException e) {
                throw new CompletionException(e);
            }
        }, trainingExecutor);
    }

    /**
     * Request cancellation of an ongoing training session.
     * Training will stop after the current epoch completes.
     */
    public void stopTraining() {
        Path cancel = currentCancelFile;
        if (cancel != null) {
            try {
                Files.writeString(cancel, "cancel");
                log.info("Training cancellation requested");
            } catch (IOException e) {
                log.warn("Failed to write cancel signal: {}", e.getMessage());
            }
        }
        ProgressMonitor monitor = currentMonitor;
        if (monitor != null) {
            monitor.stop();
        }
    }

    private TrainingResult trainInternal(TrainingConfig config, TrainingCallback callback) throws TrainingException {
        ensureOpen();
        if (config.getDataConfig() == null) {
            throw new TrainingException("dataConfig is required");
        }
        try {
            PythonScriptLoader.ensureLoaded(engine, "_jpy_progress.py");
            PythonScriptLoader.ensureLoaded(engine, "_jpy_training.py");
            Map<String, Object> kwargs = config.toPythonDict();

            ProgressMonitor monitor = null;
            Path progressFile = null;
            Path cancelFile = null;

            if (callback != null) {
                progressFile = Files.createTempFile("jpy-train-progress-", ".jsonl");
                cancelFile = Path.of(progressFile.toString() + ".cancel");
                Files.deleteIfExists(cancelFile);
                progressFile.toFile().deleteOnExit();

                currentProgressFile = progressFile;
                currentCancelFile = cancelFile;

                monitor = new ProgressMonitor(progressFile, callback);
                currentMonitor = monitor;
                monitor.start();
            }

            try {
                engine.put("_jpy_train_model_path", modelPath);
                engine.put("_jpy_train_kwargs", kwargs);
                String progressPath = progressFile != null ? progressFile.toString() : "";
                String cancelPath = cancelFile != null ? cancelFile.toString() : "";
                engine.put("_jpy_train_progress", progressPath);
                engine.put("_jpy_train_cancel", cancelPath);

                engine.exec(
                        "_jpy_train_result = jpy_train(" +
                        "_jpy_train_model_path, _jpy_train_kwargs, " +
                        "enable_logging=True, " +
                        "progress_file=_jpy_train_progress or None, " +
                        "cancel_file=_jpy_train_cancel or None)"
                );

                @SuppressWarnings("unchecked")
                Map<String, Object> result = engine.eval("_jpy_train_result");

                @SuppressWarnings("unchecked")
                List<Map<String, Object>> epochLog =
                        (List<Map<String, Object>>) result.getOrDefault("epoch_log", List.of());

                return new TrainingResult(
                        (String) result.getOrDefault("best_model", ""),
                        (String) result.getOrDefault("last_model", ""),
                        (String) result.getOrDefault("save_dir", ""),
                        ((Number) result.getOrDefault("epochs_completed", 0)).intValue(),
                        ((Number) result.getOrDefault("best_fitness", 0.0)).floatValue(),
                        epochLog
                );
            } finally {
                if (monitor != null) {
                    String error = monitor.awaitCompletion();
                    if (error != null && !error.isEmpty()) {
                        callback.onComplete(error);
                    } else {
                        callback.onComplete(null);
                    }
                    currentMonitor = null;
                }
                cleanupTrainingTempFiles(progressFile, cancelFile);
            }
        } catch (JepException e) {
            throw new TrainingException("Training failed", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new TrainingException("Training interrupted", e);
        } catch (IOException e) {
            throw new TrainingException("Failed to create progress file", e);
        }
    }

    private void cleanupTrainingTempFiles(Path progressFile, Path cancelFile) {
        try {
            if (progressFile != null) Files.deleteIfExists(progressFile);
        } catch (IOException e) {
            log.debug("Failed to delete progress file: {}", e.getMessage());
        }
        try {
            if (cancelFile != null) Files.deleteIfExists(cancelFile);
        } catch (IOException e) {
            log.debug("Failed to delete cancel file: {}", e.getMessage());
        }
        currentProgressFile = null;
        currentCancelFile = null;
    }

    // ==================== Validation ====================

    public ValidationResult validate() throws ValidationException {
        return validate(null);
    }

    public ValidationResult validate(String dataConfig) throws ValidationException {
        ensureOpen();
        try {
            PythonScriptLoader.ensureLoaded(engine, "_jpy_validation.py");
            engine.put("_jpy_val_model_path", modelPath);
            if (dataConfig != null) {
                engine.put("_jpy_val_data", dataConfig);
                engine.exec("_jpy_val_result = jpy_validate(_jpy_val_model_path, data=_jpy_val_data)");
            } else {
                engine.exec("_jpy_val_result = jpy_validate(_jpy_val_model_path)");
            }

            @SuppressWarnings("unchecked")
            Map<String, Object> result = engine.eval("_jpy_val_result");

            // Parse per-class metrics
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> rawPerClass = (List<Map<String, Object>>) result.getOrDefault("per_class", List.of());
            List<PerClassMetric> perClass = new ArrayList<>();
            if (rawPerClass != null) {
                for (Map<String, Object> m : rawPerClass) {
                    perClass.add(new PerClassMetric(
                            intVal(m, "class_id"),
                            (String) m.getOrDefault("class_name", "?"),
                            floatVal(m, "map50_95")
                    ));
                }
            }

            @SuppressWarnings("unchecked")
            Map<String, Object> speedData = (Map<String, Object>) result.get("speed");

            return new ValidationResult(
                    floatVal(result, "map50"),
                    floatVal(result, "map50_95"),
                    floatVal(result, "precision"),
                    floatVal(result, "recall"),
                    speedData,
                    perClass
            );
        } catch (JepException e) {
            throw new ValidationException("Validation failed", e);
        }
    }

    // ==================== Export ====================

    public ExportResult export(ExportFormat format) throws InferenceException {
        return export(new ExportConfig().format(format));
    }

    public ExportResult export(ExportConfig config) throws InferenceException {
        ensureOpen();
        try {
            PythonScriptLoader.ensureLoaded(engine, "_jpy_export.py");
            engine.put("_jpy_exp_model_path", modelPath);

            // Build kwargs from ExportConfig
            Map<String, Object> kwargs = new LinkedHashMap<>();
            kwargs.put("half", config.isHalf());
            kwargs.put("dynamic", config.isDynamic());
            kwargs.put("simplify", config.isSimplify());
            kwargs.put("opset", config.getOpset());
            kwargs.put("imgsz", config.getImgsz());

            engine.put("_jpy_exp_kwargs", kwargs);
            engine.put("_jpy_exp_format", config.getFormat().getKey());
            engine.exec("_jpy_exp_result = jpy_export(_jpy_exp_model_path, _jpy_exp_format, **_jpy_exp_kwargs)");

            @SuppressWarnings("unchecked")
            Map<String, Object> result = engine.eval("_jpy_exp_result");

            return new ExportResult(
                    (String) result.get("path"),
                    config.getFormat(),
                    ((Number) result.getOrDefault("size", 0)).longValue()
            );
        } catch (JepException e) {
            throw new InferenceException("Export failed", e);
        }
    }

    // ==================== Model Info ====================

    public TaskType getTaskType() { return taskType; }
    public ModelInfo getModelInfo() { return modelInfo; }
    public Map<Integer, String> getClassNames() { return classNames; }
    public int getNumClasses() { return classNames.size(); }
    public String getModelPath() { return modelPath; }

    // ==================== Lifecycle ====================

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("Model is closed");
    }

    @Override
    public void close() {
        if (!closed) {
            stopStream();
            int waitMs = 0;
            while (streaming.get() && waitMs < 2000) {
                try { Thread.sleep(50); } catch (InterruptedException e) { break; }
                waitMs += 50;
            }
            closed = true;
            log.info("Closing model: {}", modelPath);
            try {
                engine.exec("_jpy_cleanup('_jpy_m" + id + "')");
            } catch (Exception e) {
                log.debug("Error cleaning model reference: {}", e.getMessage());
            }
            try {
                engine.exec("_jpy_mv" + id + " = None");
                engine.exec("_jpy_pr" + id + " = None");
            } catch (Exception e) {
                log.debug("Error cleaning model variables: {}", e.getMessage());
            }
        }
    }

    public boolean isClosed() { return closed; }

    // ==================== Helpers ====================

    private static float floatVal(Map<String, Object> map, String key) {
        Object v = map.get(key);
        return v instanceof Number ? ((Number) v).floatValue() : 0f;
    }

    private static int intVal(Map<String, Object> map, String key) {
        Object v = map.get(key);
        return v instanceof Number ? ((Number) v).intValue() : 0;
    }

    private static String escapePythonString(String s) {
        return PythonEscape.escape(s);
    }
}
