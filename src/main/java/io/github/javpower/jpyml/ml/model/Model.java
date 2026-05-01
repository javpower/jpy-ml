package io.github.javpower.jpyml.ml.model;

import io.github.javpower.jpyml.core.PythonEngine;
import io.github.javpower.jpyml.core.PythonScriptLoader;
import io.github.javpower.jpyml.exception.InferenceException;
import io.github.javpower.jpyml.exception.ModelException;
import io.github.javpower.jpyml.exception.TrainingException;
import io.github.javpower.jpyml.ml.export.ExportConfig;
import io.github.javpower.jpyml.ml.export.ExportFormat;
import io.github.javpower.jpyml.ml.export.ExportResult;
import io.github.javpower.jpyml.ml.result.*;
import io.github.javpower.jpyml.ml.training.TrainingCallback;
import io.github.javpower.jpyml.ml.training.TrainingConfig;
import io.github.javpower.jpyml.ml.training.TrainingResult;
import io.github.javpower.jpyml.ml.validation.PerClassMetric;
import io.github.javpower.jpyml.ml.validation.ValidationResult;
import jep.JepException;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

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

    private static final AtomicLong idCounter = new AtomicLong(0);

    private final long id;
    private final String varName;
    private final String modelPath;
    private final TaskType taskType;
    private final Map<Integer, String> classNames;
    private final ModelInfo modelInfo;
    private final PythonEngine engine;
    private boolean closed = false;

    public Model(String modelPath) throws ModelException {
        this(modelPath, null);
    }

    public Model(String modelPath, TaskType overrideTask) throws ModelException {
        this.id = idCounter.getAndIncrement();
        this.varName = "_jpy_m" + this.id;
        this.modelPath = modelPath;

        try {
            engine = PythonEngine.getInstance();

            // Load Python helper scripts (idempotent)
            PythonScriptLoader.ensureLoaded(engine, "_jpy_init.py");
            PythonScriptLoader.ensureLoaded(engine, "_jpy_inference.py");

            // Load model in Python
            engine.put(varName + "_path", modelPath);
            String taskArg = overrideTask != null ? ", task='" + overrideTask.getKey() + "'" : "";
            engine.exec(varName + "_info = jpy_load_model(" + varName + "_path" + taskArg + ")");
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

            // Keep a reference to the actual Python var for this model
            engine.exec("_jpy_mv" + this.id + " = _jpy_models['" + detectedVar + "']");

        } catch (JepException e) {
            throw new ModelException("Failed to load model: " + modelPath, e);
        }
    }

    // ==================== Prediction ====================

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

    private InferenceResult predictClean(String imagePath, ModelConfig config) throws JepException {
        String mv = "_jpy_mv" + id;
        String rv = "_jpy_pr" + id;
        Map<String, Object> kwargs = config.toPythonKwargs();
        engine.put(rv + "_src", imagePath);

        // Build the predict call
        StringBuilder call = new StringBuilder();
        call.append(rv).append("_raw = ").append(mv).append("(").append(rv).append("_src");
        for (Map.Entry<String, Object> e : kwargs.entrySet()) {
            call.append(", ").append(e.getKey()).append("=");
            Object v = e.getValue();
            if (v instanceof String) call.append("'").append(v).append("'");
            else if (v instanceof Boolean) call.append((Boolean) v ? "True" : "False");
            else if (v instanceof List) {
                call.append(v.toString().replace('[', '(').replace(']', ')'));
            }
            else call.append(v);
        }
        call.append(")");

        engine.exec(call.toString());

        // Extract first result
        engine.exec(rv + " = jpy_extract_result(" + rv + "_raw[0], '" + taskType.getKey() + "')");
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
                float[][] polygon = new float[poly != null ? poly.size() : 0][2];
                if (poly != null) {
                    for (int i = 0; i < poly.size(); i++) {
                        polygon[i][0] = poly.get(i).get(0).floatValue();
                        polygon[i][1] = poly.get(i).get(1).floatValue();
                    }
                }
                masks.add(new Mask(polygon));
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

    // ==================== Video ====================

    public void predictVideo(String videoPath, Consumer<InferenceResult> frameConsumer) {
        predictVideo(videoPath, new ModelConfig(), frameConsumer);
    }

    public void predictVideo(String videoPath, ModelConfig config, Consumer<InferenceResult> frameConsumer) {
        ensureOpen();
        try {
            Map<String, Object> kwargs = config.toPythonKwargs();
            kwargs.put("stream", true);
            String mv = "_jpy_mv" + id;
            engine.put("_jpy_vid_src", videoPath);

            StringBuilder call = new StringBuilder();
            call.append("_jpy_vid_results = ").append(mv).append("(").append("_jpy_vid_src");
            for (Map.Entry<String, Object> e : kwargs.entrySet()) {
                call.append(", ").append(e.getKey()).append("=");
                Object v = e.getValue();
                if (v instanceof Boolean) call.append((Boolean) v ? "True" : "False");
                else if (v instanceof String) call.append("'").append(v).append("'");
                else call.append(v);
            }
            call.append(")");

            engine.exec(call.toString());

            // Iterate stream
            engine.exec(
                    "_jpy_vid_frames = []\n" +
                    "for _jpy_vr in _jpy_vid_results:\n" +
                    "    _jpy_vid_frames.append(jpy_extract_result(_jpy_vr, '" + taskType.getKey() + "'))\n"
            );

            @SuppressWarnings("unchecked")
            List<Map<String, Object>> frames = engine.eval("_jpy_vid_frames");
            if (frames != null) {
                for (Map<String, Object> frame : frames) {
                    frameConsumer.accept(buildResult(frame));
                }
            }
        } catch (JepException e) {
            throw new InferenceException("Video prediction failed: " + videoPath, e);
        }
    }

    // ==================== Training ====================

    public TrainingResult train(TrainingConfig config) throws TrainingException {
        return train(config, null);
    }

    public TrainingResult train(TrainingConfig config, TrainingCallback callback) throws TrainingException {
        ensureOpen();
        if (config.getDataConfig() == null) {
            throw new TrainingException("dataConfig is required");
        }
        try {
            PythonScriptLoader.ensureLoaded(engine, "_jpy_training.py");
            Map<String, Object> kwargs = config.toPythonDict();

            engine.put("_jpy_train_model_path", modelPath);
            String dictStr = mapToPythonDict(kwargs);
            engine.exec("_jpy_train_kwargs = " + dictStr);
            boolean enableLogging = callback != null;
            engine.exec("_jpy_train_result = jpy_train(_jpy_train_model_path, _jpy_train_kwargs, enable_logging=" + (enableLogging ? "True" : "False") + ")");

            @SuppressWarnings("unchecked")
            Map<String, Object> result = engine.eval("_jpy_train_result");

            // Extract epoch log
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> epochLog = (List<Map<String, Object>>) result.getOrDefault("epoch_log", List.of());

            // Replay epoch log to callback
            if (callback != null && epochLog != null) {
                for (Map<String, Object> entry : epochLog) {
                    int epoch = ((Number) entry.getOrDefault("epoch", 0)).intValue();
                    callback.onEpoch(epoch, entry.toString());
                }
            }

            return new TrainingResult(
                    (String) result.getOrDefault("best_model", ""),
                    (String) result.getOrDefault("last_model", ""),
                    (String) result.getOrDefault("save_dir", ""),
                    ((Number) result.getOrDefault("epochs_completed", 0)).intValue(),
                    ((Number) result.getOrDefault("best_fitness", 0.0)).floatValue(),
                    epochLog
            );
        } catch (JepException e) {
            throw new TrainingException("Training failed", e);
        }
    }

    // ==================== Validation ====================

    public ValidationResult validate() throws InferenceException {
        return validate(null);
    }

    public ValidationResult validate(String dataConfig) throws InferenceException {
        ensureOpen();
        try {
            PythonScriptLoader.ensureLoaded(engine, "_jpy_validation.py");
            engine.put("_jpy_val_model_path", modelPath);
            if (dataConfig != null) {
                engine.exec("_jpy_val_result = jpy_validate(_jpy_val_model_path, data='" + dataConfig + "')");
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
            throw new InferenceException("Validation failed", e);
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

            String dictStr = mapToPythonDict(kwargs);
            engine.exec("_jpy_exp_kwargs = " + dictStr);
            engine.exec("_jpy_exp_result = jpy_export(_jpy_exp_model_path, '" + config.getFormat().getKey() + "', **_jpy_exp_kwargs)");

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
            closed = true;
            try {
                engine.exec("_jpy_mv" + id + " = None");
                engine.exec("_jpy_pr" + id + " = None");
            } catch (Exception ignored) {
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

    private static String dictToPythonKwargs(Map<String, Object> kwargs) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Object> e : kwargs.entrySet()) {
            if (sb.length() > 0) sb.append(", ");
            sb.append(e.getKey()).append("=");
            Object v = e.getValue();
            if (v instanceof String) sb.append("'").append(v).append("'");
            else if (v instanceof Boolean) sb.append((Boolean) v ? "True" : "False");
            else if (v instanceof List) {
                sb.append(v.toString().replace('[', '(').replace(']', ')'));
            }
            else sb.append(v);
        }
        return sb.toString();
    }

    private static String mapToPythonDict(Map<String, Object> map) {
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, Object> e : map.entrySet()) {
            if (!first) sb.append(", ");
            first = false;
            sb.append("'").append(e.getKey()).append("': ");
            Object v = e.getValue();
            if (v instanceof String) sb.append("'").append(v).append("'");
            else if (v instanceof Boolean) sb.append((Boolean) v ? "True" : "False");
            else if (v instanceof Number) sb.append(v);
            else sb.append("'").append(v).append("'");
        }
        sb.append("}");
        return sb.toString();
    }
}
