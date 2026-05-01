package io.github.javpower.jpyml;

import io.github.javpower.jpyml.core.PythonRuntime;
import io.github.javpower.jpyml.ml.export.ExportFormat;
import io.github.javpower.jpyml.ml.export.ExportResult;
import io.github.javpower.jpyml.ml.model.Device;
import io.github.javpower.jpyml.ml.model.Model;
import io.github.javpower.jpyml.ml.model.ModelConfig;
import io.github.javpower.jpyml.ml.model.TaskType;
import io.github.javpower.jpyml.ml.result.*;
import io.github.javpower.jpyml.ml.training.EpochMetric;
import io.github.javpower.jpyml.ml.training.TrainingConfig;
import io.github.javpower.jpyml.ml.training.TrainingResult;
import io.github.javpower.jpyml.ml.validation.PerClassMetric;
import io.github.javpower.jpyml.ml.validation.ValidationResult;
import org.junit.jupiter.api.*;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for the unified Model API.
 * Tests all 5 YOLO task types: detect, segment, classify, pose, obb.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class ModelIntegrationTest {

    private static final String MODEL_PATH = "yolov8n.pt";
    private static final String TEST_IMAGE = "https://ultralytics.com/images/bus.jpg";
    private static final Path PROJECT_ROOT = Path.of(System.getProperty("user.dir"));

    @BeforeAll
    static void init() throws Exception {
        Path pythonBin = PROJECT_ROOT.resolve(".venv/bin/python3");
        Path jepLib = PROJECT_ROOT.resolve(".venv/lib/python3.13/site-packages/jep/libjep.jnilib");
        PythonRuntime.init(pythonBin, jepLib);
    }

    @Test
    @Order(1)
    void testModelLoad() throws Exception {
        try (Model model = new Model(MODEL_PATH)) {
            assertNotNull(model.getTaskType());
            Assertions.assertEquals(TaskType.DETECT, model.getTaskType());
            assertNotNull(model.getModelInfo());
            assertNotNull(model.getClassNames());
            assertTrue(model.getNumClasses() > 0);
            assertFalse(model.isClosed());
        }
    }

    @Test
    @Order(2)
    void testDetection() throws Exception {
        try (Model model = new Model(MODEL_PATH)) {
            InferenceResult result = model.predict(TEST_IMAGE);

            assertNotNull(result);
            assertInstanceOf(DetectionResult.class, result);

            DetectionResult dr = (DetectionResult) result;
            assertTrue(dr.getBoxes().size() > 0, "Should detect objects");

            ClassPrediction first = dr.getBoxes().get(0);
            assertNotNull(first.box());
            assertTrue(first.confidence() > 0);
            assertTrue(first.confidence() <= 1);
            assertNotNull(first.className());
            assertFalse(first.className().isEmpty());

            System.out.println("Detection: " + dr.getBoxes().size() + " objects");
            for (ClassPrediction p : dr.getBoxes()) {
                System.out.println("  " + p);
            }
        }
    }

    @Test
    @Order(3)
    void testDetectionWithConfig() throws Exception {
        try (Model model = new Model(MODEL_PATH)) {
            ModelConfig config = new ModelConfig()
                    .confidence(0.7f)
                    .imageSize(320);

            InferenceResult result = model.predict(TEST_IMAGE, config);
            assertNotNull(result);
            assertInstanceOf(DetectionResult.class, result);
            System.out.println("Detection (conf>0.7): " + result.count() + " objects");
        }
    }

    @Test
    @Order(4)
    void testSegmentation() throws Exception {
        try (Model model = new Model("yolov8n-seg.pt")) {
            assertEquals(TaskType.SEGMENT, model.getTaskType());

            InferenceResult result = model.predict(TEST_IMAGE);
            assertInstanceOf(SegmentationResult.class, result);

            SegmentationResult sr = (SegmentationResult) result;
            assertTrue(sr.getBoxes().size() > 0);
            assertTrue(sr.getMasks().size() > 0);

            Mask firstMask = sr.getMasks().get(0);
            assertTrue(firstMask.getPointCount() > 0);

            System.out.println("Segmentation: " + sr.getBoxes().size() + " objects, " +
                    sr.getMasks().size() + " masks");
        }
    }

    @Test
    @Order(5)
    void testClassification() throws Exception {
        try (Model model = new Model("yolov8n-cls.pt")) {
            assertEquals(TaskType.CLASSIFY, model.getTaskType());

            InferenceResult result = model.predict(TEST_IMAGE);
            assertInstanceOf(ClassificationResult.class, result);

            ClassificationResult cr = (ClassificationResult) result;
            assertTrue(cr.getTopK().size() > 0);
            assertTrue(cr.getTop1Confidence() > 0);

            System.out.println("Classification: top1=" + cr.getTop1ClassName() +
                    " (" + cr.getTop1Confidence() + ")");
        }
    }

    @Test
    @Order(6)
    void testPose() throws Exception {
        try (Model model = new Model("yolov8n-pose.pt")) {
            assertEquals(TaskType.POSE, model.getTaskType());

            InferenceResult result = model.predict(TEST_IMAGE);
            assertInstanceOf(PoseResult.class, result);

            PoseResult pr = (PoseResult) result;
            assertTrue(pr.getBoxes().size() > 0);
            assertTrue(pr.getKeypoints().size() > 0);

            KeypointCollection kc = pr.getKeypoints().get(0);
            assertEquals(17, kc.size()); // COCO 17 keypoints
            assertNotNull(kc.getNose());

            System.out.println("Pose: " + pr.personCount() + " persons");
        }
    }

    @Test
    @Order(7)
    void testOBB() throws Exception {
        try (Model model = new Model("yolov8n-obb.pt")) {
            assertEquals(TaskType.OBB, model.getTaskType());

            // OBB usually needs aerial/satellite images; use bus.jpg as fallback
            InferenceResult result = model.predict(TEST_IMAGE);
            assertInstanceOf(OBBResult.class, result);

            OBBResult or = (OBBResult) result;
            System.out.println("OBB: " + or.getPredictions().size() + " predictions");
            for (OBBPrediction p : or.getPredictions()) {
                System.out.println("  " + p.className() + " " + p.confidence());
            }
        }
    }

    @Test
    @Order(8)
    void testModelClose() throws Exception {
        Model model = new Model(MODEL_PATH);
        assertFalse(model.isClosed());
        model.close();
        assertTrue(model.isClosed());

        assertThrows(IllegalStateException.class, () -> model.predict(TEST_IMAGE));
    }

    @Test
    @Order(9)
    void testInferenceSpeed() throws Exception {
        try (Model model = new Model(MODEL_PATH)) {
            InferenceResult result = model.predict(TEST_IMAGE);
            InferenceSpeed speed = result.getSpeed();
            assertNotNull(speed);
            System.out.println("Speed: preprocess=" + speed.preprocessMs() +
                    "ms, inference=" + speed.inferenceMs() +
                    "ms, postprocess=" + speed.postprocessMs() + "ms");
        }
    }

    @Test
    @Order(15)
    void testBatchPrediction() throws Exception {
        try (Model model = new Model(MODEL_PATH)) {
            List<String> images = List.of(TEST_IMAGE, TEST_IMAGE, TEST_IMAGE);
            List<InferenceResult> results = model.predict(images);

            assertNotNull(results);
            assertEquals(3, results.size(), "Batch should return one result per image");

            for (int i = 0; i < results.size(); i++) {
                assertInstanceOf(DetectionResult.class, results.get(i));
                DetectionResult dr = (DetectionResult) results.get(i);
                assertTrue(dr.getBoxes().size() > 0, "Image " + i + " should have detections");
                System.out.println("Batch image " + i + ": " + dr.getBoxes().size() + " objects");
            }
        }
    }

    @Test
    @Order(16)
    void testVideoPrediction() throws Exception {
        String videoUrl = "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4";
        try (Model model = new Model(MODEL_PATH)) {
            AtomicInteger frameCount = new AtomicInteger(0);
            model.predictVideo(videoUrl, frame -> {
                int n = frameCount.incrementAndGet();
                if (n <= 3) {
                    assertTrue(frame.count() >= 0, "Frame should have valid count");
                    System.out.println("Video frame " + n + ": " + frame.count() + " objects");
                }
            });
            assertTrue(frameCount.get() > 0, "Should process at least one frame");
            System.out.println("Video: processed " + frameCount.get() + " frames");
        }
    }

    @Test
    @Order(17)
    void testStreamWithAnnotatedImage() throws Exception {
        String videoUrl = "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4";
        try (Model model = new Model(MODEL_PATH)) {
            ModelConfig config = new ModelConfig().vidStride(20);
            AtomicInteger frameCount = new AtomicInteger(0);

            new Thread(() -> {
                try { Thread.sleep(15000); } catch (InterruptedException ignored) {}
                model.stopStream();
            }).start();

            model.predictStream(videoUrl, config, true, sf -> {
                int n = frameCount.incrementAndGet();
                assertNotNull(sf.getResult(), "Should have inference result");
                assertTrue(sf.getFrameIndex() >= 0, "Should have valid frame index");

                if (n == 1) {
                    assertTrue(sf.hasImage(), "First frame should have annotated image");
                    assertTrue(sf.getAnnotatedImage().length > 1000,
                            "Annotated JPEG should be >1KB, got " + sf.getAnnotatedImage().length);
                    System.out.println("Stream frame: " + sf.getFrameIndex() +
                            ", objects=" + sf.getResult().count() +
                            ", image=" + sf.getAnnotatedImage().length + " bytes");
                }
            });

            assertTrue(frameCount.get() > 0, "Should process at least one frame");
            System.out.println("Stream: processed " + frameCount.get() + " frames with annotated images");
        }
    }

    @Test
    @Order(10)
    void testYOLO26Detection() throws Exception {
        try (Model model = new Model("yolo26n.pt")) {
            assertEquals(TaskType.DETECT, model.getTaskType());

            InferenceResult result = model.predict(TEST_IMAGE);
            assertInstanceOf(DetectionResult.class, result);

            DetectionResult dr = (DetectionResult) result;
            assertTrue(dr.getBoxes().size() > 0, "YOLO26 should detect objects");

            System.out.println("YOLO26 Detection: " + dr.getBoxes().size() + " objects");
            for (ClassPrediction p : dr.getBoxes()) {
                System.out.println("  " + p);
            }
        }
    }

    @Test
    @Order(11)
    void testOnnxInference() throws Exception {
        // Ensure ONNX model exists (exported from testExportOnnx or pre-existing)
        Path onnxPath = PROJECT_ROOT.resolve("yolov8n.onnx");
        if (!onnxPath.toFile().exists()) {
            try (Model model = new Model(MODEL_PATH)) {
                ExportResult exported = model.export(ExportFormat.ONNX);
                System.out.println("Exported ONNX: " + exported.getOutputPath());
            }
        }

        try (Model model = new Model("yolov8n.onnx", TaskType.DETECT)) {
            assertEquals(TaskType.DETECT, model.getTaskType());
            assertTrue(model.getNumClasses() > 0);

            InferenceResult result = model.predict(TEST_IMAGE);
            assertInstanceOf(DetectionResult.class, result);

            DetectionResult dr = (DetectionResult) result;
            assertTrue(dr.getBoxes().size() > 0, "ONNX model should detect objects");

            System.out.println("ONNX Detection: " + dr.getBoxes().size() + " objects");
            for (ClassPrediction p : dr.getBoxes()) {
                System.out.println("  " + p);
            }
        }
    }

    @Test
    @Order(12)
    void testExportOnnx() throws Exception {
        try (Model model = new Model(MODEL_PATH)) {
            ExportResult result = model.export(ExportFormat.ONNX);

            assertNotNull(result.getOutputPath());
            assertTrue(result.getFileSizeBytes() > 0);
            assertEquals(ExportFormat.ONNX, result.getFormat());

            System.out.println("Export: " + result.getOutputPath() +
                    " (" + result.getFileSizeMB() + " MB)");
        }
    }

    // ==================== Training Tests ====================

    @Test
    @Order(20)
    void testTrainWithCallback() throws Exception {
        try (Model model = new Model(MODEL_PATH)) {
            TrainingConfig config = new TrainingConfig()
                    .dataConfig("coco128.yaml")
                    .epochs(2)
                    .imageSize(320)
                    .batchSize(16)
                    .device(Device.cpu())
                    .project("runs/test_train")
                    .name("java_test")
                    .val(false)
                    .plots(false);

            AtomicInteger callbackCount = new AtomicInteger(0);

            TrainingResult result = model.train(config, (epoch, log) -> {
                callbackCount.incrementAndGet();
                System.out.println("  [callback] epoch " + epoch + ": " + log);
            });

            assertNotNull(result.getBestModelPath());
            assertFalse(result.getBestModelPath().isEmpty());
            assertTrue(result.getCompletedEpochs() > 0);
            assertTrue(result.getEpochMetrics().size() > 0, "Should have epoch metrics");

            System.out.println("Training result: " + result);
            System.out.println("Callbacks received: " + callbackCount.get());
            System.out.println("Epoch metrics: " + result.getEpochMetrics().size());
            for (EpochMetric m : result.getEpochMetrics()) {
                System.out.println("  " + m);
            }
        }
    }

    @Test
    @Order(21)
    void testValidate() throws Exception {
        try (Model model = new Model(MODEL_PATH)) {
            ValidationResult result = model.validate("coco128.yaml");

            assertTrue(result.getMAP50() >= 0);
            assertTrue(result.getMAP5095() >= 0);
            assertTrue(result.getPrecision() >= 0);
            assertTrue(result.getRecall() >= 0);

            System.out.println("Validation: mAP50=" + result.getMAP50() +
                    ", mAP50-95=" + result.getMAP5095() +
                    ", P=" + result.getPrecision() +
                    ", R=" + result.getRecall());

            List<PerClassMetric> perClass = result.getPerClassMetrics();
            if (!perClass.isEmpty()) {
                System.out.println("Per-class metrics: " + perClass.size() + " classes");
                for (int i = 0; i < Math.min(5, perClass.size()); i++) {
                    System.out.println("  " + perClass.get(i));
                }
            }
        }
    }

    @Test
    @Order(22)
    void testTrainThenPredict() throws Exception {
        // Train a quick model, then use it for prediction
        try (Model model = new Model(MODEL_PATH)) {
            TrainingConfig config = new TrainingConfig()
                    .dataConfig("coco128.yaml")
                    .epochs(1)
                    .imageSize(320)
                    .batchSize(16)
                    .device(Device.cpu())
                    .project("runs/test_train")
                    .name("predict_test")
                    .val(false)
                    .plots(false);

            TrainingResult trainResult = model.train(config);
            String bestModel = trainResult.getBestModelPath();
            assertNotNull(bestModel);
            System.out.println("Trained model: " + bestModel);

            // Load the trained model and predict
            try (Model trainedModel = new Model(bestModel, TaskType.DETECT)) {
                InferenceResult result = trainedModel.predict(TEST_IMAGE);
                assertInstanceOf(DetectionResult.class, result);
                DetectionResult dr = (DetectionResult) result;
                System.out.println("Trained model prediction: " + dr.getBoxes().size() + " objects");
            }
        }
    }
}
