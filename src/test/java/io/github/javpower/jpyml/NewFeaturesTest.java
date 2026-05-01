package io.github.javpower.jpyml;

import io.github.javpower.jpyml.core.PythonRuntime;
import io.github.javpower.jpyml.ml.annotation.ImageVisualizer;
import io.github.javpower.jpyml.ml.model.Model;
import io.github.javpower.jpyml.ml.model.ModelConfig;
import io.github.javpower.jpyml.ml.model.ModelHub;
import io.github.javpower.jpyml.ml.result.*;
import org.junit.jupiter.api.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class NewFeaturesTest {

    private static final Path PROJECT_ROOT = Path.of(System.getProperty("user.dir"));
    private static final String TEST_IMAGE = "bus.jpg";

    @BeforeAll
    static void init() throws Exception {
        Path pythonBin = PROJECT_ROOT.resolve(".venv/bin/python3");
        Path jepLib = PROJECT_ROOT.resolve(".venv/lib/python3.13/site-packages/jep/libjep.jnilib");
        Assumptions.assumeTrue(Files.exists(pythonBin), "Python venv not found");
        Assumptions.assumeTrue(Files.exists(jepLib), "Jep native lib not found");
        PythonRuntime.init(pythonBin, jepLib);
    }

    // ── Serialization Tests ─────────────────────────────────────────

    @Test
    @Order(1)
    void testDetectionResultToJson() throws Exception {
        try (Model model = new Model("yolov8n.pt")) {
            InferenceResult result = model.predict(TEST_IMAGE);
            assertInstanceOf(DetectionResult.class, result);
            String json = result.toJson();

            assertNotNull(json);
            assertTrue(json.startsWith("{"));
            assertTrue(json.contains("\"task\":\"detect\""));
            assertTrue(json.contains("\"boxes\""));
            assertTrue(json.endsWith("}"));

            System.out.println("Detection JSON (first 200 chars): " +
                    json.substring(0, Math.min(200, json.length())));
        }
    }

    @Test
    @Order(2)
    void testDetectionResultToMap() throws Exception {
        try (Model model = new Model("yolov8n.pt")) {
            InferenceResult result = model.predict(TEST_IMAGE);
            Map<String, Object> map = result.toMap();

            assertEquals("detect", map.get("task"));
            assertTrue((int) map.get("count") > 0);
            assertNotNull(map.get("boxes"));
            assertNotNull(map.get("speed"));
        }
    }

    @Test
    @Order(3)
    void testClassificationResultToJson() throws Exception {
        try (Model model = new Model("yolov8n-cls.pt")) {
            InferenceResult result = model.predict(TEST_IMAGE);
            assertInstanceOf(ClassificationResult.class, result);
            String json = result.toJson();

            assertTrue(json.contains("\"task\":\"classify\""));
            assertTrue(json.contains("\"predictions\""));
        }
    }

    @Test
    @Order(4)
    void testSegmentationResultToJson() throws Exception {
        try (Model model = new Model("yolov8n-seg.pt")) {
            InferenceResult result = model.predict(TEST_IMAGE);
            assertInstanceOf(SegmentationResult.class, result);
            String json = result.toJson();

            assertTrue(json.contains("\"task\":\"segment\""));
            assertTrue(json.contains("\"masks\""));
        }
    }

    @Test
    @Order(5)
    void testPoseResultToJson() throws Exception {
        try (Model model = new Model("yolov8n-pose.pt")) {
            InferenceResult result = model.predict(TEST_IMAGE);
            assertInstanceOf(PoseResult.class, result);
            String json = result.toJson();

            assertTrue(json.contains("\"task\":\"pose\""));
            assertTrue(json.contains("\"keypoints\""));
        }
    }

    // ── Direct Image Input Tests ────────────────────────────────────

    @Test
    @Order(10)
    void testPredictFromByteArray() throws Exception {
        Path imagePath = PROJECT_ROOT.resolve(TEST_IMAGE);
        Assumptions.assumeTrue(Files.exists(imagePath), "Test image not found");
        byte[] imageBytes = Files.readAllBytes(imagePath);

        try (Model model = new Model("yolov8n.pt")) {
            InferenceResult result = model.predict(imageBytes);
            assertInstanceOf(DetectionResult.class, result);
            assertTrue(result.count() > 0, "Should detect objects from byte[]");

            System.out.println("Byte[] prediction: " + result.count() + " objects");
        }
    }

    @Test
    @Order(11)
    void testPredictFromBufferedImage() throws Exception {
        Path imagePath = PROJECT_ROOT.resolve(TEST_IMAGE);
        Assumptions.assumeTrue(Files.exists(imagePath), "Test image not found");
        BufferedImage image = ImageIO.read(imagePath.toFile());

        try (Model model = new Model("yolov8n.pt")) {
            InferenceResult result = model.predict(image);
            assertInstanceOf(DetectionResult.class, result);
            assertTrue(result.count() > 0, "Should detect objects from BufferedImage");

            System.out.println("BufferedImage prediction: " + result.count() + " objects");
        }
    }

    @Test
    @Order(12)
    void testPredictFromBytesWithConfig() throws Exception {
        Path imagePath = PROJECT_ROOT.resolve(TEST_IMAGE);
        Assumptions.assumeTrue(Files.exists(imagePath), "Test image not found");
        byte[] imageBytes = Files.readAllBytes(imagePath);

        try (Model model = new Model("yolov8n.pt")) {
            InferenceResult result = model.predict(imageBytes,
                    new ModelConfig().confidence(0.7f).imageSize(320));

            assertNotNull(result);
            System.out.println("Byte[] with config: " + result.count() + " objects");
        }
    }

    // ── Async Tests ─────────────────────────────────────────────────

    @Test
    @Order(20)
    void testPredictAsync() throws Exception {
        try (Model model = new Model("yolov8n.pt")) {
            CompletableFuture<InferenceResult> future = model.predictAsync(TEST_IMAGE);
            InferenceResult result = future.get(30, TimeUnit.SECONDS);

            assertNotNull(result);
            assertTrue(result.count() > 0);

            System.out.println("Async prediction: " + result.count() + " objects");
        }
    }

    @Test
    @Order(21)
    void testPredictAsyncWithConfig() throws Exception {
        try (Model model = new Model("yolov8n.pt")) {
            CompletableFuture<InferenceResult> future = model.predictAsync(TEST_IMAGE,
                    new ModelConfig().confidence(0.7f));
            InferenceResult result = future.get(30, TimeUnit.SECONDS);

            assertNotNull(result);
        }
    }

    @Test
    @Order(22)
    void testPredictAsyncFromBytes() throws Exception {
        Path imagePath = PROJECT_ROOT.resolve(TEST_IMAGE);
        Assumptions.assumeTrue(Files.exists(imagePath), "Test image not found");
        byte[] imageBytes = Files.readAllBytes(imagePath);

        try (Model model = new Model("yolov8n.pt")) {
            CompletableFuture<InferenceResult> future = model.predictAsync(imageBytes);
            InferenceResult result = future.get(30, TimeUnit.SECONDS);

            assertNotNull(result);
            assertTrue(result.count() > 0);
        }
    }

    // ── ModelHub Tests ──────────────────────────────────────────────

    @Test
    @Order(30)
    void testModelHubRegistry() {
        assertTrue(ModelHub.isRegistered("yolov8n"));
        assertTrue(ModelHub.isRegistered("yolov8n-seg"));
        assertTrue(ModelHub.isRegistered("sam2.1_t"));
        assertFalse(ModelHub.isRegistered("nonexistent_model"));

        List<ModelHub.ModelEntry> models = ModelHub.listAvailable();
        assertFalse(models.isEmpty());
        assertTrue(models.size() > 15);
    }

    @Test
    @Order(31)
    void testModelHubGetCachedPath() {
        Path cached = ModelHub.getCachedPath("yolov8n");
        assertNotNull(cached);
    }

    // ── ImageVisualizer Tests ───────────────────────────────────────

    @Test
    @Order(40)
    void testVisualizeDetection() throws Exception {
        Path imagePath = PROJECT_ROOT.resolve(TEST_IMAGE);
        Assumptions.assumeTrue(Files.exists(imagePath), "Test image not found");
        BufferedImage image = ImageIO.read(imagePath.toFile());

        try (Model model = new Model("yolov8n.pt")) {
            InferenceResult result = model.predict(TEST_IMAGE);
            assertInstanceOf(DetectionResult.class, result);

            ImageVisualizer viz = new ImageVisualizer();
            BufferedImage annotated = viz.visualize(image, result);

            assertNotNull(annotated);
            assertEquals(image.getWidth(), annotated.getWidth());
            assertEquals(image.getHeight(), annotated.getHeight());

            System.out.println("Visualized detection: " + result.count() + " boxes on " +
                    annotated.getWidth() + "x" + annotated.getHeight() + " image");
        }
    }

    @Test
    @Order(41)
    void testVisualizeFromBytes() throws Exception {
        Path imagePath = PROJECT_ROOT.resolve(TEST_IMAGE);
        Assumptions.assumeTrue(Files.exists(imagePath), "Test image not found");
        byte[] imageBytes = Files.readAllBytes(imagePath);

        try (Model model = new Model("yolov8n.pt")) {
            InferenceResult result = model.predict(imageBytes);

            ImageVisualizer viz = new ImageVisualizer();
            byte[] annotatedBytes = viz.visualizeToBytes(imageBytes, result);

            assertNotNull(annotatedBytes);
            assertTrue(annotatedBytes.length > 0);

            BufferedImage readBack = ImageIO.read(new ByteArrayInputStream(annotatedBytes));
            assertNotNull(readBack);

            System.out.println("Visualized bytes: " + annotatedBytes.length + " bytes");
        }
    }

    @Test
    @Order(42)
    void testVisualizeSegmentation() throws Exception {
        Path imagePath = PROJECT_ROOT.resolve(TEST_IMAGE);
        Assumptions.assumeTrue(Files.exists(imagePath), "Test image not found");
        BufferedImage image = ImageIO.read(imagePath.toFile());

        try (Model model = new Model("yolov8n-seg.pt")) {
            InferenceResult result = model.predict(TEST_IMAGE);
            assertInstanceOf(SegmentationResult.class, result);

            ImageVisualizer viz = new ImageVisualizer().maskAlpha(0.4f);
            BufferedImage annotated = viz.visualize(image, result);

            assertNotNull(annotated);
            System.out.println("Visualized segmentation: " +
                    ((SegmentationResult) result).getMasks().size() + " masks");
        }
    }

    @Test
    @Order(43)
    void testVisualizePose() throws Exception {
        Path imagePath = PROJECT_ROOT.resolve(TEST_IMAGE);
        Assumptions.assumeTrue(Files.exists(imagePath), "Test image not found");
        BufferedImage image = ImageIO.read(imagePath.toFile());

        try (Model model = new Model("yolov8n-pose.pt")) {
            InferenceResult result = model.predict(TEST_IMAGE);
            assertInstanceOf(PoseResult.class, result);

            ImageVisualizer viz = new ImageVisualizer();
            BufferedImage annotated = viz.visualize(image, result);

            assertNotNull(annotated);
            System.out.println("Visualized pose: " + ((PoseResult) result).personCount() + " persons");
        }
    }
}
