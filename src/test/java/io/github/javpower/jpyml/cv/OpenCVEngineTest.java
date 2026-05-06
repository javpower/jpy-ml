package io.github.javpower.jpyml.cv;

import io.github.javpower.jpyml.core.PythonRuntime;
import org.junit.jupiter.api.*;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class OpenCVEngineTest {

    private static Path testImage;
    private static Path outputDir;
    private static OpenCVEngine engine;

    @BeforeAll
    static void init() throws Exception {
        Path projectRoot = Path.of(System.getProperty("user.dir"));
        Path pythonBin = projectRoot.resolve(".venv/bin/python3");
        Path jepLib = projectRoot.resolve(".venv/lib/python3.12/site-packages/jep/libjep.jnilib");
        Assumptions.assumeTrue(Files.exists(pythonBin), "Python venv not found");
        Assumptions.assumeTrue(Files.exists(jepLib), "Jep native lib not found");
        PythonRuntime.init(pythonBin, jepLib);

        engine = new OpenCVEngine();

        // Download test image
        testImage = Files.createTempFile("jpy-test-opencv-", ".jpg");
        try (var in = new java.net.URL("https://ultralytics.com/images/bus.jpg").openStream()) {
            Files.copy(in, testImage, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }

        outputDir = Files.createTempDirectory("jpy-test-opencv-out");
    }

    @AfterAll
    static void cleanup() throws Exception {
        if (engine != null) {
            engine.close();
            engine = null;
        }
        if (testImage != null) Files.deleteIfExists(testImage);
        if (outputDir != null) {
            try (var stream = Files.walk(outputDir)) {
                stream.sorted(java.util.Comparator.reverseOrder()).map(Path::toFile).forEach(java.io.File::delete);
            }
        }
    }

    @Test
    @Order(1)
    void testImread() throws Exception {
        OpenCVEngine.ImageInfo info = engine.imread(testImage.toString());
        assertNotNull(info);
        assertEquals(testImage.toString(), info.path());
        assertTrue(info.width() > 0);
        assertTrue(info.height() > 0);
        assertTrue(info.channels() > 0);
    }

    @Test
    @Order(2)
    void testCvtColor() throws Exception {
        String out = outputDir.resolve("gray.jpg").toString();
        String result = engine.cvtColor(testImage.toString(), out, "BGR2GRAY");
        assertEquals(out, result);
        assertTrue(Files.exists(Path.of(out)));
    }

    @Test
    @Order(3)
    void testResize() throws Exception {
        String out = outputDir.resolve("resized.jpg").toString();
        String result = engine.resize(testImage.toString(), out, 320, 240);
        assertEquals(out, result);

        OpenCVEngine.ImageInfo info = engine.imread(out);
        assertEquals(320, info.width());
        assertEquals(240, info.height());
    }

    @Test
    @Order(4)
    void testBlur() throws Exception {
        String out = outputDir.resolve("blurred.jpg").toString();
        String result = engine.blur(testImage.toString(), out, 5);
        assertEquals(out, result);
        assertTrue(Files.exists(Path.of(out)));
    }

    @Test
    @Order(5)
    void testCanny() throws Exception {
        String out = outputDir.resolve("edges.jpg").toString();
        String result = engine.canny(testImage.toString(), out, 100, 200);
        assertEquals(out, result);
        assertTrue(Files.exists(Path.of(out)));
    }

    @Test
    @Order(6)
    void testThreshold() throws Exception {
        String out = outputDir.resolve("threshold.jpg").toString();
        String result = engine.threshold(testImage.toString(), out, 127, 255);
        assertEquals(out, result);
        assertTrue(Files.exists(Path.of(out)));
    }

    @Test
    @Order(7)
    void testMorphology() throws Exception {
        String out = outputDir.resolve("morph.jpg").toString();
        String result = engine.morphology(testImage.toString(), out, "DILATE", 3);
        assertEquals(out, result);
        assertTrue(Files.exists(Path.of(out)));
    }

    @Test
    @Order(8)
    void testFindContours() throws Exception {
        OpenCVEngine.ContourResult result = engine.findContours(testImage.toString(), null);
        assertNotNull(result);
        assertTrue(result.count() >= 0);
        assertNotNull(result.contours());
    }
}
