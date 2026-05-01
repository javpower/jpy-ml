package io.github.javpower.jpyml.mp;

import io.github.javpower.jpyml.core.PythonRuntime;
import org.junit.jupiter.api.*;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class MediaPipeEngineTest {

    private static Path testImage;
    private static MediaPipeEngine engine;

    @BeforeAll
    static void init() throws Exception {
        Path projectRoot = Path.of(System.getProperty("user.dir"));
        Path pythonBin = projectRoot.resolve(".venv/bin/python3");
        Path jepLib = projectRoot.resolve(".venv/lib/python3.13/site-packages/jep/libjep.jnilib");
        Assumptions.assumeTrue(Files.exists(pythonBin), "Python venv not found");
        Assumptions.assumeTrue(Files.exists(jepLib), "Jep native lib not found");
        PythonRuntime.init(pythonBin, jepLib);

        engine = new MediaPipeEngine();

        testImage = projectRoot.resolve("bus.jpg");
        Assumptions.assumeTrue(Files.exists(testImage), "Test image bus.jpg not found");
    }

    @AfterAll
    static void cleanup() {
        // testImage is a local file, no cleanup needed
    }

    @Test
    @Order(1)
    void testDetectHands() throws Exception {
        MediaPipeEngine.HandResult result = engine.detectHands(testImage.toString());
        assertNotNull(result);
        assertNotNull(result.hands());
    }

    @Test
    @Order(2)
    void testDetectFace() throws Exception {
        MediaPipeEngine.FaceResult result = engine.detectFace("face.jpeg");
        assertNotNull(result);
        assertNotNull(result.faces());
    }

    @Test
    @Order(3)
    void testDetectPose() throws Exception {
        MediaPipeEngine.PoseResult result = engine.detectPose(testImage.toString());
        assertNotNull(result);
        assertNotNull(result.landmarks());
    }

    @Test
    @Order(4)
    void testClose() throws Exception {
        try (MediaPipeEngine mp = new MediaPipeEngine()) {
            // Verify engine can be created and closed
        }
    }
}
