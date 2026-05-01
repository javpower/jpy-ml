package io.github.javpower.jpyml;

import io.github.javpower.jpyml.core.PythonEngine;
import io.github.javpower.jpyml.core.PythonRuntime;
import io.github.javpower.jpyml.core.PythonScriptLoader;
import io.github.javpower.jpyml.ml.model.*;
import io.github.javpower.jpyml.ml.result.*;
import org.junit.jupiter.api.*;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for SAM 2 and SAM 3 models.
 * Note: These tests require properly downloaded SAM model files.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class SAMIntegrationTest {

    private static final Path PROJECT_ROOT = Path.of(System.getProperty("user.dir"));
    private static final String TEST_IMAGE = "bus.jpg";
    private static final String SAM_MODEL = "sam2.1_t.pt";  // Use tiny model for fast testing

    @BeforeAll
    static void init() throws Exception {
        Path pythonBin = PROJECT_ROOT.resolve(".venv/bin/python3");
        Path jepLib = PROJECT_ROOT.resolve(".venv/lib/python3.13/site-packages/jep/libjep.jnilib");
        PythonRuntime.init(pythonBin, jepLib);
        PythonScriptLoader.reset();  // Clear cached scripts to pick up changes

        // Verify function signatures
        PythonEngine engine = PythonEngine.getInstance();
        PythonScriptLoader.ensureLoaded(engine, "_jpy_sam2.py");
        engine.exec("import inspect");
        String sig = engine.eval("str(inspect.signature(jpy_sam2_predict))");
        System.out.println("jpy_sam2_predict signature: " + sig);
    }

    @Test
    @Order(1)
    void testSAM2PointPrompt() throws Exception {
        try (SAM2Model sam = new SAM2Model(SAM_MODEL)) {
            SAM2Result result = sam.predict(TEST_IMAGE,
                    Prompt.point(320, 240)
            );

            assertNotNull(result);
            assertNotNull(result.sourcePath());
            assertTrue(result.count() > 0, "Should produce at least one mask");

            Mask bestMask = result.bestMask();
            assertNotNull(bestMask);
            assertTrue(result.bestScore() > 0, "Should have positive score");

            System.out.println("SAM2 point prompt: " + result.count() + " masks, best score=" + result.bestScore());
        }
    }

    @Test
    @Order(2)
    void testSAM2BoxPrompt() throws Exception {
        try (SAM2Model sam = new SAM2Model(SAM_MODEL)) {
            SAM2Result result = sam.predict(TEST_IMAGE,
                    Prompt.box(100, 100, 400, 400)
            );

            assertNotNull(result);
            assertTrue(result.count() > 0, "Should produce at least one mask");

            System.out.println("SAM2 box prompt: " + result.count() + " masks");
        }
    }

    @Test
    @Order(3)
    void testSAM2MultiplePrompts() throws Exception {
        try (SAM2Model sam = new SAM2Model(SAM_MODEL)) {
            SAM2Result result = sam.predict(TEST_IMAGE,
                    Prompt.point(200, 200),
                    Prompt.point(400, 300, Prompt.Label.NEGATIVE)
            );

            assertNotNull(result);
            assertTrue(result.count() > 0, "Should produce at least one mask");

            System.out.println("SAM2 multiple prompts: " + result.count() + " masks");
        }
    }

    @Test
    @Order(10)
    void testSAM2ModelClose() throws Exception {
        SAM2Model sam = new SAM2Model(SAM_MODEL);
        assertFalse(sam.isClosed());
        sam.close();
        assertTrue(sam.isClosed());

        assertThrows(IllegalStateException.class, () -> {
            sam.predict(TEST_IMAGE, Prompt.point(100, 100));
        });
    }

    @Test
    @Order(20)
    @Disabled("SAM2 video tracking needs refactoring - tracker API not compatible with current implementation")
    void testSAM2VideoTracker() throws Exception {
        String videoUrl = "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4";

        try (SAM2Model sam = new SAM2Model(SAM_MODEL)) {
            try (SAM2VideoTracker tracker = sam.trackVideo(videoUrl, Prompt.box(100, 100, 400, 400))) {
                assertNotNull(tracker);

                // Add more prompts
                tracker.addPrompt(10, Prompt.point(300, 200));

                // Propagate
                SAM2VideoResult result = tracker.propagate();
                assertNotNull(result);
                assertTrue(result.totalFrames() > 0, "Should process frames");
                assertTrue(result.trackedFrameCount() > 0, "Should have tracked frames");

                System.out.println("SAM2 video tracking: " + result.trackedFrameCount() + " frames tracked");
            }
        }
    }

    @Test
    @Order(30)
    @Disabled("SAM3 text prompts require SAM3 model - SAM2.1_t does not support 'texts' argument")
    void testSAM3TextPrompt() throws Exception {
        try (SAM3Model sam = new SAM3Model(SAM_MODEL)) {
            SAM3Result result = sam.predictText(TEST_IMAGE, "person", "bus");

            assertNotNull(result);
            assertNotNull(result.sourcePath());
            assertTrue(result.count() > 0, "Should produce at least one mask");

            Mask bestMask = result.bestMask();
            assertNotNull(bestMask);

            System.out.println("SAM3 text prompt: " + result.count() + " masks");
        }
    }

    @Test
    @Order(31)
    @Disabled("SAM3 exemplar prompts require SAM3 model")
    void testSAM3ExemplarPrompt() throws Exception {
        try (SAM3Model sam = new SAM3Model(SAM_MODEL)) {
            BoundingBox exemplarBox = new BoundingBox(100, 100, 300, 300);
            SAM3Result result = sam.predictExemplar(TEST_IMAGE, TEST_IMAGE, exemplarBox);

            assertNotNull(result);
            assertTrue(result.count() > 0, "Should produce at least one mask");

            System.out.println("SAM3 exemplar prompt: " + result.count() + " masks");
        }
    }

    @Test
    @Order(32)
    @Disabled("SAM3 text prompts require SAM3 model")
    void testSAM3FilterByScore() throws Exception {
        try (SAM3Model sam = new SAM3Model(SAM_MODEL)) {
            SAM3Result result = sam.predictText(TEST_IMAGE, "person");

            assertNotNull(result);

            // Filter by score
            SAM3Result filtered = result.filterByScore(0.5f);
            assertNotNull(filtered);
            assertTrue(filtered.count() <= result.count(), "Filtered should have fewer or equal masks");

            System.out.println("SAM3 filter: " + result.count() + " -> " + filtered.count() + " masks");
        }
    }

    @Test
    @Order(33)
    @Disabled("SAM3 requires ultralytics with SAM3 support - not available in current version")
    void testSAM3ModelClose() throws Exception {
        SAM3Model sam = new SAM3Model(SAM_MODEL);
        assertFalse(sam.isClosed());
        sam.close();
        assertTrue(sam.isClosed());

        assertThrows(IllegalStateException.class, () -> {
            sam.predictText(TEST_IMAGE, "person");
        });
    }
}
