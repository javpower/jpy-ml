package io.github.javpower.jpyml;

import io.github.javpower.jpyml.core.PythonRuntime;
import io.github.javpower.jpyml.ml.model.Prompt;
import io.github.javpower.jpyml.ml.model.SAM2Model;
import io.github.javpower.jpyml.ml.model.SAM2VideoTracker;
import io.github.javpower.jpyml.ml.model.SAM3Model;
import io.github.javpower.jpyml.ml.result.*;
import org.junit.jupiter.api.*;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Path;
import javax.imageio.ImageIO;

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
    private static final String SAM3_MODEL = "sam3.pt";  // Use tiny model for fast testing

    @BeforeAll
    static void init() throws Exception {
        Path pythonBin = PROJECT_ROOT.resolve(".venv/bin/python3");
        Path jepLib = PROJECT_ROOT.resolve(".venv/lib/python3.13/site-packages/jep/libjep.jnilib");
        PythonRuntime.init(pythonBin, jepLib);
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
    void testSAM2VideoTracker() throws Exception {
        String videoUrl = "/Volumes/macEx/AI2/jp/jpy-ml/solutions_ci_demo.mp4";

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
    void testSAM3TextPrompt() throws Exception {
        try (SAM3Model sam = new SAM3Model(SAM3_MODEL)) {
            SAM3Result result = sam.predictText(TEST_IMAGE, "person", "bus");

            assertNotNull(result);
            assertNotNull(result.sourcePath());
            assertTrue(result.count() > 0, "Should produce at least one mask");

            Mask bestMask = result.bestMask();
            assertNotNull(bestMask);

            // Draw and save annotated image for visual inspection
            BufferedImage img = ImageIO.read(new File(TEST_IMAGE));
            Graphics2D g = img.createGraphics();
            g.setStroke(new BasicStroke(3));
            Color[] colors = {Color.RED, Color.GREEN, Color.BLUE, Color.ORANGE, Color.MAGENTA, Color.CYAN};
            for (int i = 0; i < result.count(); i++) {
                Color color = colors[i % colors.length];
                Mask mask = result.masks().get(i);
                float[][] polygon = mask.getPolygon();
                int[] xPoints = new int[polygon.length];
                int[] yPoints = new int[polygon.length];
                for (int j = 0; j < polygon.length; j++) {
                    xPoints[j] = Math.round(polygon[j][0]);
                    yPoints[j] = Math.round(polygon[j][1]);
                }
                // Fill mask with semi-transparent color
                g.setColor(new Color(color.getRed(), color.getGreen(), color.getBlue(), 80));
                g.fillPolygon(xPoints, yPoints, polygon.length);
                // Draw mask outline
                g.setColor(color);
                g.drawPolygon(xPoints, yPoints, polygon.length);
                // Draw score label
                float score = i < result.scores().size() ? result.scores().get(i) : 0;
                int labelX = xPoints.length > 0 ? xPoints[0] : 10;
                int labelY = yPoints.length > 0 ? yPoints[0] : 20;
                g.setFont(new Font("SansSerif", Font.BOLD, 14));
                g.drawString(String.format("%.2f", score), labelX, labelY);
            }
            g.dispose();

            Path outputPath = PROJECT_ROOT.resolve("sam3_text_result.jpg");
            ImageIO.write(img, "jpg", outputPath.toFile());
            System.out.println("SAM3 text prompt: " + result.count() + " masks, saved to " + outputPath);
        }
    }

    @Test
    @Order(31)
    void testSAM3ExemplarPrompt() throws Exception {
        try (SAM3Model sam = new SAM3Model(SAM3_MODEL)) {
            BoundingBox exemplarBox = new BoundingBox(100, 100, 300, 300);
            SAM3Result result = sam.predictExemplar(TEST_IMAGE, TEST_IMAGE, exemplarBox);

            assertNotNull(result);
            assertTrue(result.count() > 0, "Should produce at least one mask");

            System.out.println("SAM3 exemplar prompt: " + result.count() + " masks");
        }
    }

    @Test
    @Order(32)
    void testSAM3FilterByScore() throws Exception {
        try (SAM3Model sam = new SAM3Model(SAM3_MODEL)) {
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
    void testSAM3ModelClose() throws Exception {
        SAM3Model sam = new SAM3Model(SAM3_MODEL);
        assertFalse(sam.isClosed());
        sam.close();
        assertTrue(sam.isClosed());

        assertThrows(IllegalStateException.class, () -> {
            sam.predictText(TEST_IMAGE, "person");
        });
    }
}
