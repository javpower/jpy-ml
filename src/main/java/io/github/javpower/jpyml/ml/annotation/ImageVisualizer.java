package io.github.javpower.jpyml.ml.annotation;

import io.github.javpower.jpyml.ml.model.TaskType;
import io.github.javpower.jpyml.ml.result.*;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Pure Java2D result visualization. Draws bounding boxes, masks, keypoints,
 * and labels on images without any Python dependency.
 */
public class ImageVisualizer {

    private static final int[] CLASS_COLORS = {
            0xFF3838, 0xFF9D97, 0xFF701F, 0xFFB21D, 0xCFD231,
            0x48F90A, 0x92CC17, 0x3DDB86, 0x1A9334, 0x00B445,
            0x3399FF, 0x12EAED, 0x1FDBDB, 0x8ED833, 0xDB8E33,
    };

    // COCO 17 skeleton connections
    private static final int[][] SKELETON = {
            {0, 1}, {0, 2}, {1, 3}, {2, 4},      // head
            {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, // arms
            {5, 11}, {6, 12}, {11, 12},            // torso
            {11, 13}, {13, 15}, {12, 14}, {14, 16}, // legs
    };

    private float lineWidth = 2.0f;
    private float fontSize = 14.0f;
    private boolean drawLabels = true;
    private float maskAlpha = 0.35f;

    public ImageVisualizer lineWidth(float w) { this.lineWidth = w; return this; }
    public ImageVisualizer fontSize(float s) { this.fontSize = s; return this; }
    public ImageVisualizer drawLabels(boolean v) { this.drawLabels = v; return this; }
    public ImageVisualizer maskAlpha(float a) { this.maskAlpha = a; return this; }

    /**
     * Draw inference results on a copy of the image.
     */
    public BufferedImage visualize(BufferedImage image, InferenceResult result) {
        BufferedImage canvas = deepCopy(image);
        Graphics2D g = canvas.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        try {
            switch (result.getTaskType()) {
                case DETECT -> drawDetections(g, (DetectionResult) result);
                case SEGMENT -> drawSegmentation(g, (SegmentationResult) result);
                case CLASSIFY -> drawClassification(g, (ClassificationResult) result, canvas.getWidth());
                case POSE -> drawPose(g, (PoseResult) result);
                case OBB -> drawOBB(g, (OBBResult) result);
            }
        } finally {
            g.dispose();
        }
        return canvas;
    }

    /**
     * Visualize from file path.
     */
    public BufferedImage visualize(String imagePath, InferenceResult result) throws IOException {
        return visualize(ImageIO.read(new File(imagePath)), result);
    }

    /**
     * Visualize from byte[] (JPEG/PNG).
     */
    public BufferedImage visualize(byte[] imageData, InferenceResult result) throws IOException {
        return visualize(ImageIO.read(new ByteArrayInputStream(imageData)), result);
    }

    /**
     * Get annotated image as JPEG bytes.
     */
    public byte[] visualizeToBytes(BufferedImage image, InferenceResult result) throws IOException {
        BufferedImage annotated = visualize(image, result);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(annotated, "jpg", baos);
        return baos.toByteArray();
    }

    public byte[] visualizeToBytes(byte[] imageData, InferenceResult result) throws IOException {
        return visualizeToBytes(ImageIO.read(new ByteArrayInputStream(imageData)), result);
    }

    // ── Detection ─────────────────────────────────────────────────────

    private void drawDetections(Graphics2D g, DetectionResult result) {
        int i = 0;
        for (ClassPrediction pred : result.getBoxes()) {
            Color color = classColor(i++);
            BoundingBox box = pred.box();
            g.setColor(color);
            g.setStroke(new BasicStroke(lineWidth));
            g.drawRect((int) box.x1(), (int) box.y1(),
                    (int) box.width(), (int) box.height());

            if (drawLabels) {
                drawLabel(g, String.format("%s %.0f%%", pred.className(), pred.confidence() * 100),
                        (int) box.x1(), (int) box.y1(), color);
            }
        }
    }

    // ── Segmentation ──────────────────────────────────────────────────

    private void drawSegmentation(Graphics2D g, SegmentationResult result) {
        // Draw masks first (underneath boxes)
        for (int i = 0; i < result.getMasks().size(); i++) {
            Mask mask = result.getMasks().get(i);
            Color color = classColor(i);
            drawMask(g, mask, color);
        }
        // Draw boxes on top
        int i = 0;
        for (ClassPrediction pred : result.getBoxes()) {
            Color color = classColor(i++);
            BoundingBox box = pred.box();
            g.setColor(color);
            g.setStroke(new BasicStroke(lineWidth));
            g.drawRect((int) box.x1(), (int) box.y1(),
                    (int) box.width(), (int) box.height());

            if (drawLabels) {
                drawLabel(g, String.format("%s %.0f%%", pred.className(), pred.confidence() * 100),
                        (int) box.x1(), (int) box.y1(), color);
            }
        }
    }

    // ── Classification ────────────────────────────────────────────────

    private void drawClassification(Graphics2D g, ClassificationResult result, int imageWidth) {
        g.setFont(g.getFont().deriveFont(fontSize));
        FontMetrics fm = g.getFontMetrics();
        List<ClassPrediction> topK = result.getTopK();
        int lineHeight = fm.getHeight() + 4;
        int boxHeight = topK.size() * lineHeight + 8;
        int boxWidth = imageWidth - 20;

        // Background
        g.setColor(new Color(0, 0, 0, 160));
        g.fillRect(10, 10, boxWidth, boxHeight);

        // Labels
        for (int i = 0; i < topK.size(); i++) {
            ClassPrediction pred = topK.get(i);
            Color color = classColor(i);
            int y = 10 + 4 + (i + 1) * lineHeight - fm.getDescent();
            String text = String.format("%s %.1f%%", pred.className(), pred.confidence() * 100);

            // Color bar
            g.setColor(color);
            g.fillRect(14, y - fm.getAscent(), 4, fm.getHeight());

            // Text
            g.setColor(Color.WHITE);
            g.drawString(text, 24, y);
        }
    }

    // ── Pose ──────────────────────────────────────────────────────────

    private void drawPose(Graphics2D g, PoseResult result) {
        List<ClassPrediction> boxes = result.getBoxes();
        List<KeypointCollection> persons = result.getKeypoints();

        for (int p = 0; p < persons.size(); p++) {
            Color color = classColor(p);
            KeypointCollection kpts = persons.get(p);

            // Draw skeleton
            g.setStroke(new BasicStroke(lineWidth));
            for (int[] pair : SKELETON) {
                if (pair[0] < kpts.size() && pair[1] < kpts.size()) {
                    Keypoint a = kpts.get(pair[0]);
                    Keypoint b = kpts.get(pair[1]);
                    if (a.confidence() > 0.3f && b.confidence() > 0.3f) {
                        g.setColor(color);
                        g.drawLine((int) a.x(), (int) a.y(), (int) b.x(), (int) b.y());
                    }
                }
            }

            // Draw keypoints
            for (Keypoint k : kpts.getAll()) {
                if (k.confidence() > 0.3f) {
                    g.setColor(color);
                    g.fillOval((int) k.x() - 3, (int) k.y() - 3, 6, 6);
                }
            }

            // Draw box
            if (p < boxes.size()) {
                BoundingBox box = boxes.get(p).box();
                g.setColor(color);
                g.setStroke(new BasicStroke(lineWidth));
                g.drawRect((int) box.x1(), (int) box.y1(),
                        (int) box.width(), (int) box.height());
            }
        }
    }

    // ── OBB ───────────────────────────────────────────────────────────

    private void drawOBB(Graphics2D g, OBBResult result) {
        int i = 0;
        for (OBBPrediction pred : result.getPredictions()) {
            Color color = classColor(i++);
            RotatedBoundingBox rbox = pred.box();

            AffineTransform old = g.getTransform();
            g.rotate(Math.toRadians(rbox.angleDegrees()), rbox.centerX(), rbox.centerY());

            float hw = rbox.width() / 2;
            float hh = rbox.height() / 2;
            g.setColor(color);
            g.setStroke(new BasicStroke(lineWidth));
            g.drawRect((int) (rbox.centerX() - hw), (int) (rbox.centerY() - hh),
                    (int) rbox.width(), (int) rbox.height());

            g.setTransform(old);

            if (drawLabels) {
                drawLabel(g, String.format("%s %.0f%%", pred.className(), pred.confidence() * 100),
                        (int) rbox.centerX(), (int) (rbox.centerY() - rbox.height() / 2), color);
            }
        }
    }

    // ── Shared drawing helpers ────────────────────────────────────────

    private void drawMask(Graphics2D g, Mask mask, Color color) {
        float[][] polygon = mask.getPolygon();
        if (polygon.length < 3) return;

        int[] xPoints = new int[polygon.length];
        int[] yPoints = new int[polygon.length];
        for (int i = 0; i < polygon.length; i++) {
            xPoints[i] = (int) polygon[i][0];
            yPoints[i] = (int) polygon[i][1];
        }

        // Semi-transparent fill
        g.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, maskAlpha));
        g.setColor(color);
        g.fillPolygon(xPoints, yPoints, polygon.length);

        // Outline
        g.setComposite(AlphaComposite.SrcOver);
        g.setStroke(new BasicStroke(lineWidth));
        g.drawPolygon(xPoints, yPoints, polygon.length);
    }

    private void drawLabel(Graphics2D g, String text, int x, int y, Color bgColor) {
        if (!drawLabels) return;
        g.setFont(g.getFont().deriveFont(fontSize));
        FontMetrics fm = g.getFontMetrics();
        int tw = fm.stringWidth(text) + 6;
        int th = fm.getHeight() + 2;
        int labelY = y - th;
        if (labelY < 0) labelY = y + th;

        g.setColor(bgColor);
        g.fillRect(x, labelY, tw, th);
        g.setColor(Color.WHITE);
        g.drawString(text, x + 3, labelY + fm.getAscent());
    }

    private static Color classColor(int index) {
        int rgb = CLASS_COLORS[index % CLASS_COLORS.length];
        return new Color(rgb, true);
    }

    private static BufferedImage deepCopy(BufferedImage src) {
        BufferedImage copy = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g = copy.createGraphics();
        g.drawImage(src, 0, 0, null);
        g.dispose();
        return copy;
    }
}
