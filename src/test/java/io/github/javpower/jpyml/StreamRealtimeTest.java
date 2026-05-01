package io.github.javpower.jpyml;

import io.github.javpower.jpyml.core.PythonRuntime;
import io.github.javpower.jpyml.ml.model.Model;
import io.github.javpower.jpyml.ml.model.ModelConfig;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.condition.DisabledIf;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertTrue;

@DisabledIf("isHeadless")
public class StreamRealtimeTest {

    static boolean isHeadless() {
        return GraphicsEnvironment.isHeadless();
    }

    private static final Path PROJECT_ROOT = Path.of(System.getProperty("user.dir"));
    private static final String MODEL_PATH = "yolov8n.pt"; // 按实际路径调整

    @BeforeAll
    static void init() throws Exception {
        Path pythonBin = PROJECT_ROOT.resolve(".venv/bin/python3");
        Path jepLib = PROJECT_ROOT.resolve(".venv/lib/python3.13/site-packages/jep/libjep.jnilib");
        PythonRuntime.init(pythonBin, jepLib);
    }
    @Test
    void testStreamWithRealtimeDisplay() throws Exception {
        RealtimeDemo ui = new RealtimeDemo();

        String videoUrl = "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4";
        try (Model model = new Model(MODEL_PATH)) {
            ModelConfig config = new ModelConfig().vidStride(10);
            AtomicInteger frameCount = new AtomicInteger(0);

            new Thread(() -> {
                try { Thread.sleep(15000); } catch (InterruptedException ignored) {}
                model.stopStream();
            }).start();

            model.predictStream("0", config, true, sf -> {
                int n = frameCount.incrementAndGet();
                if (sf.hasImage()) {
                    ui.updateFrame(
                            sf.getAnnotatedImage(),
                            sf.getFrameIndex(),
                            sf.getResult().count()
                    );
                }
            });

            assertTrue(frameCount.get() > 0, "Should process at least one frame");
            System.out.println("Stream finished. Total frames: " + frameCount.get());
        }
    }

    // ==================== Swing UI 组件 ====================

    static class RealtimeDemo extends JFrame {
        private final ImagePanel imagePanel = new ImagePanel();
        private final JLabel infoLabel = new JLabel("Waiting for stream...", SwingConstants.CENTER);

        public RealtimeDemo() {
            setTitle("YOLO Realtime Stream");
            setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            setLayout(new BorderLayout());
            add(imagePanel, BorderLayout.CENTER);
            add(infoLabel, BorderLayout.SOUTH);
            setSize(1280, 720);
            setLocationRelativeTo(null);
            setVisible(true);
        }

        public void updateFrame(byte[] jpegBytes, int frameIdx, int objCount) {
            SwingUtilities.invokeLater(() -> {
                try {
                    BufferedImage img = ImageIO.read(new ByteArrayInputStream(jpegBytes));
                    if (img != null) {
                        imagePanel.setImage(img);
                        infoLabel.setText(String.format(
                                "Frame: %d | Objects: %d | Image: %d bytes",
                                frameIdx, objCount, jpegBytes.length
                        ));
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
        }
    }

    static class ImagePanel extends JPanel {
        private BufferedImage image;

        public synchronized void setImage(BufferedImage img) {
            this.image = img;
            repaint();
        }

        @Override
        protected synchronized void paintComponent(Graphics g) {
            super.paintComponent(g);
            if (image == null) return;
            double scaleX = (double) getWidth() / image.getWidth();
            double scaleY = (double) getHeight() / image.getHeight();
            double scale = Math.min(scaleX, scaleY);
            int w = (int) (image.getWidth() * scale);
            int h = (int) (image.getHeight() * scale);
            int x = (getWidth() - w) / 2;
            int y = (getHeight() - h) / 2;
            g.drawImage(image, x, y, w, h, this);
        }
    }
}