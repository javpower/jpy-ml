package io.github.javpower.jpyml.cv;

import io.github.javpower.jpyml.core.PythonEngine;
import io.github.javpower.jpyml.core.PythonScriptLoader;
import io.github.javpower.jpyml.exception.InferenceException;
import jep.JepException;

import java.util.List;
import java.util.Map;

/**
 * OpenCV engine for traditional computer vision operations.
 * <p>
 * Provides Java API for common OpenCV operations like image I/O,
 * color conversion, filtering, edge detection, and contour finding.
 * <p>
 * Usage:
 * <pre>
 *   OpenCVEngine cv = new OpenCVEngine();
 *   cv.cvtColor("input.jpg", "output.jpg", "BGR2GRAY");
 *   cv.canny("input.jpg", "edges.jpg", 100, 200);
 * </pre>
 */
public class OpenCVEngine {

    private final PythonEngine engine;
    private boolean scriptsLoaded = false;

    public OpenCVEngine() throws JepException {
        this.engine = PythonEngine.getInstance();
    }

    private void ensureScripts() throws JepException {
        if (!scriptsLoaded) {
            PythonScriptLoader.ensureLoaded(engine, "_jpy_opencv.py");
            scriptsLoaded = true;
        }
    }

    /**
     * Read image and get its properties.
     *
     * @param imagePath path to the image
     * @return ImageInfo with width, height, channels
     */
    public ImageInfo imread(String imagePath) throws InferenceException {
        try {
            ensureScripts();
            engine.put("_jpy_cv_path", imagePath);
            engine.exec("_jpy_cv_info = jpy_imread(_jpy_cv_path)");
            @SuppressWarnings("unchecked")
            Map<String, Object> result = engine.eval("_jpy_cv_info");
            return new ImageInfo(
                    (String) result.get("path"),
                    ((Number) result.get("width")).intValue(),
                    ((Number) result.get("height")).intValue(),
                    ((Number) result.get("channels")).intValue()
            );
        } catch (JepException e) {
            throw new InferenceException("Failed to read image: " + imagePath, e);
        }
    }

    /**
     * Write image to file.
     *
     * @param sourcePath path to the source image
     * @param outputPath path to save the image
     * @return output path
     */
    public String imwrite(String sourcePath, String outputPath) throws InferenceException {
        try {
            ensureScripts();
            engine.put("_jpy_cv_src", sourcePath);
            engine.put("_jpy_cv_out", outputPath);
            engine.exec("_jpy_cv_result = jpy_imwrite(_jpy_cv_src, _jpy_cv_out)");
            return engine.eval("_jpy_cv_result");
        } catch (JepException e) {
            throw new InferenceException("Failed to write image", e);
        }
    }

    /**
     * Convert image color space.
     *
     * @param sourcePath source image path
     * @param outputPath output image path
     * @param conversion color conversion code (e.g., "BGR2GRAY", "BGR2RGB")
     * @return output path
     */
    public String cvtColor(String sourcePath, String outputPath, String conversion) throws InferenceException {
        try {
            ensureScripts();
            engine.put("_jpy_cv_src", sourcePath);
            engine.put("_jpy_cv_out", outputPath);
            engine.put("_jpy_cv_conv", conversion);
            engine.exec("_jpy_cv_result = jpy_cvtColor(_jpy_cv_src, _jpy_cv_out, _jpy_cv_conv)");
            return engine.eval("_jpy_cv_result");
        } catch (JepException e) {
            throw new InferenceException("Failed to convert color", e);
        }
    }

    /**
     * Resize an image.
     *
     * @param sourcePath source image path
     * @param outputPath output image path
     * @param width      target width
     * @param height     target height
     * @return output path
     */
    public String resize(String sourcePath, String outputPath, int width, int height) throws InferenceException {
        try {
            ensureScripts();
            engine.put("_jpy_cv_src", sourcePath);
            engine.put("_jpy_cv_out", outputPath);
            engine.put("_jpy_cv_w", width);
            engine.put("_jpy_cv_h", height);
            engine.exec("_jpy_cv_result = jpy_resize(_jpy_cv_src, _jpy_cv_out, _jpy_cv_w, _jpy_cv_h)");
            return engine.eval("_jpy_cv_result");
        } catch (JepException e) {
            throw new InferenceException("Failed to resize image", e);
        }
    }

    /**
     * Apply Gaussian blur.
     *
     * @param sourcePath source image path
     * @param outputPath output image path
     * @param kernelSize kernel size (must be odd)
     * @return output path
     */
    public String blur(String sourcePath, String outputPath, int kernelSize) throws InferenceException {
        try {
            ensureScripts();
            engine.put("_jpy_cv_src", sourcePath);
            engine.put("_jpy_cv_out", outputPath);
            engine.put("_jpy_cv_ksize", kernelSize);
            engine.exec("_jpy_cv_result = jpy_blur(_jpy_cv_src, _jpy_cv_out, _jpy_cv_ksize, _jpy_cv_ksize)");
            return engine.eval("_jpy_cv_result");
        } catch (JepException e) {
            throw new InferenceException("Failed to apply blur", e);
        }
    }

    /**
     * Apply Canny edge detection.
     *
     * @param sourcePath source image path
     * @param outputPath output image path
     * @param threshold1 first threshold
     * @param threshold2 second threshold
     * @return output path
     */
    public String canny(String sourcePath, String outputPath, double threshold1, double threshold2)
            throws InferenceException {
        try {
            ensureScripts();
            engine.put("_jpy_cv_src", sourcePath);
            engine.put("_jpy_cv_out", outputPath);
            engine.put("_jpy_cv_t1", threshold1);
            engine.put("_jpy_cv_t2", threshold2);
            engine.exec("_jpy_cv_result = jpy_canny(_jpy_cv_src, _jpy_cv_out, _jpy_cv_t1, _jpy_cv_t2)");
            return engine.eval("_jpy_cv_result");
        } catch (JepException e) {
            throw new InferenceException("Failed to apply Canny edge detection", e);
        }
    }

    /**
     * Find contours in an image.
     *
     * @param sourcePath source image path
     * @param outputPath optional path to save contour visualization (null to skip)
     * @return contour result with contour data
     */
    public ContourResult findContours(String sourcePath, String outputPath) throws InferenceException {
        try {
            ensureScripts();
            engine.put("_jpy_cv_src", sourcePath);
            engine.put("_jpy_cv_out", outputPath);
            engine.exec("_jpy_cv_result = jpy_find_contours(_jpy_cv_src, _jpy_cv_out)");
            @SuppressWarnings("unchecked")
            Map<String, Object> result = engine.eval("_jpy_cv_result");

            int count = ((Number) result.get("count")).intValue();
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> contours = (List<Map<String, Object>>) result.get("contours");

            return new ContourResult(count, contours);
        } catch (JepException e) {
            throw new InferenceException("Failed to find contours", e);
        }
    }

    /**
     * Image information record.
     */
    public record ImageInfo(String path, int width, int height, int channels) {}

    /**
     * Contour detection result.
     */
    public record ContourResult(int count, List<Map<String, Object>> contours) {}
}
