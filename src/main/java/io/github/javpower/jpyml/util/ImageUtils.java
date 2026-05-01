package io.github.javpower.jpyml.util;

import io.github.javpower.jpyml.core.PythonEngine;
import jep.JepException;

import java.util.Map;

/**
 * Image utilities backed by Python PIL.
 */
public class ImageUtils {

    private final PythonEngine engine;

    public ImageUtils() throws JepException {
        this.engine = PythonEngine.getInstance();
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> getImageInfo(String imagePath) {
        try {
            engine.put("_jpy_img_path", imagePath);
            engine.exec("""
                    from PIL import Image as _jpy_pil_img
                    _jpy_img_obj = _jpy_pil_img.open(_jpy_img_path)
                    _jpy_img_info = {
                        'width': _jpy_img_obj.width,
                        'height': _jpy_img_obj.height,
                        'mode': _jpy_img_obj.mode,
                        'format': _jpy_img_obj.format
                    }
                    _jpy_img_obj.close()
                    """);
            return engine.eval("_jpy_img_info");
        } catch (JepException e) {
            throw new RuntimeException("Failed to get image info: " + imagePath, e);
        }
    }

    public void resize(String inputPath, String outputPath, int width, int height) {
        try {
            engine.put("_jpy_img_in", inputPath);
            engine.put("_jpy_img_out", outputPath);
            engine.put("_jpy_img_w", width);
            engine.put("_jpy_img_h", height);
            engine.exec("""
                    from PIL import Image as _jpy_pil_img
                    _jpy_img_obj = _jpy_pil_img.open(_jpy_img_in)
                    _jpy_img_obj = _jpy_img_obj.resize((_jpy_img_w, _jpy_img_h))
                    _jpy_img_obj.save(_jpy_img_out)
                    _jpy_img_obj.close()
                    """);
        } catch (JepException e) {
            throw new RuntimeException("Failed to resize image: " + inputPath, e);
        }
    }

    public void convert(String inputPath, String outputPath, String format) {
        try {
            engine.put("_jpy_img_in", inputPath);
            engine.put("_jpy_img_out", outputPath);
            engine.put("_jpy_img_fmt", format.toUpperCase());
            engine.exec("""
                    from PIL import Image as _jpy_pil_img
                    _jpy_img_obj = _jpy_pil_img.open(_jpy_img_in)
                    _jpy_img_obj.save(_jpy_img_out, format=_jpy_img_fmt)
                    _jpy_img_obj.close()
                    """);
        } catch (JepException e) {
            throw new RuntimeException("Failed to convert image: " + inputPath, e);
        }
    }

    public void crop(String inputPath, String outputPath, int x1, int y1, int x2, int y2) {
        try {
            engine.put("_jpy_img_in", inputPath);
            engine.put("_jpy_img_out", outputPath);
            engine.put("_jpy_img_box", new int[]{x1, y1, x2, y2});
            engine.exec("""
                    from PIL import Image as _jpy_pil_img
                    _jpy_img_obj = _jpy_pil_img.open(_jpy_img_in)
                    _jpy_img_obj = _jpy_img_obj.crop(_jpy_img_box)
                    _jpy_img_obj.save(_jpy_img_out)
                    _jpy_img_obj.close()
                    """);
        } catch (JepException e) {
            throw new RuntimeException("Failed to crop image: " + inputPath, e);
        }
    }
}
