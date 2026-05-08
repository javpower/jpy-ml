package io.github.javpower.jpyml.ml.model;

import io.github.javpower.jpyml.core.DependencyManager;
import io.github.javpower.jpyml.core.PythonEngine;
import io.github.javpower.jpyml.core.PythonScriptLoader;
import io.github.javpower.jpyml.exception.InferenceException;
import io.github.javpower.jpyml.exception.ModelException;
import io.github.javpower.jpyml.ml.result.*;
import jep.JepException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * SAM 2 (Segment Anything Model 2) for interactive promptable segmentation.
 * <p>
 * SAM 2 segments objects based on interactive prompts (points, boxes, masks)
 * rather than automatic detection like YOLO.
 * <p>
 * Usage:
 * <pre>
 *   try (SAM2Model sam = new SAM2Model("sam2_b.pt")) {
 *       SAM2Result result = sam.predict("photo.jpg",
 *           Prompt.point(100, 200),
 *           Prompt.box(50, 50, 300, 400)
 *       );
 *       List&lt;Mask&gt; masks = result.getMasks();
 *   }
 * </pre>
 *
 * @see Prompt
 * @see SAM2Result
 */
public class SAM2Model implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(SAM2Model.class);
    private static final AtomicLong idCounter = new AtomicLong(0);

    private final long id;
    private final String varName;
    private final String modelPath;
    private final PythonEngine engine;
    private volatile boolean closed = false;

    /**
     * Load a SAM 2 model.
     *
     * @param modelPath path to the SAM 2 model file (e.g., "sam2_b.pt")
     * @throws ModelException if the model fails to load
     */
    public SAM2Model(String modelPath) throws ModelException {
        this.id = idCounter.getAndIncrement();
        this.varName = "_jpy_sam2_" + this.id;
        this.modelPath = modelPath;

        try {
            DependencyManager.ensure("cv");
            engine = PythonEngine.getInstance();
            PythonScriptLoader.ensureLoaded(engine, "_jpy_init.py");
            PythonScriptLoader.ensureLoaded(engine, "_jpy_sam2.py");

            engine.put(varName + "_path", modelPath);
            engine.put(varName + "_name", varName);
            engine.exec(varName + " = jpy_sam2_load(" + varName + "_path, " + varName + "_name)");
        } catch (JepException e) {
            throw new ModelException("Failed to load SAM 2 model: " + modelPath, e);
        }
    }

    /**
     * Predict segmentation masks for the given image with the specified prompts.
     *
     * @param imagePath path to the image file
     * @param prompts   one or more prompts (points, boxes, masks)
     * @return segmentation result containing masks and scores
     * @throws InferenceException if prediction fails
     */
    public SAM2Result predict(String imagePath, Prompt... prompts) throws InferenceException {
        ensureOpen();
        try {
            String rv = varName + "_result";

            // Build prompts list in Python
            List<Object> promptList = new ArrayList<>(Prompt.toPythonList(prompts));

            engine.put(varName + "_prompts", promptList);
            engine.put(varName + "_img", imagePath);
            engine.put(varName + "_name", varName);

            engine.exec(varName + "_result = jpy_sam2_predict(" + varName + "_name, " + varName + "_img, " + varName + "_prompts)");
            @SuppressWarnings("unchecked")
            Map<String, Object> result = engine.eval(varName + "_result");
            return buildSAM2Result(result);

        } catch (Exception e) {
            throw new InferenceException("SAM 2 prediction failed on: " + imagePath, e);
        }
    }

    /**
     * Start video tracking with the given prompts.
     *
     * @param videoPath path to the video file
     * @param prompts   prompts for the first frame
     * @return video tracker object
     * @throws InferenceException if initialization fails
     */
    public SAM2VideoTracker trackVideo(String videoPath, Prompt... prompts) throws InferenceException {
        ensureOpen();
        try {
            PythonScriptLoader.ensureLoaded(engine, "_jpy_sam2_video.py");

            List<Object> promptList = new ArrayList<>(Prompt.toPythonList(prompts));

            engine.put(varName + "_video_path", videoPath);
            engine.put(varName + "_video_prompts", promptList);
            engine.exec(varName + "_video_model = _jpy_sam2_models['" + varName + "']");
            engine.exec(varName + "_tracker = jpy_sam2_video_start(" + varName + "_video_model, " +
                    varName + "_video_path, " + varName + "_video_prompts)");

            return new SAM2VideoTracker(engine, varName + "_tracker");

        } catch (Exception e) {
            throw new InferenceException("Failed to start SAM 2 video tracking", e);
        }
    }

    private SAM2Result buildSAM2Result(Map<String, Object> data) {
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> rawMasks = (List<Map<String, Object>>) data.getOrDefault("masks", List.of());

        List<Mask> masks = new ArrayList<>();
        List<Float> scores = new ArrayList<>();

        for (Map<String, Object> rm : rawMasks) {
            @SuppressWarnings("unchecked")
            List<List<Number>> polygon = (List<List<Number>>) rm.getOrDefault("polygon", List.of());
            masks.add(new Mask(ResultParseUtil.parsePolygon(polygon)));
            scores.add(((Number) rm.getOrDefault("score", 0)).floatValue());
        }

        return new SAM2Result(
                (String) data.getOrDefault("path", ""),
                masks,
                scores
        );
    }

    public String getModelPath() {
        return modelPath;
    }

    public boolean isClosed() {
        return closed;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("SAM2Model is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            try {
                engine.exec("if '" + varName + "' in _jpy_sam2_models: del _jpy_sam2_models['" + varName + "']");
            } catch (Exception e) {
                log.debug("Error cleaning SAM2 model dict: {}", e.getMessage());
            }
            try {
                engine.exec(varName + " = None");
            } catch (Exception e) {
                log.debug("Error cleaning SAM2 variable: {}", e.getMessage());
            }
        }
    }
}
