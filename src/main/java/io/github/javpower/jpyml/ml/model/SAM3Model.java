package io.github.javpower.jpyml.ml.model;

import io.github.javpower.jpyml.core.PythonEngine;
import io.github.javpower.jpyml.core.PythonScriptLoader;
import io.github.javpower.jpyml.exception.InferenceException;
import io.github.javpower.jpyml.exception.ModelException;
import io.github.javpower.jpyml.ml.result.*;
import jep.JepException;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * SAM 3 (Segment Anything Model 3) for concept-level segmentation.
 * <p>
 * SAM 3 enables segmentation using natural language text prompts or image exemplars,
 * allowing you to describe what you want segmented rather than manually pointing at it.
 * <p>
 * Usage:
 * <pre>
 *   try (SAM3Model sam = new SAM3Model("sam3.pt")) {
 *       // Text-based segmentation
 *       SAM3Result result = sam.predictText("street.jpg", "person", "car");
 *
 *       // Exemplar-based segmentation
 *       SAM3Result result = sam.predictExemplar("target.jpg", "reference.jpg",
 *           BoundingBox.of(10, 20, 200, 300));
 *   }
 * </pre>
 *
 * @see Prompt.Text
 * @see SAM3Result
 */
public class SAM3Model implements AutoCloseable {

    private static final AtomicLong idCounter = new AtomicLong(0);

    private final long id;
    private final String varName;
    private final String modelPath;
    private final PythonEngine engine;
    private boolean closed = false;

    /**
     * Load a SAM 3 model.
     *
     * @param modelPath path to the SAM 3 model file
     * @throws ModelException if the model fails to load
     */
    public SAM3Model(String modelPath) throws ModelException {
        this.id = idCounter.getAndIncrement();
        this.varName = "_jpy_sam3_" + this.id;
        this.modelPath = modelPath;

        try {
            engine = PythonEngine.getInstance();
            PythonScriptLoader.ensureLoaded(engine, "_jpy_init.py");
            PythonScriptLoader.ensureLoaded(engine, "_jpy_sam3.py");

            engine.put(varName + "_path", modelPath);
            engine.put(varName + "_name", varName);
            engine.exec(varName + " = jpy_sam3_load(" + varName + "_path, " + varName + "_name)");
        } catch (JepException e) {
            throw new ModelException("Failed to load SAM 3 model: " + modelPath, e);
        }
    }

    /**
     * Predict segmentation using text prompts.
     * <p>
     * Example: {@code sam.predictText("street.jpg", "person", "car")}
     *
     * @param imagePath    path to the image file
     * @param textPrompts  text descriptions of what to segment
     * @return segmentation result containing masks and scores
     * @throws InferenceException if prediction fails
     */
    public SAM3Result predictText(String imagePath, String... textPrompts) throws InferenceException {
        ensureOpen();
        try {
            String rv = varName + "_result";
            List<String> prompts = Arrays.asList(textPrompts);

            engine.put(varName + "_img", imagePath);
            engine.put(varName + "_texts", prompts);
            engine.put(varName + "_name", varName);
            engine.exec(rv + " = jpy_sam3_predict_text(" + varName + "_name, " +
                    varName + "_img, " + varName + "_texts)");

            @SuppressWarnings("unchecked")
            Map<String, Object> result = engine.eval(rv);
            return buildSAM3Result(result);

        } catch (Exception e) {
            throw new InferenceException("SAM 3 text prediction failed on: " + imagePath, e);
        }
    }

    /**
     * Predict segmentation using an image exemplar.
     * <p>
     * Finds objects in the target image that are similar to the exemplar region.
     *
     * @param imagePath     path to the target image
     * @param exemplarPath  path to the exemplar image
     * @param exemplarBox   bounding box in the exemplar image defining the reference object
     * @return segmentation result containing masks and scores
     * @throws InferenceException if prediction fails
     */
    public SAM3Result predictExemplar(String imagePath, String exemplarPath, BoundingBox exemplarBox)
            throws InferenceException {
        ensureOpen();
        try {
            String rv = varName + "_result";

            engine.put(varName + "_img", imagePath);
            engine.put(varName + "_exemplar", exemplarPath);
            engine.put(varName + "_box", new double[]{
                    exemplarBox.x1(), exemplarBox.y1(), exemplarBox.x2(), exemplarBox.y2()
            });
            engine.put(varName + "_name", varName);
            engine.exec(rv + " = jpy_sam3_predict_exemplar(" + varName + "_name, " +
                    varName + "_img, " + varName + "_exemplar, " + varName + "_box)");

            @SuppressWarnings("unchecked")
            Map<String, Object> result = engine.eval(rv);
            return buildSAM3Result(result);

        } catch (Exception e) {
            throw new InferenceException("SAM 3 exemplar prediction failed on: " + imagePath, e);
        }
    }

    private SAM3Result buildSAM3Result(Map<String, Object> data) {
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> rawMasks = (List<Map<String, Object>>) data.getOrDefault("masks", List.of());

        List<Mask> masks = new ArrayList<>();
        List<Float> scores = new ArrayList<>();
        List<Integer> classIds = new ArrayList<>();

        for (Map<String, Object> rm : rawMasks) {
            @SuppressWarnings("unchecked")
            List<List<Number>> polygon = (List<List<Number>>) rm.getOrDefault("polygon", List.of());
            float[][] polyArray = new float[polygon.size()][2];
            for (int i = 0; i < polygon.size(); i++) {
                polyArray[i][0] = polygon.get(i).get(0).floatValue();
                polyArray[i][1] = polygon.get(i).get(1).floatValue();
            }
            masks.add(new Mask(polyArray));
            scores.add(((Number) rm.getOrDefault("score", 0)).floatValue());
            classIds.add(((Number) rm.getOrDefault("class_id", 0)).intValue());
        }

        return new SAM3Result(
                (String) data.getOrDefault("path", ""),
                masks,
                scores,
                classIds
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
            throw new IllegalStateException("SAM3Model is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            try {
                engine.exec(varName + " = None");
            } catch (Exception ignored) {
            }
        }
    }
}
