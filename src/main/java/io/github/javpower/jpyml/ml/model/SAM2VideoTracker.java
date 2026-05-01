package io.github.javpower.jpyml.ml.model;

import io.github.javpower.jpyml.core.PythonEngine;
import io.github.javpower.jpyml.exception.InferenceException;
import io.github.javpower.jpyml.ml.result.Mask;
import io.github.javpower.jpyml.ml.result.SAM2VideoResult;
import jep.JepException;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * SAM 2 video tracker for object tracking across video frames.
 * <p>
 * Usage:
 * <pre>
 *   SAM2VideoTracker tracker = sam.trackVideo("video.mp4", Prompt.box(50, 50, 300, 400));
 *   tracker.addPrompt(0, Prompt.point(200, 200));
 *   SAM2VideoResult result = tracker.propagate();
 * </pre>
 */
public class SAM2VideoTracker implements AutoCloseable {

    private final PythonEngine engine;
    private final String trackerVar;
    private boolean closed = false;

    SAM2VideoTracker(PythonEngine engine, String trackerVar) {
        this.engine = engine;
        this.trackerVar = trackerVar;
    }

    /**
     * Add a prompt to a specific frame.
     *
     * @param frameIndex frame index (0-based)
     * @param prompt     the prompt to add
     * @throws InferenceException if adding the prompt fails
     */
    public void addPrompt(int frameIndex, Prompt prompt) throws InferenceException {
        ensureOpen();
        try {
            Map<String, Object> pm = new LinkedHashMap<>();
            if (prompt instanceof Prompt.Point point) {
                pm.put("type", "point");
                pm.put("x", point.x());
                pm.put("y", point.y());
                pm.put("label", point.label().getValue());
            } else if (prompt instanceof Prompt.Box box) {
                pm.put("type", "box");
                pm.put("x1", box.x1());
                pm.put("y1", box.y1());
                pm.put("x2", box.x2());
                pm.put("y2", box.y2());
            }

            engine.put(trackerVar + "_prompt", pm);
            engine.put(trackerVar + "_frame", frameIndex);
            engine.exec("jpy_sam2_video_add_prompt(" + trackerVar + ", " + trackerVar + "_frame, " + trackerVar + "_prompt)");
        } catch (JepException e) {
            throw new InferenceException("Failed to add prompt to frame " + frameIndex, e);
        }
    }

    /**
     * Propagate prompts through all frames and return tracking results.
     *
     * @return video tracking results for all frames
     * @throws InferenceException if propagation fails
     */
    public SAM2VideoResult propagate() throws InferenceException {
        ensureOpen();
        try {
            engine.exec(trackerVar + "_result = jpy_sam2_video_propagate(" + trackerVar + ")");

            @SuppressWarnings("unchecked")
            Map<String, Object> result = engine.eval(trackerVar + "_result");

            int totalFrames = ((Number) result.getOrDefault("total_frames", 0)).intValue();
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> frameResults = (List<Map<String, Object>>) result.getOrDefault("frames", List.of());

            List<SAM2VideoResult.FrameResult> frames = new ArrayList<>();
            for (Map<String, Object> fr : frameResults) {
                int frameIdx = ((Number) fr.getOrDefault("frame_index", 0)).intValue();
                @SuppressWarnings("unchecked")
                List<List<Number>> polygon = (List<List<Number>>) fr.getOrDefault("polygon", List.of());
                float[][] polyArray = new float[polygon.size()][2];
                for (int i = 0; i < polygon.size(); i++) {
                    polyArray[i][0] = polygon.get(i).get(0).floatValue();
                    polyArray[i][1] = polygon.get(i).get(1).floatValue();
                }
                float score = ((Number) fr.getOrDefault("score", 0)).floatValue();
                frames.add(new SAM2VideoResult.FrameResult(frameIdx, new Mask(polyArray), score));
            }

            return new SAM2VideoResult(totalFrames, frames);

        } catch (Exception e) {
            throw new InferenceException("Failed to propagate video tracking", e);
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("SAM2VideoTracker is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            try {
                engine.exec(trackerVar + " = None");
            } catch (Exception ignored) {
            }
        }
    }
}
