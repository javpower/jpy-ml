package io.github.javpower.jpyml.ml.model;

import io.github.javpower.jpyml.core.PythonEngine;
import io.github.javpower.jpyml.exception.InferenceException;
import io.github.javpower.jpyml.ml.result.Mask;
import io.github.javpower.jpyml.ml.result.ResultParseUtil;
import io.github.javpower.jpyml.ml.result.SAM2VideoResult;
import jep.JepException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

    private static final Logger log = LoggerFactory.getLogger(SAM2VideoTracker.class);
    private final PythonEngine engine;
    private final String trackerVar;
    private volatile boolean closed = false;

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
            Map<String, Object> pm = prompt.toPythonMap();

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
                float score = ((Number) fr.getOrDefault("score", 0)).floatValue();
                frames.add(new SAM2VideoResult.FrameResult(frameIdx, new Mask(ResultParseUtil.parsePolygon(polygon)), score));
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
            } catch (Exception e) {
                log.debug("Error cleaning tracker variable: {}", e.getMessage());
            }
        }
    }
}
