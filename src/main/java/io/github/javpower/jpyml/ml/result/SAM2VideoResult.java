package io.github.javpower.jpyml.ml.result;

import java.util.Collections;
import java.util.List;

/**
 * Result of SAM 2 video tracking.
 * Contains tracking results for all frames.
 *
 * @param totalFrames total number of frames processed
 * @param frames      per-frame tracking results
 */
public record SAM2VideoResult(
        int totalFrames,
        List<FrameResult> frames
) {
    public SAM2VideoResult {
        frames = Collections.unmodifiableList(frames);
    }

    /**
     * Get tracking result for a specific frame.
     *
     * @param frameIndex frame index (0-based)
     * @return frame result, or null if not available
     */
    public FrameResult getFrame(int frameIndex) {
        return frames.stream()
                .filter(f -> f.frameIndex() == frameIndex)
                .findFirst()
                .orElse(null);
    }

    /**
     * Get the number of frames with tracking results.
     */
    public int trackedFrameCount() {
        return frames.size();
    }

    /**
     * Per-frame tracking result.
     *
     * @param frameIndex frame index
     * @param mask       tracked object mask
     * @param score      confidence score
     */
    public record FrameResult(int frameIndex, Mask mask, float score) {}
}
