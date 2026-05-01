package io.github.javpower.jpyml.ml.result;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

/**
 * Base interface for raw (zero-copy) inference results.
 * Provides access to direct buffers for high-performance data processing.
 */
public interface RawInferenceResult extends InferenceResult {

    /**
     * Get the number of detected items (boxes, keypoints, etc.).
     *
     * @return count of primary items
     */
    int getBoxCount();

    /**
     * Release all buffers back to the pool for reuse.
     * After calling this method, the result should not be used.
     */
    void release();
}
