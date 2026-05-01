package io.github.javpower.jpyml.ml.result;

/**
 * A single frame from video/webcam stream inference.
 * Contains both the inference result and the annotated image (JPEG bytes).
 */
public class StreamFrame {

    private final InferenceResult result;
    private final byte[] annotatedImage;
    private final int frameIndex;

    public StreamFrame(InferenceResult result, byte[] annotatedImage, int frameIndex) {
        this.result = result;
        this.annotatedImage = annotatedImage;
        this.frameIndex = frameIndex;
    }

    public InferenceResult getResult() {
        return result;
    }

    /**
     * Returns the annotated image as JPEG bytes (boxes/masks drawn by Ultralytics).
     * Can be directly written to a file, served via HTTP, or decoded for display.
     */
    public byte[] getAnnotatedImage() {
        return annotatedImage;
    }

    public int getFrameIndex() {
        return frameIndex;
    }

    public boolean hasImage() {
        return annotatedImage != null && annotatedImage.length > 0;
    }
}
