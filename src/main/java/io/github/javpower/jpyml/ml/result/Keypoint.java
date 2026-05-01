package io.github.javpower.jpyml.ml.result;
public record Keypoint(float x, float y, float confidence) {
    public boolean isVisible(float threshold) { return confidence > threshold; }
}
