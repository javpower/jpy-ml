package io.github.javpower.jpyml.ml.result;
public record ClassPrediction(BoundingBox box, float confidence, int classId, String className) {
    @Override public String toString() {
        return String.format("%s %.1f%% %s", className, confidence * 100, box);
    }
}
