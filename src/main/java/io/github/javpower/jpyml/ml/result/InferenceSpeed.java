package io.github.javpower.jpyml.ml.result;
public record InferenceSpeed(float preprocessMs, float inferenceMs, float postprocessMs) {
    public float totalMs() { return preprocessMs + inferenceMs + postprocessMs; }
}
