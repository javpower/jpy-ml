package io.github.javpower.jpyml.ml.training;
@FunctionalInterface
public interface TrainingCallback {
    void onEpoch(int epoch, String logLine);
}
