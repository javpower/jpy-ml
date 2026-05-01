package io.github.javpower.jpyml.ml.training;

public record EpochMetric(
    int epoch,
    float boxLoss,
    float clsLoss,
    float dflLoss,
    float fitness
) {
    @Override
    public String toString() {
        return String.format("Epoch[%d] box=%.4f cls=%.4f dfl=%.4f fitness=%.4f",
            epoch, boxLoss, clsLoss, dflLoss, fitness);
    }
}
