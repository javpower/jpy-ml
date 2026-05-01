package io.github.javpower.jpyml.ml.validation;

import java.util.List;
import java.util.Map;

public class ValidationResult {
    private final float map50;
    private final float map5095;
    private final float precision;
    private final float recall;
    private final Map<String, Object> speed;
    private final List<PerClassMetric> perClassMetrics;

    public ValidationResult(float map50, float map5095, float precision, float recall) {
        this(map50, map5095, precision, recall, null, List.of());
    }

    public ValidationResult(float map50, float map5095, float precision, float recall,
                            Map<String, Object> speed, List<PerClassMetric> perClassMetrics) {
        this.map50 = map50;
        this.map5095 = map5095;
        this.precision = precision;
        this.recall = recall;
        this.speed = speed;
        this.perClassMetrics = perClassMetrics != null ? perClassMetrics : List.of();
    }

    public float getMAP50() { return map50; }
    public float getMAP5095() { return map5095; }
    public float getPrecision() { return precision; }
    public float getRecall() { return recall; }
    public Map<String, Object> getSpeed() { return speed; }
    public List<PerClassMetric> getPerClassMetrics() { return perClassMetrics; }

    @Override
    public String toString() {
        return String.format("Validation{mAP50=%.3f, mAP50-95=%.3f, P=%.3f, R=%.3f, classes=%d}",
                map50, map5095, precision, recall, perClassMetrics.size());
    }
}
