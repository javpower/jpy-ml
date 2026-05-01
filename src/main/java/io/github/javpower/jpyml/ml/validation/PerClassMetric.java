package io.github.javpower.jpyml.ml.validation;

public record PerClassMetric(int classId, String className, float map5095) {
    @Override
    public String toString() {
        return String.format("%s (id=%d): mAP50-95=%.4f", className, classId, map5095);
    }
}
