package io.github.javpower.jpyml.ml.result;

public record OBBPrediction(RotatedBoundingBox box, float confidence, int classId, String className) {}
