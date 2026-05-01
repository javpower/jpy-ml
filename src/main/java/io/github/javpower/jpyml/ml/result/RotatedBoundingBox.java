package io.github.javpower.jpyml.ml.result;
public record RotatedBoundingBox(float centerX, float centerY, float width, float height, float angleDegrees) {
    public float[] cornerPoints() {
        double rad = Math.toRadians(angleDegrees);
        double cos = Math.cos(rad), sin = Math.sin(rad);
        float hw = width / 2, hh = height / 2;
        float[] corners = new float[8];
        float[][] offsets = {{-hw,-hh},{hw,-hh},{hw,hh},{-hw,hh}};
        for (int i = 0; i < 4; i++) {
            corners[i*2] = (float)(centerX + offsets[i][0]*cos - offsets[i][1]*sin);
            corners[i*2+1] = (float)(centerY + offsets[i][0]*sin + offsets[i][1]*cos);
        }
        return corners;
    }
}
