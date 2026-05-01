package io.github.javpower.jpyml.ml.result;
public class Mask {
    private final float[][] polygon;
    public Mask(float[][] polygon) { this.polygon = polygon; }
    public float[][] getPolygon() { return polygon; }
    public int getPointCount() { return polygon.length; }
}
