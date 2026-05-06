package io.github.javpower.jpyml.ml.result;
public record BoundingBox(float x1, float y1, float x2, float y2) {
    public BoundingBox {
        if (x2 < x1 || y2 < y1) {
            throw new IllegalArgumentException(
                    "Invalid bounding box: requires x2>=x1 and y2>=y1, got (" +
                    x1 + "," + y1 + ")-(" + x2 + "," + y2 + ")");
        }
    }
    public float width() { return x2 - x1; }
    public float height() { return y2 - y1; }
    public float centerX() { return (x1 + x2) / 2; }
    public float centerY() { return (y1 + y2) / 2; }
    public float area() { return width() * height(); }
    public float[] toArray() { return new float[]{x1, y1, x2, y2}; }
}
