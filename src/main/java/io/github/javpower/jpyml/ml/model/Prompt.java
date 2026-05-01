package io.github.javpower.jpyml.ml.model;

/**
 * Represents a prompt for interactive segmentation models like SAM 2.
 * Prompts can be points, boxes, or masks that guide the model on what to segment.
 */
public sealed interface Prompt permits Prompt.Point, Prompt.Box, Prompt.Mask, Prompt.Text {

    /**
     * A point prompt with x,y coordinates and optional label.
     */
    record Point(int x, int y, Label label) implements Prompt {
        public Point(int x, int y) {
            this(x, y, Label.POSITIVE);
        }
    }

    /**
     * A bounding box prompt with coordinates.
     */
    record Box(int x1, int y1, int x2, int y2) implements Prompt {
        public static Box of(int x1, int y1, int x2, int y2) {
            return new Box(x1, y1, x2, y2);
        }
    }

    /**
     * A mask prompt represented as a 2D boolean array.
     */
    record Mask(boolean[][] mask) implements Prompt {}

    /**
     * A text prompt for concept-level segmentation (SAM 3).
     */
    record Text(String text) implements Prompt {}

    /**
     * Label for point prompts indicating whether the point is inside or outside the object.
     */
    enum Label {
        POSITIVE(1),
        NEGATIVE(0);

        private final int value;

        Label(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }
    }

    // Static factory methods for convenience

    /**
     * Create a positive point prompt.
     */
    static Point point(int x, int y) {
        return new Point(x, y, Label.POSITIVE);
    }

    /**
     * Create a point prompt with explicit label.
     */
    static Point point(int x, int y, Label label) {
        return new Point(x, y, label);
    }

    /**
     * Create a box prompt.
     */
    static Box box(int x1, int y1, int x2, int y2) {
        return new Box(x1, y1, x2, y2);
    }

    /**
     * Create a text prompt.
     */
    static Text text(String text) {
        return new Text(text);
    }
}
