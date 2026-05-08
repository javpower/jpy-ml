package io.github.javpower.jpyml.ml.model;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Represents a prompt for interactive segmentation models like SAM 2.
 * Prompts can be points, boxes, or text that guide the model on what to segment.
 */
public sealed interface Prompt permits Prompt.Point, Prompt.Box, Prompt.Text {

    /**
     * Convert this prompt to a Python-compatible map for Jep transfer.
     */
    Map<String, Object> toPythonMap();

    /**
     * A point prompt with x,y coordinates and optional label.
     */
    record Point(int x, int y, Label label) implements Prompt {
        public Point(int x, int y) {
            this(x, y, Label.POSITIVE);
        }

        @Override
        public Map<String, Object> toPythonMap() {
            Map<String, Object> pm = new LinkedHashMap<>();
            pm.put("type", "point");
            pm.put("x", x);
            pm.put("y", y);
            pm.put("label", label.getValue());
            return pm;
        }
    }

    /**
     * A bounding box prompt with coordinates.
     */
    record Box(int x1, int y1, int x2, int y2) implements Prompt {
        public static Box of(int x1, int y1, int x2, int y2) {
            return new Box(x1, y1, x2, y2);
        }

        @Override
        public Map<String, Object> toPythonMap() {
            Map<String, Object> pm = new LinkedHashMap<>();
            pm.put("type", "box");
            pm.put("x1", x1);
            pm.put("y1", y1);
            pm.put("x2", x2);
            pm.put("y2", y2);
            return pm;
        }
    }

    /**
     * A text prompt for concept-level segmentation (SAM 3).
     */
    record Text(String text) implements Prompt {
        @Override
        public Map<String, Object> toPythonMap() {
            Map<String, Object> pm = new LinkedHashMap<>();
            pm.put("type", "text");
            pm.put("text", text);
            return pm;
        }
    }

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

    static Point point(int x, int y) {
        return new Point(x, y, Label.POSITIVE);
    }

    static Point point(int x, int y, Label label) {
        return new Point(x, y, label);
    }

    static Box box(int x1, int y1, int x2, int y2) {
        return new Box(x1, y1, x2, y2);
    }

    static Text text(String text) {
        return new Text(text);
    }

    /**
     * Convert an array of prompts to a list of Python-compatible maps.
     */
    static List<Map<String, Object>> toPythonList(Prompt... prompts) {
        List<Map<String, Object>> list = new ArrayList<>();
        for (Prompt p : prompts) {
            list.add(p.toPythonMap());
        }
        return list;
    }
}
