package io.github.javpower.jpyml.ml.result;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class BoundingBoxTest {

    @Test
    void testConstructor() {
        BoundingBox box = new BoundingBox(10.0f, 20.0f, 100.0f, 200.0f);
        assertEquals(10.0f, box.x1());
        assertEquals(20.0f, box.y1());
        assertEquals(100.0f, box.x2());
        assertEquals(200.0f, box.y2());
    }

    @Test
    void testWidth() {
        BoundingBox box = new BoundingBox(10.0f, 20.0f, 100.0f, 200.0f);
        assertEquals(90.0f, box.width());
    }

    @Test
    void testHeight() {
        BoundingBox box = new BoundingBox(10.0f, 20.0f, 100.0f, 200.0f);
        assertEquals(180.0f, box.height());
    }

    @Test
    void testCenterX() {
        BoundingBox box = new BoundingBox(10.0f, 20.0f, 100.0f, 200.0f);
        assertEquals(55.0f, box.centerX());
    }

    @Test
    void testCenterY() {
        BoundingBox box = new BoundingBox(10.0f, 20.0f, 100.0f, 200.0f);
        assertEquals(110.0f, box.centerY());
    }

    @Test
    void testArea() {
        BoundingBox box = new BoundingBox(10.0f, 20.0f, 100.0f, 200.0f);
        assertEquals(90.0f * 180.0f, box.area());
    }

    @Test
    void testToArray() {
        BoundingBox box = new BoundingBox(10.0f, 20.0f, 100.0f, 200.0f);
        float[] arr = box.toArray();
        assertArrayEquals(new float[]{10.0f, 20.0f, 100.0f, 200.0f}, arr);
    }

    @Test
    void testEquals() {
        BoundingBox box1 = new BoundingBox(10.0f, 20.0f, 100.0f, 200.0f);
        BoundingBox box2 = new BoundingBox(10.0f, 20.0f, 100.0f, 200.0f);
        assertEquals(box1, box2);
    }

    @Test
    void testHashCode() {
        BoundingBox box1 = new BoundingBox(10.0f, 20.0f, 100.0f, 200.0f);
        BoundingBox box2 = new BoundingBox(10.0f, 20.0f, 100.0f, 200.0f);
        assertEquals(box1.hashCode(), box2.hashCode());
    }

    @Test
    void testToString() {
        BoundingBox box = new BoundingBox(10.0f, 20.0f, 100.0f, 200.0f);
        assertNotNull(box.toString());
    }
}
