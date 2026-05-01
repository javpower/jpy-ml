package io.github.javpower.jpyml.ml.result;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class KeypointTest {

    @Test
    void testConstructor() {
        Keypoint kpt = new Keypoint(100.0f, 200.0f, 0.95f);
        assertEquals(100.0f, kpt.x());
        assertEquals(200.0f, kpt.y());
        assertEquals(0.95f, kpt.confidence());
    }

    @Test
    void testIsVisible() {
        Keypoint visible = new Keypoint(100.0f, 200.0f, 0.95f);
        assertTrue(visible.isVisible(0.5f));

        Keypoint invisible = new Keypoint(100.0f, 200.0f, 0.1f);
        assertFalse(invisible.isVisible(0.5f));
    }

    @Test
    void testEquals() {
        Keypoint kpt1 = new Keypoint(100.0f, 200.0f, 0.95f);
        Keypoint kpt2 = new Keypoint(100.0f, 200.0f, 0.95f);
        assertEquals(kpt1, kpt2);
    }

    @Test
    void testHashCode() {
        Keypoint kpt1 = new Keypoint(100.0f, 200.0f, 0.95f);
        Keypoint kpt2 = new Keypoint(100.0f, 200.0f, 0.95f);
        assertEquals(kpt1.hashCode(), kpt2.hashCode());
    }

    @Test
    void testToString() {
        Keypoint kpt = new Keypoint(100.0f, 200.0f, 0.95f);
        assertNotNull(kpt.toString());
    }
}
