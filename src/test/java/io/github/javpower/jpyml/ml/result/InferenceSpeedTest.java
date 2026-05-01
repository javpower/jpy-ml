package io.github.javpower.jpyml.ml.result;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class InferenceSpeedTest {

    @Test
    void testConstructor() {
        InferenceSpeed speed = new InferenceSpeed(1.5f, 10.2f, 0.8f);
        assertEquals(1.5f, speed.preprocessMs());
        assertEquals(10.2f, speed.inferenceMs());
        assertEquals(0.8f, speed.postprocessMs());
    }

    @Test
    void testTotalMs() {
        InferenceSpeed speed = new InferenceSpeed(1.5f, 10.2f, 0.8f);
        assertEquals(12.5f, speed.totalMs());
    }

    @Test
    void testEquals() {
        InferenceSpeed speed1 = new InferenceSpeed(1.5f, 10.2f, 0.8f);
        InferenceSpeed speed2 = new InferenceSpeed(1.5f, 10.2f, 0.8f);
        assertEquals(speed1, speed2);
    }

    @Test
    void testHashCode() {
        InferenceSpeed speed1 = new InferenceSpeed(1.5f, 10.2f, 0.8f);
        InferenceSpeed speed2 = new InferenceSpeed(1.5f, 10.2f, 0.8f);
        assertEquals(speed1.hashCode(), speed2.hashCode());
    }

    @Test
    void testToString() {
        InferenceSpeed speed = new InferenceSpeed(1.5f, 10.2f, 0.8f);
        assertNotNull(speed.toString());
    }
}
