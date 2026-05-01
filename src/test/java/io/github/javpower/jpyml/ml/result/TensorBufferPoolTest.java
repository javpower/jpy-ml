package io.github.javpower.jpyml.ml.result;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import static org.junit.jupiter.api.Assertions.*;

class TensorBufferPoolTest {

    private TensorBufferPool pool;

    @BeforeEach
    void setUp() {
        pool = TensorBufferPool.getInstance();
        pool.clear();
    }

    @Test
    void testAcquireFloatBuffer() {
        FloatBuffer buf = pool.acquireFloatBuffer(100);
        assertNotNull(buf);
        assertTrue(buf.isDirect());
        assertEquals(100, buf.limit());
        assertEquals(100, buf.remaining());
    }

    @Test
    void testAcquireIntBuffer() {
        IntBuffer buf = pool.acquireIntBuffer(50);
        assertNotNull(buf);
        assertTrue(buf.isDirect());
        assertEquals(50, buf.limit());
        assertEquals(50, buf.remaining());
    }

    @Test
    void testBufferReuse() {
        FloatBuffer buf1 = pool.acquireFloatBuffer(100);
        pool.release(buf1);
        FloatBuffer buf2 = pool.acquireFloatBuffer(100);
        assertSame(buf1, buf2);
    }

    @Test
    void testBufferReallocationForLargerSize() {
        FloatBuffer buf1 = pool.acquireFloatBuffer(100);
        pool.release(buf1);
        // Request a size larger than default capacity (4096)
        FloatBuffer buf2 = pool.acquireFloatBuffer(5000);
        assertNotSame(buf1, buf2);
    }

    @Test
    void testClear() {
        pool.acquireFloatBuffer(100);
        pool.acquireIntBuffer(50);
        assertEquals(0, pool.getFloatPoolSize());
        assertEquals(0, pool.getIntPoolSize());
    }

    @Test
    void testPoolSize() {
        FloatBuffer buf = pool.acquireFloatBuffer(100);
        pool.release(buf);
        assertEquals(1, pool.getFloatPoolSize());
    }
}
