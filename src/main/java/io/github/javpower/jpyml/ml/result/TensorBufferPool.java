package io.github.javpower.jpyml.ml.result;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Pool of DirectByteBuffer instances for zero-copy data transfer between Python and Java.
 * <p>
 * DirectByteBuffer is not garbage collected, so we pool and reuse them to avoid
 * memory leaks and reduce allocation overhead.
 * <p>
 * Usage:
 * <pre>
 *   TensorBufferPool pool = TensorBufferPool.getInstance();
 *   FloatBuffer buf = pool.acquireFloatBuffer(1024);
 *   try {
 *       // use buffer
 *   } finally {
 *       pool.release(buf);
 *   }
 * </pre>
 */
public class TensorBufferPool {

    private static final int DEFAULT_FLOAT_CAPACITY = 4096;  // 4K floats
    private static final int DEFAULT_INT_CAPACITY = 1024;    // 1K ints

    private final Queue<FloatBuffer> floatPool = new ConcurrentLinkedQueue<>();
    private final Queue<IntBuffer> intPool = new ConcurrentLinkedQueue<>();

    private static final TensorBufferPool INSTANCE = new TensorBufferPool();

    public static TensorBufferPool getInstance() {
        return INSTANCE;
    }

    private TensorBufferPool() {
    }

    /**
     * Acquire a FloatBuffer with at least the specified capacity.
     * The buffer is cleared and ready for writing.
     *
     * @param minCapacity minimum number of floats the buffer must hold
     * @return a direct FloatBuffer
     */
    public FloatBuffer acquireFloatBuffer(int minCapacity) {
        FloatBuffer buf = floatPool.poll();
        while (buf != null) {
            if (buf.capacity() >= minCapacity) {
                buf.clear();
                buf.limit(minCapacity);
                return buf;
            }
            buf = floatPool.poll();
        }
        // Allocate new buffer with some headroom
        int capacity = Math.max(minCapacity, DEFAULT_FLOAT_CAPACITY);
        buf = ByteBuffer.allocateDirect(capacity * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        buf.limit(minCapacity);
        return buf;
    }

    /**
     * Acquire an IntBuffer with at least the specified capacity.
     * The buffer is cleared and ready for writing.
     *
     * @param minCapacity minimum number of ints the buffer must hold
     * @return a direct IntBuffer
     */
    public IntBuffer acquireIntBuffer(int minCapacity) {
        IntBuffer buf = intPool.poll();
        while (buf != null) {
            if (buf.capacity() >= minCapacity) {
                buf.clear();
                buf.limit(minCapacity);
                return buf;
            }
            buf = intPool.poll();
        }
        // Allocate new buffer with some headroom
        int capacity = Math.max(minCapacity, DEFAULT_INT_CAPACITY);
        buf = ByteBuffer.allocateDirect(capacity * 4)
                .order(ByteOrder.nativeOrder())
                .asIntBuffer();
        buf.limit(minCapacity);
        return buf;
    }

    /**
     * Release a FloatBuffer back to the pool for reuse.
     *
     * @param buf the buffer to release, may be null
     */
    public void release(FloatBuffer buf) {
        if (buf != null) {
            floatPool.offer(buf);
        }
    }

    /**
     * Release an IntBuffer back to the pool for reuse.
     *
     * @param buf the buffer to release, may be null
     */
    public void release(IntBuffer buf) {
        if (buf != null) {
            intPool.offer(buf);
        }
    }

    /**
     * Clear all pooled buffers. Useful for testing or memory pressure scenarios.
     */
    public void clear() {
        floatPool.clear();
        intPool.clear();
    }

    /**
     * Get the number of FloatBuffers currently in the pool.
     */
    public int getFloatPoolSize() {
        return floatPool.size();
    }

    /**
     * Get the number of IntBuffers currently in the pool.
     */
    public int getIntPoolSize() {
        return intPool.size();
    }
}
