package io.github.javpower.jpyml.ml.result;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

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
    private static final int MAX_POOL_SIZE = 16;

    private final Queue<FloatBuffer> floatPool = new ConcurrentLinkedQueue<>();
    private final Queue<IntBuffer> intPool = new ConcurrentLinkedQueue<>();
    private final AtomicInteger floatPoolSize = new AtomicInteger(0);
    private final AtomicInteger intPoolSize = new AtomicInteger(0);

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
        FloatBuffer best = null;
        FloatBuffer buf = floatPool.poll();
        while (buf != null) {
            floatPoolSize.decrementAndGet();
            if (buf.capacity() >= minCapacity) {
                if (best == null || buf.capacity() < best.capacity()) {
                    if (best != null) {
                        floatPool.offer(best);
                        floatPoolSize.incrementAndGet();
                    }
                    best = buf;
                } else {
                    floatPool.offer(buf);
                    floatPoolSize.incrementAndGet();
                }
            }
            // too-small buffers are discarded (DirectByteBuffer cannot be freed explicitly)
            buf = floatPool.poll();
        }
        if (best != null) {
            best.clear();
            best.limit(minCapacity);
            return best;
        }
        // Allocate new buffer with some headroom
        int capacity = Math.max(minCapacity, DEFAULT_FLOAT_CAPACITY);
        best = ByteBuffer.allocateDirect(capacity * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        best.limit(minCapacity);
        return best;
    }

    /**
     * Acquire an IntBuffer with at least the specified capacity.
     * The buffer is cleared and ready for writing.
     *
     * @param minCapacity minimum number of ints the buffer must hold
     * @return a direct IntBuffer
     */
    public IntBuffer acquireIntBuffer(int minCapacity) {
        IntBuffer best = null;
        IntBuffer buf = intPool.poll();
        while (buf != null) {
            intPoolSize.decrementAndGet();
            if (buf.capacity() >= minCapacity) {
                if (best == null || buf.capacity() < best.capacity()) {
                    if (best != null) {
                        intPool.offer(best);
                        intPoolSize.incrementAndGet();
                    }
                    best = buf;
                } else {
                    intPool.offer(buf);
                    intPoolSize.incrementAndGet();
                }
            }
            // too-small buffers are discarded (DirectByteBuffer cannot be freed explicitly)
            buf = intPool.poll();
        }
        if (best != null) {
            best.clear();
            best.limit(minCapacity);
            return best;
        }
        // Allocate new buffer with some headroom
        int capacity = Math.max(minCapacity, DEFAULT_INT_CAPACITY);
        best = ByteBuffer.allocateDirect(capacity * 4)
                .order(ByteOrder.nativeOrder())
                .asIntBuffer();
        best.limit(minCapacity);
        return best;
    }

    /**
     * Release a FloatBuffer back to the pool for reuse.
     *
     * @param buf the buffer to release, may be null
     */
    public void release(FloatBuffer buf) {
        if (buf != null && floatPoolSize.get() < MAX_POOL_SIZE) {
            floatPool.offer(buf);
            floatPoolSize.incrementAndGet();
        }
    }

    /**
     * Release an IntBuffer back to the pool for reuse.
     *
     * @param buf the buffer to release, may be null
     */
    public void release(IntBuffer buf) {
        if (buf != null && intPoolSize.get() < MAX_POOL_SIZE) {
            intPool.offer(buf);
            intPoolSize.incrementAndGet();
        }
    }

    /**
     * Clear all pooled buffers. Useful for testing or memory pressure scenarios.
     */
    public void clear() {
        floatPool.clear();
        intPool.clear();
        floatPoolSize.set(0);
        intPoolSize.set(0);
    }

    /**
     * Get the number of FloatBuffers currently in the pool.
     */
    public int getFloatPoolSize() {
        return floatPoolSize.get();
    }

    /**
     * Get the number of IntBuffers currently in the pool.
     */
    public int getIntPoolSize() {
        return intPoolSize.get();
    }
}
