package io.github.javpower.jpyml.ml.result;

import io.github.javpower.jpyml.ml.model.TaskType;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Zero-copy detection result that holds direct references to native buffers.
 * <p>
 * This class provides two ways to access detection results:
 * <ol>
 *   <li>{@link #getBoxes()} - Lazy-loaded strongly-typed list (convenient)</li>
 *   <li>{@link #getRawBoxesXYXY()}, {@link #getRawConfidences()}, {@link #getRawClassIds()} - Raw buffer access (high-performance)</li>
 * </ol>
 * <p>
 * The raw buffer access avoids object allocation and is suitable for high-throughput
 * scenarios where you need to process detection results without creating Java objects.
 */
public class RawDetectionResult implements RawInferenceResult {

    private final String sourcePath;
    private final int origWidth, origHeight;
    private final InferenceSpeed speed;
    private final Map<Integer, String> classNames;

    // Zero-copy buffers (directly from Python via DirectNDArray)
    private final FloatBuffer boxesXYXY;  // shape: (N, 4) - x1, y1, x2, y2 per box
    private final FloatBuffer confidences; // shape: (N,) - confidence per box
    private final IntBuffer classIds;      // shape: (N,) - class id per box
    private final int boxCount;

    // Lazy-loaded strongly-typed list
    private List<ClassPrediction> boxesList;

    // Buffer pool reference for cleanup
    private final TensorBufferPool bufferPool;

    public RawDetectionResult(String sourcePath, int origWidth, int origHeight,
                              InferenceSpeed speed, Map<Integer, String> classNames,
                              FloatBuffer boxesXYXY, FloatBuffer confidences,
                              IntBuffer classIds, int boxCount) {
        this.sourcePath = sourcePath;
        this.origWidth = origWidth;
        this.origHeight = origHeight;
        this.speed = speed;
        this.classNames = classNames;
        this.boxesXYXY = boxesXYXY;
        this.confidences = confidences;
        this.classIds = classIds;
        this.boxCount = boxCount;
        this.bufferPool = TensorBufferPool.getInstance();
    }

    @Override
    public String getSourcePath() {
        return sourcePath;
    }

    @Override
    public int getOriginalWidth() {
        return origWidth;
    }

    @Override
    public int getOriginalHeight() {
        return origHeight;
    }

    @Override
    public InferenceSpeed getSpeed() {
        return speed;
    }

    @Override
    public TaskType getTaskType() {
        return TaskType.DETECT;
    }

    @Override
    public Map<Integer, String> getClassNames() {
        return classNames;
    }

    @Override
    public int count() {
        return boxCount;
    }

    /**
     * Get strongly-typed detection boxes. Lazy-loaded on first access.
     *
     * @return list of ClassPrediction objects
     */
    public List<ClassPrediction> getBoxes() {
        if (boxesList == null) {
            boxesList = new ArrayList<>(boxCount);
            for (int i = 0; i < boxCount; i++) {
                int offset = i * 4;
                BoundingBox box = new BoundingBox(
                        boxesXYXY.get(offset),
                        boxesXYXY.get(offset + 1),
                        boxesXYXY.get(offset + 2),
                        boxesXYXY.get(offset + 3)
                );
                float conf = confidences.get(i);
                int clsId = classIds.get(i);
                boxesList.add(new ClassPrediction(box, conf, clsId,
                        classNames.getOrDefault(clsId, "?")));
            }
        }
        return boxesList;
    }

    /**
     * Filter boxes by class name.
     */
    public List<ClassPrediction> filterByClass(String name) {
        return getBoxes().stream()
                .filter(b -> b.className().equals(name))
                .collect(Collectors.toList());
    }

    /**
     * Filter boxes by minimum confidence.
     */
    public List<ClassPrediction> filterByConfidence(float minConf) {
        return getBoxes().stream()
                .filter(b -> b.confidence() >= minConf)
                .collect(Collectors.toList());
    }

    // ==================== Raw Buffer Access (High Performance) ====================

    /**
     * Get the raw boxes buffer. Each box is represented as 4 consecutive floats: x1, y1, x2, y2.
     * <p>
     * The buffer position is at 0 and limit is at boxCount * 4.
     *
     * @return direct FloatBuffer containing box coordinates
     */
    public FloatBuffer getRawBoxesXYXY() {
        boxesXYXY.rewind();
        return boxesXYXY;
    }

    /**
     * Get the raw confidences buffer. Each confidence is a single float.
     * <p>
     * The buffer position is at 0 and limit is at boxCount.
     *
     * @return direct FloatBuffer containing confidences
     */
    public FloatBuffer getRawConfidences() {
        confidences.rewind();
        return confidences;
    }

    /**
     * Get the raw class IDs buffer. Each class ID is a single int.
     * <p>
     * The buffer position is at 0 and limit is at boxCount.
     *
     * @return direct IntBuffer containing class IDs
     */
    public IntBuffer getRawClassIds() {
        classIds.rewind();
        return classIds;
    }

    /**
     * Get the number of detected boxes.
     *
     * @return box count
     */
    @Override
    public int getBoxCount() {
        return boxCount;
    }

    /**
     * Get the box coordinates for a specific index.
     *
     * @param index box index (0-based)
     * @return BoundingBox with coordinates
     */
    public BoundingBox getBoxAt(int index) {
        if (index < 0 || index >= boxCount) {
            throw new IndexOutOfBoundsException("Box index " + index + " out of range [0, " + boxCount + ")");
        }
        int offset = index * 4;
        return new BoundingBox(
                boxesXYXY.get(offset),
                boxesXYXY.get(offset + 1),
                boxesXYXY.get(offset + 2),
                boxesXYXY.get(offset + 3)
        );
    }

    /**
     * Get the confidence for a specific index.
     *
     * @param index box index (0-based)
     * @return confidence value
     */
    public float getConfidenceAt(int index) {
        if (index < 0 || index >= boxCount) {
            throw new IndexOutOfBoundsException("Box index " + index + " out of range [0, " + boxCount + ")");
        }
        return confidences.get(index);
    }

    /**
     * Get the class ID for a specific index.
     *
     * @param index box index (0-based)
     * @return class ID
     */
    public int getClassIdAt(int index) {
        if (index < 0 || index >= boxCount) {
            throw new IndexOutOfBoundsException("Box index " + index + " out of range [0, " + boxCount + ")");
        }
        return classIds.get(index);
    }

    /**
     * Release buffers back to the pool. Call this when done with the result
     * to allow buffer reuse.
     */
    @Override
    public void release() {
        bufferPool.release(boxesXYXY);
        bufferPool.release(confidences);
        bufferPool.release(classIds);
    }
}
