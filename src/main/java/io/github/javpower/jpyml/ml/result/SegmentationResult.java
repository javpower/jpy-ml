package io.github.javpower.jpyml.ml.result;
import java.util.*;

import io.github.javpower.jpyml.ml.model.TaskType;

public class SegmentationResult implements InferenceResult {
    private final String sourcePath;
    private final int origWidth, origHeight;
    private final InferenceSpeed speed;
    private final Map<Integer, String> classNames;
    private final List<ClassPrediction> boxes;
    private final List<Mask> masks;
    public SegmentationResult(String sp, int w, int h, InferenceSpeed s, Map<Integer,String> cn, List<ClassPrediction> b, List<Mask> m) {
        sourcePath=sp; origWidth=w; origHeight=h; speed=s; classNames=cn;
        boxes = Collections.unmodifiableList(new ArrayList<>(b));
        masks = Collections.unmodifiableList(new ArrayList<>(m));
    }
    @Override public String getSourcePath() { return sourcePath; }
    @Override public int getOriginalWidth() { return origWidth; }
    @Override public int getOriginalHeight() { return origHeight; }
    @Override public InferenceSpeed getSpeed() { return speed; }
    @Override public TaskType getTaskType() { return TaskType.SEGMENT; }
    @Override public Map<Integer,String> getClassNames() { return classNames; }
    @Override public int count() { return boxes.size(); }
    public List<ClassPrediction> getBoxes() { return boxes; }
    public List<Mask> getMasks() { return masks; }
}
