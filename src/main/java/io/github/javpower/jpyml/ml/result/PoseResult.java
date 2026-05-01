package io.github.javpower.jpyml.ml.result;
import java.util.*;

import io.github.javpower.jpyml.ml.model.TaskType;

public class PoseResult implements InferenceResult {
    private final String sourcePath;
    private final int origWidth, origHeight;
    private final InferenceSpeed speed;
    private final Map<Integer, String> classNames;
    private final List<ClassPrediction> boxes;
    private final List<KeypointCollection> keypoints;
    public PoseResult(String sp, int w, int h, InferenceSpeed s, Map<Integer,String> cn, List<ClassPrediction> b, List<KeypointCollection> k) {
        sourcePath=sp; origWidth=w; origHeight=h; speed=s; classNames=cn; boxes=b; keypoints=k;
    }
    @Override public String getSourcePath() { return sourcePath; }
    @Override public int getOriginalWidth() { return origWidth; }
    @Override public int getOriginalHeight() { return origHeight; }
    @Override public InferenceSpeed getSpeed() { return speed; }
    @Override public TaskType getTaskType() { return TaskType.POSE; }
    @Override public Map<Integer,String> getClassNames() { return classNames; }
    @Override public int count() { return boxes.size(); }
    public int personCount() { return boxes.size(); }
    public List<ClassPrediction> getBoxes() { return boxes; }
    public List<KeypointCollection> getKeypoints() { return keypoints; }
}
