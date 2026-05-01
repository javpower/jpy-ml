package io.github.javpower.jpyml.ml.result;
import java.util.*;
import java.util.stream.Collectors;

import io.github.javpower.jpyml.ml.model.TaskType;

public class DetectionResult implements InferenceResult {
    private final String sourcePath;
    private final int origWidth, origHeight;
    private final InferenceSpeed speed;
    private final Map<Integer, String> classNames;
    private final List<ClassPrediction> boxes;
    public DetectionResult(String sourcePath, int origWidth, int origHeight, InferenceSpeed speed, Map<Integer, String> classNames, List<ClassPrediction> boxes) {
        this.sourcePath = sourcePath; this.origWidth = origWidth; this.origHeight = origHeight;
        this.speed = speed; this.classNames = classNames; this.boxes = boxes;
    }
    @Override public String getSourcePath() { return sourcePath; }
    @Override public int getOriginalWidth() { return origWidth; }
    @Override public int getOriginalHeight() { return origHeight; }
    @Override public InferenceSpeed getSpeed() { return speed; }
    @Override public TaskType getTaskType() { return TaskType.DETECT; }
    @Override public Map<Integer, String> getClassNames() { return classNames; }
    @Override public int count() { return boxes.size(); }
    public List<ClassPrediction> getBoxes() { return boxes; }
    public List<ClassPrediction> filterByClass(String name) { return boxes.stream().filter(b -> b.className().equals(name)).collect(Collectors.toList()); }
    public List<ClassPrediction> filterByConfidence(float minConf) { return boxes.stream().filter(b -> b.confidence() >= minConf).collect(Collectors.toList()); }
}
