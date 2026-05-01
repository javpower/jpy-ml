package io.github.javpower.jpyml.ml.result;
import java.util.*;

import io.github.javpower.jpyml.ml.model.TaskType;

public class ClassificationResult implements InferenceResult {
    private final String sourcePath;
    private final int origWidth, origHeight;
    private final InferenceSpeed speed;
    private final Map<Integer, String> classNames;
    private final List<ClassPrediction> predictions;
    public ClassificationResult(String sp, int w, int h, InferenceSpeed s, Map<Integer,String> cn, List<ClassPrediction> p) {
        sourcePath=sp; origWidth=w; origHeight=h; speed=s; classNames=cn; predictions=p;
    }
    @Override public String getSourcePath() { return sourcePath; }
    @Override public int getOriginalWidth() { return origWidth; }
    @Override public int getOriginalHeight() { return origHeight; }
    @Override public InferenceSpeed getSpeed() { return speed; }
    @Override public TaskType getTaskType() { return TaskType.CLASSIFY; }
    @Override public Map<Integer,String> getClassNames() { return classNames; }
    @Override public int count() { return predictions.size(); }
    public int getTop1ClassId() { return predictions.isEmpty() ? -1 : predictions.get(0).classId(); }
    public String getTop1ClassName() { return predictions.isEmpty() ? "" : predictions.get(0).className(); }
    public float getTop1Confidence() { return predictions.isEmpty() ? 0 : predictions.get(0).confidence(); }
    public List<ClassPrediction> getTopK() { return predictions; }
}
