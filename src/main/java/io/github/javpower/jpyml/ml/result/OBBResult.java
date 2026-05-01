package io.github.javpower.jpyml.ml.result;
import java.util.*;

import io.github.javpower.jpyml.ml.model.TaskType;

public class OBBResult implements InferenceResult {
    private final String sourcePath;
    private final int origWidth, origHeight;
    private final InferenceSpeed speed;
    private final Map<Integer, String> classNames;
    private final List<OBBPrediction> predictions;
    public OBBResult(String sp, int w, int h, InferenceSpeed s, Map<Integer,String> cn, List<OBBPrediction> p) {
        sourcePath=sp; origWidth=w; origHeight=h; speed=s; classNames=cn; predictions=p;
    }
    @Override public String getSourcePath() { return sourcePath; }
    @Override public int getOriginalWidth() { return origWidth; }
    @Override public int getOriginalHeight() { return origHeight; }
    @Override public InferenceSpeed getSpeed() { return speed; }
    @Override public TaskType getTaskType() { return TaskType.OBB; }
    @Override public Map<Integer,String> getClassNames() { return classNames; }
    @Override public int count() { return predictions.size(); }
    public List<OBBPrediction> getPredictions() { return predictions; }
}
