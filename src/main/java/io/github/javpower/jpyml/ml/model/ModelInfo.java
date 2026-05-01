package io.github.javpower.jpyml.ml.model;

import java.util.Map;

public class ModelInfo {
    private final TaskType taskType;
    private final Map<Integer, String> classNames;
    private final int numClasses;
    private final long parameters;
    private final int layers;

    public ModelInfo(TaskType taskType, Map<Integer, String> classNames, int numClasses, long parameters, int layers) {
        this.taskType = taskType;
        this.classNames = classNames;
        this.numClasses = numClasses;
        this.parameters = parameters;
        this.layers = layers;
    }

    public TaskType getTaskType() { return taskType; }
    public Map<Integer, String> getClassNames() { return classNames; }
    public int getNumClasses() { return numClasses; }
    public long getParameters() { return parameters; }
    public int getLayers() { return layers; }

    @Override
    public String toString() {
        return String.format("ModelInfo{task=%s, classes=%d, params=%dM, layers=%d}",
                taskType, numClasses, parameters / 1_000_000, layers);
    }
}
