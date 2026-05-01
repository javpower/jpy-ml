package io.github.javpower.jpyml.ml.result;
import java.util.Map;

import io.github.javpower.jpyml.ml.model.TaskType;

public interface InferenceResult {
    String getSourcePath();
    int getOriginalWidth();
    int getOriginalHeight();
    InferenceSpeed getSpeed();
    TaskType getTaskType();
    Map<Integer, String> getClassNames();
    int count();

    default String toJson() { return ResultSerializer.toJson(this); }
    default Map<String, Object> toMap() { return ResultSerializer.toMap(this); }
}
