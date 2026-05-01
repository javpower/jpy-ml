package io.github.javpower.jpyml.ml.model;
public enum TaskType {
    DETECT("detect"), SEGMENT("segment"), CLASSIFY("classify"), POSE("pose"), OBB("obb");
    private final String key;
    TaskType(String key) { this.key = key; }
    public String getKey() { return key; }
    public static TaskType fromString(String s) {
        for (TaskType t : values()) if (t.key.equals(s)) return t;
        return DETECT;
    }
}
