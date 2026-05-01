package io.github.javpower.jpyml.ml.training;

import io.github.javpower.jpyml.ml.model.Device;

import java.util.LinkedHashMap;
import java.util.Map;

public class TrainingConfig {
    private String dataConfig;
    private int epochs = 100;
    private int batchSize = 16;
    private int imageSize = 640;
    private Device device = null;
    private int workers = 8;
    private OptimizerType optimizer = OptimizerType.AUTO;
    private float initialLR = 0.01f;
    private float momentum = 0.937f;
    private float weightDecay = 0.0005f;
    private int patience = 100;
    private int seed = 0;
    private boolean amp = true;
    private boolean val = true;
    private boolean plots = true;
    private String project = "runs/train";
    private String name = "jpy_train";
    private boolean resume = false;
    private AugmentationConfig augmentation = new AugmentationConfig();

    public TrainingConfig dataConfig(String v) { dataConfig = v; return this; }
    public TrainingConfig epochs(int v) { epochs = v; return this; }
    public TrainingConfig batchSize(int v) { batchSize = v; return this; }
    public TrainingConfig imageSize(int v) { imageSize = v; return this; }
    public TrainingConfig device(Device v) { device = v; return this; }
    public TrainingConfig device(String v) { device = Device.fromString(v); return this; }
    public TrainingConfig workers(int v) { workers = v; return this; }
    public TrainingConfig optimizer(OptimizerType v) { optimizer = v; return this; }
    public TrainingConfig initialLR(float v) { initialLR = v; return this; }
    public TrainingConfig momentum(float v) { momentum = v; return this; }
    public TrainingConfig weightDecay(float v) { weightDecay = v; return this; }
    public TrainingConfig patience(int v) { patience = v; return this; }
    public TrainingConfig seed(int v) { seed = v; return this; }
    public TrainingConfig amp(boolean v) { amp = v; return this; }
    public TrainingConfig val(boolean v) { val = v; return this; }
    public TrainingConfig plots(boolean v) { plots = v; return this; }
    public TrainingConfig project(String v) { project = v; return this; }
    public TrainingConfig name(String v) { name = v; return this; }
    public TrainingConfig resume(boolean v) { resume = v; return this; }
    public TrainingConfig augmentation(AugmentationConfig v) { augmentation = v; return this; }

    public String getDataConfig() { return dataConfig; }

    public Map<String, Object> toPythonDict() {
        Map<String, Object> m = new LinkedHashMap<>();
        if (dataConfig != null) m.put("data", dataConfig);
        m.put("epochs", epochs);
        m.put("batch", batchSize);
        m.put("imgsz", imageSize);
        if (device != null) m.put("device", device.toPython());
        m.put("workers", workers);
        if (optimizer != OptimizerType.AUTO) m.put("optimizer", optimizer.name().toLowerCase());
        m.put("lr0", initialLR);
        m.put("momentum", momentum);
        m.put("weight_decay", weightDecay);
        m.put("patience", patience);
        m.put("seed", seed);
        m.put("amp", amp);
        m.put("val", val);
        m.put("plots", plots);
        m.put("project", project);
        m.put("name", name);
        if (resume) m.put("resume", true);
        if (augmentation != null) m.putAll(augmentation.toMap());
        return m;
    }
}
