package io.github.javpower.jpyml.llm.config;

import io.github.javpower.jpyml.ml.model.Device;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Training hyperparameters for LLM fine-tuning.
 */
public class LLMTrainConfig {

    private int epochs = 3;
    private int batchSize = 4;
    private int gradientAccumulation = 4;
    private double learningRate = 2e-4;
    private String lrScheduler = "cosine";
    private int warmupSteps = 100;
    private int maxSeqLength = 2048;
    private int loggingSteps = 10;
    private int saveSteps = 500;
    private int seed = 42;
    private boolean gradientCheckpointing = true;
    private Device device = null; // null = auto-detect
    private String outputDir = null; // null = ~/.jpy-ml/llm-output

    public static LLMTrainConfig create() { return new LLMTrainConfig(); }

    public LLMTrainConfig epochs(int v) { this.epochs = v; return this; }
    public LLMTrainConfig batchSize(int v) { this.batchSize = v; return this; }
    public LLMTrainConfig gradientAccumulation(int v) { this.gradientAccumulation = v; return this; }
    public LLMTrainConfig learningRate(double v) { this.learningRate = v; return this; }
    public LLMTrainConfig lrScheduler(String v) { this.lrScheduler = v; return this; }
    public LLMTrainConfig warmupSteps(int v) { this.warmupSteps = v; return this; }
    public LLMTrainConfig maxSeqLength(int v) { this.maxSeqLength = v; return this; }
    public LLMTrainConfig loggingSteps(int v) { this.loggingSteps = v; return this; }
    public LLMTrainConfig saveSteps(int v) { this.saveSteps = v; return this; }
    public LLMTrainConfig seed(int v) { this.seed = v; return this; }
    public LLMTrainConfig gradientCheckpointing(boolean v) { this.gradientCheckpointing = v; return this; }
    public LLMTrainConfig device(Device v) { this.device = v; return this; }
    public LLMTrainConfig outputDir(String v) { this.outputDir = v; return this; }

    public Device getDevice() { return device; }

    /** Convert to Python kwargs dict. */
    public Map<String, Object> toDict() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("epochs", epochs);
        m.put("batch_size", batchSize);
        m.put("gradient_accumulation", gradientAccumulation);
        m.put("learning_rate", learningRate);
        m.put("lr_scheduler", lrScheduler);
        m.put("warmup_steps", warmupSteps);
        m.put("max_seq_length", maxSeqLength);
        m.put("logging_steps", loggingSteps);
        m.put("save_steps", saveSteps);
        m.put("seed", seed);
        m.put("gradient_checkpointing", gradientCheckpointing);
        if (outputDir != null) m.put("output_dir", outputDir);
        if (device != null) m.put("device", device.toPython());
        return m;
    }
}
