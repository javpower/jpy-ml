package io.github.javpower.jpyml.llm;

import java.util.List;
import java.util.Map;

/**
 * Result of LLM fine-tuning.
 */
public class LLMTrainingResult {

    private final String adapterPath;
    private final List<Map<String, Object>> log;
    private final float finalLoss;

    public LLMTrainingResult(String adapterPath, List<Map<String, Object>> log, float finalLoss) {
        this.adapterPath = adapterPath;
        this.log = log;
        this.finalLoss = finalLoss;
    }

    public String getAdapterPath() { return adapterPath; }
    public List<Map<String, Object>> getLog() { return log; }
    public float getFinalLoss() { return finalLoss; }

    /**
     * Merge the LoRA adapter back into the base model.
     *
     * @return path to the merged model directory
     */
    public String mergeAdapter() {
        return LLMModel.mergeAdapterInternal(adapterPath);
    }
}
