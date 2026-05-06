package io.github.javpower.jpyml.llm.config;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Generation parameters for LLM inference (chat).
 */
public class GenerationConfig {

    private int maxNewTokens = 512;
    private double temperature = 0.7;
    private double topP = 0.9;
    private boolean doSample = true;
    private double repetitionPenalty = 1.1;

    public static GenerationConfig create() { return new GenerationConfig(); }

    public GenerationConfig maxNewTokens(int v) { this.maxNewTokens = v; return this; }
    public GenerationConfig temperature(double v) { this.temperature = v; return this; }
    public GenerationConfig topP(double v) { this.topP = v; return this; }
    public GenerationConfig doSample(boolean v) { this.doSample = v; return this; }
    public GenerationConfig repetitionPenalty(double v) { this.repetitionPenalty = v; return this; }

    public Map<String, Object> toDict() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("max_new_tokens", maxNewTokens);
        m.put("temperature", temperature);
        m.put("top_p", topP);
        m.put("do_sample", doSample);
        m.put("repetition_penalty", repetitionPenalty);
        return m;
    }
}
