package io.github.javpower.jpyml.llm.config;

import java.util.Arrays;
import java.util.List;

/**
 * LoRA adapter configuration for parameter-efficient fine-tuning.
 */
public class LoRAConfig {

    private int rank = 16;
    private int alpha = 32;
    private List<String> targetModules = null; // null = auto-detect from model
    private double dropout = 0.05;
    private String bias = "none";

    public static LoRAConfig create() { return new LoRAConfig(); }

    public LoRAConfig rank(int rank) { this.rank = rank; return this; }
    public LoRAConfig alpha(int alpha) { this.alpha = alpha; return this; }
    public LoRAConfig targetModules(String... modules) { this.targetModules = Arrays.asList(modules); return this; }
    public LoRAConfig dropout(double dropout) { this.dropout = dropout; return this; }
    public LoRAConfig bias(String bias) { this.bias = bias; return this; }

    /** Convert to Python kwargs dict. */
    public java.util.Map<String, Object> toDict() {
        java.util.Map<String, Object> m = new java.util.LinkedHashMap<>();
        m.put("rank", rank);
        m.put("alpha", alpha);
        if (targetModules != null) m.put("target_modules", targetModules);
        m.put("dropout", dropout);
        m.put("bias", bias);
        return m;
    }
}
