package io.github.javpower.jpyml.llm.config;

/**
 * Quantization strategy for LLM model loading.
 * AUTO selects NF4 on Linux+CUDA, no quantization on macOS.
 */
public enum Quantization {
    /** No quantization, full precision */
    NONE,
    /** 4-bit NF4 quantization via bitsandbytes (Linux+CUDA only) */
    NF4,
    /** 8-bit quantization via bitsandbytes (Linux+CUDA only) */
    INT8,
    /** Auto-select based on platform: NF4 on Linux+CUDA, NONE otherwise */
    AUTO
}
