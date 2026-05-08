package io.github.javpower.jpyml.flux;

import io.github.javpower.jpyml.ml.model.Device;

/**
 * Configuration for FLUX.1 image generation.
 */
public class FluxConfig {
    private String modelId = "black-forest-labs/FLUX.1-schnell";
    private Device device = null;
    private String dtype = "auto";
    private String variant = null;
    private int width = 1024;
    private int height = 1024;
    private int steps = 20;
    private float guidance = 3.5f;
    private long seed = -1;
    private String negativePrompt = null;
    private int numImages = 1;

    public FluxConfig modelId(String v) { modelId = v; return this; }
    public FluxConfig device(Device v) { device = v; return this; }
    public FluxConfig device(String v) { device = Device.fromString(v); return this; }
    public FluxConfig dtype(String v) { dtype = v; return this; }
    public FluxConfig variant(String v) { variant = v; return this; }
    public FluxConfig width(int v) { width = v; return this; }
    public FluxConfig height(int v) { height = v; return this; }
    public FluxConfig steps(int v) { steps = v; return this; }
    public FluxConfig guidance(float v) { guidance = v; return this; }
    public FluxConfig seed(long v) { seed = v; return this; }
    public FluxConfig negativePrompt(String v) { negativePrompt = v; return this; }
    public FluxConfig numImages(int v) { numImages = v; return this; }

    public String getModelId() { return modelId; }
    public Device getDevice() { return device; }
    public String getDtype() { return dtype; }
    public String getVariant() { return variant; }
    public int getWidth() { return width; }
    public int getHeight() { return height; }
    public int getSteps() { return steps; }
    public float getGuidance() { return guidance; }
    public long getSeed() { return seed; }
    public String getNegativePrompt() { return negativePrompt; }
    public int getNumImages() { return numImages; }

    /**
     * Common presets for FLUX models.
     */
    public static FluxConfig schnell() {
        return new FluxConfig()
                .modelId("black-forest-labs/FLUX.1-schnell")
                .steps(4)
                .guidance(0.0f);
    }

    public static FluxConfig dev() {
        return new FluxConfig()
                .modelId("black-forest-labs/FLUX.1-dev")
                .steps(20)
                .guidance(3.5f);
    }
}
