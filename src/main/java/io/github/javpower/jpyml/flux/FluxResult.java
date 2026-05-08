package io.github.javpower.jpyml.flux;

import java.util.List;

/**
 * Result of FLUX.1 image generation.
 */
public class FluxResult {
    private final String status;
    private final List<String> outputPaths;
    private final int width;
    private final int height;
    private final int steps;
    private final float guidance;
    private final Long seed;
    private final float elapsedSeconds;
    private final String modelId;
    private final String prompt;
    private final String error;

    private FluxResult(String status, List<String> outputPaths, int width, int height,
                       int steps, float guidance, Long seed, float elapsedSeconds,
                       String modelId, String prompt, String error) {
        this.status = status;
        this.outputPaths = outputPaths;
        this.width = width;
        this.height = height;
        this.steps = steps;
        this.guidance = guidance;
        this.seed = seed;
        this.elapsedSeconds = elapsedSeconds;
        this.modelId = modelId;
        this.prompt = prompt;
        this.error = error;
    }

    @SuppressWarnings("unchecked")
    public static FluxResult fromMap(java.util.Map<String, Object> map) {
        String status = (String) map.getOrDefault("status", "unknown");
        List<String> paths = (List<String>) map.getOrDefault("output_paths", List.of());

        String singlePath = (String) map.get("output_path");
        if (singlePath != null && paths.isEmpty()) {
            paths = List.of(singlePath);
        }

        return new FluxResult(
                status,
                paths,
                ((Number) map.getOrDefault("width", 0)).intValue(),
                ((Number) map.getOrDefault("height", 0)).intValue(),
                ((Number) map.getOrDefault("steps", 0)).intValue(),
                ((Number) map.getOrDefault("guidance", 0)).floatValue(),
                map.containsKey("seed") ? ((Number) map.get("seed")).longValue() : null,
                ((Number) map.getOrDefault("elapsed_seconds", 0)).floatValue(),
                (String) map.getOrDefault("model_id", ""),
                (String) map.getOrDefault("prompt", ""),
                (String) map.get("error")
        );
    }

    public boolean isSuccess() { return "success".equals(status); }
    public String getStatus() { return status; }
    public List<String> getOutputPaths() { return outputPaths; }
    public String getFirstOutputPath() { return outputPaths.isEmpty() ? null : outputPaths.get(0); }
    public int getWidth() { return width; }
    public int getHeight() { return height; }
    public int getSteps() { return steps; }
    public float getGuidance() { return guidance; }
    public Long getSeed() { return seed; }
    public float getElapsedSeconds() { return elapsedSeconds; }
    public String getModelId() { return modelId; }
    public String getPrompt() { return prompt; }
    public String getError() { return error; }

    @Override
    public String toString() {
        if (!isSuccess()) {
            return "FluxResult{error='" + error + "'}";
        }
        return String.format("FluxResult{output='%s', %dx%d, steps=%d, %.1fs}",
                getFirstOutputPath(), width, height, steps, elapsedSeconds);
    }
}
