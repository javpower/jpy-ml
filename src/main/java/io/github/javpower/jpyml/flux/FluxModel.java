package io.github.javpower.jpyml.flux;

import io.github.javpower.jpyml.core.PythonEngine;
import io.github.javpower.jpyml.core.PythonScriptLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * FLUX.1 text-to-image generation model.
 *
 * <pre>{@code
 * try (FluxModel flux = new FluxModel("black-forest-labs/FLUX.1-schnell")) {
 *     FluxResult result = flux.generate("A cat in space", "output.png");
 *     System.out.println(result);
 * }
 * }</pre>
 */
public class FluxModel implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(FluxModel.class);

    private final String modelId;
    private final String device;
    private final String dtype;
    private final String variant;
    private boolean loaded = false;

    public FluxModel(String modelId) {
        this(modelId, "auto", "auto", null);
    }

    public FluxModel(String modelId, String device) {
        this(modelId, device, "auto", null);
    }

    public FluxModel(String modelId, String device, String dtype, String variant) {
        this.modelId = modelId;
        this.device = device;
        this.dtype = dtype;
        this.variant = variant;
    }

    /**
     * Load model into memory.
     */
    public void load() {
        if (loaded) return;
        PythonEngine engine = PythonEngine.getInstance();
        PythonScriptLoader.ensureLoaded(engine, "_jpy_flux.py");

        engine.exec(String.format(
                "_jpy_flux_load_result = jpy_flux_load(%s, %s, %s, %s)",
                quote(modelId), quote(device), quote(dtype), quote(variant)));

        loaded = true;
        log.info("FLUX model loaded: {}", modelId);
    }

    /**
     * Generate image from text prompt.
     */
    public FluxResult generate(String prompt, String outputPath) {
        return generate(prompt, outputPath, new FluxConfig().modelId(modelId));
    }

    /**
     * Generate image with custom config.
     */
    public FluxResult generate(String prompt, String outputPath, FluxConfig config) {
        load();

        PythonEngine engine = PythonEngine.getInstance();

        StringBuilder code = new StringBuilder();
        code.append("_jpy_flux_gen_result = jpy_flux_generate(");
        code.append(quote(config.getModelId())).append(", ");
        code.append(quote(prompt)).append(", ");
        code.append(quote(outputPath)).append(", ");
        code.append("device=").append(quote(device)).append(", ");
        code.append("dtype=").append(quote(dtype)).append(", ");
        code.append("variant=").append(quote(variant)).append(", ");
        code.append("width=").append(config.getWidth()).append(", ");
        code.append("height=").append(config.getHeight()).append(", ");
        code.append("steps=").append(config.getSteps()).append(", ");
        code.append("guidance=").append(config.getGuidance()).append(", ");
        code.append("seed=").append(config.getSeed()).append(", ");
        if (config.getNegativePrompt() != null) {
            code.append("negative_prompt=").append(quote(config.getNegativePrompt())).append(", ");
        }
        code.append("num_images=").append(config.getNumImages());
        code.append(")");

        engine.exec(code.toString());

        @SuppressWarnings("unchecked")
        Map<String, Object> result = (Map<String, Object>) engine.eval("_jpy_flux_gen_result");
        return FluxResult.fromMap(result);
    }

    /**
     * Image-to-image generation.
     */
    public FluxResult img2img(String prompt, String inputPath, String outputPath, FluxConfig config) {
        load();

        PythonEngine engine = PythonEngine.getInstance();

        StringBuilder code = new StringBuilder();
        code.append("_jpy_flux_i2i_result = jpy_flux_img2img(");
        code.append(quote(config.getModelId())).append(", ");
        code.append(quote(prompt)).append(", ");
        code.append(quote(inputPath)).append(", ");
        code.append(quote(outputPath)).append(", ");
        code.append("device=").append(quote(device)).append(", ");
        code.append("dtype=").append(quote(dtype)).append(", ");
        code.append("variant=").append(quote(variant)).append(", ");
        code.append("steps=").append(config.getSteps()).append(", ");
        code.append("guidance=").append(config.getGuidance()).append(", ");
        code.append("seed=").append(config.getSeed());
        code.append(")");

        engine.exec(code.toString());

        @SuppressWarnings("unchecked")
        Map<String, Object> result = (Map<String, Object>) engine.eval("_jpy_flux_i2i_result");
        return FluxResult.fromMap(result);
    }

    /**
     * Unload model from memory.
     */
    public void unload() {
        if (!loaded) return;
        PythonEngine engine = PythonEngine.getInstance();
        engine.exec("jpy_flux_unload(" + quote(modelId) + ")");
        loaded = false;
        log.info("FLUX model unloaded: {}", modelId);
    }

    /**
     * List available FLUX models.
     */
    @SuppressWarnings("unchecked")
    public static List<Map<String, Object>> listModels() {
        PythonEngine engine = PythonEngine.getInstance();
        PythonScriptLoader.ensureLoaded(engine, "_jpy_flux.py");
        engine.exec("_jpy_flux_models = jpy_flux_list_models()");
        Map<String, Object> result = (Map<String, Object>) engine.eval("_jpy_flux_models");
        return (List<Map<String, Object>>) result.get("models");
    }

    @Override
    public void close() {
        unload();
    }

    public String getModelId() { return modelId; }
    public boolean isLoaded() { return loaded; }

    private static String quote(String s) {
        if (s == null) return "None";
        return "'" + s.replace("'", "\\'") + "'";
    }
}
