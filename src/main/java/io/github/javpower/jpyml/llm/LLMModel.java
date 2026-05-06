package io.github.javpower.jpyml.llm;

import io.github.javpower.jpyml.core.PythonEngine;
import io.github.javpower.jpyml.core.PythonScriptLoader;
import io.github.javpower.jpyml.llm.config.GenerationConfig;
import io.github.javpower.jpyml.llm.config.Quantization;
import io.github.javpower.jpyml.llm.data.ChatMessage;
import io.github.javpower.jpyml.llm.data.ChatResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Unified entry point for LLM fine-tuning and inference.
 * <p>
 * Usage:
 * <pre>
 *   LLMModel model = LLMModel.download("Qwen/Qwen2.5-0.5B-Instruct")
 *       .quantize(Quantization.AUTO);
 *
 *   // Fine-tune
 *   LLMTrainingResult result = model.fineTune()
 *       .lora(LoRAConfig.create().rank(8))
 *       .dataset("chat_data.jsonl")
 *       .run();
 *
 *   // Chat
 *   ChatResponse resp = model.chat(
 *       ChatMessage.user("Hello!")
 *   );
 * </pre>
 */
public class LLMModel {

    private static final Logger log = LoggerFactory.getLogger(LLMModel.class);

    private final String modelPath;
    private final String adapterPath;
    private final Quantization quantization;
    private final PythonEngine engine;

    private LLMModel(String modelPath, String adapterPath, Quantization quantization) {
        this.modelPath = modelPath;
        this.adapterPath = adapterPath;
        this.quantization = quantization;
        this.engine = PythonEngine.getInstance();
    }

    // ==================== Factory ====================

    /**
     * Load a local LLM model.
     *
     * @param modelPath local path to the model directory
     */
    public static LLMModel load(String modelPath) {
        return new LLMModel(modelPath, null, Quantization.AUTO);
    }

    /**
     * Download a model from HuggingFace Hub, then load it.
     * Uses cache at ~/.jpy-ml/llm-models/.
     *
     * @param modelId HuggingFace model ID (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
     */
    public static LLMModel download(String modelId) {
        return download(modelId, null);
    }

    /**
     * Download a model from HuggingFace Hub with progress callback.
     */
    public static LLMModel download(String modelId, Consumer<String> progressListener) {
        DependencyManager.ensure("llm");
        PythonEngine engine = PythonEngine.getInstance();

        try {
            PythonScriptLoader.ensureLoaded(engine, "_jpy_progress.py");
            PythonScriptLoader.ensureLoaded(engine, "_jpy_llm_download.py");

            engine.put("_jpy_llm_download_id", modelId);
            engine.exec("_jpy_llm_download_result = jpy_llm_download(_jpy_llm_download_id)");

            @SuppressWarnings("unchecked")
            Map<String, Object> result = engine.eval("_jpy_llm_download_result");
            String path = (String) result.get("path");
            boolean cached = Boolean.TRUE.equals(result.get("cached"));
            log.info("Model {} at {} (cached={})", modelId, path, cached);

            return new LLMModel(path, null, Quantization.AUTO);
        } catch (Exception e) {
            throw new RuntimeException("Failed to download model: " + modelId, e);
        }
    }

    // ==================== Builder setters ====================

    public LLMModel quantize(Quantization q) {
        return new LLMModel(this.modelPath, this.adapterPath, q);
    }

    public LLMModel adapter(String adapterPath) {
        return new LLMModel(this.modelPath, adapterPath, this.quantization);
    }

    // ==================== Fine-tuning ====================

    public LLMFineTuner fineTune() {
        DependencyManager.ensure("llm");
        return new LLMFineTuner(this);
    }

    // ==================== Inference ====================

    /**
     * Synchronous chat completion.
     */
    public ChatResponse chat(ChatMessage... messages) {
        return chat(List.of(messages), GenerationConfig.create());
    }

    /**
     * Synchronous chat completion with generation config.
     */
    public ChatResponse chat(List<ChatMessage> messages, GenerationConfig config) {
        DependencyManager.ensure("llm");
        try {
            PythonScriptLoader.ensureLoaded(engine, "_jpy_llm_inference.py");

            List<Map<String, String>> msgList = new ArrayList<>();
            for (ChatMessage msg : messages) {
                msgList.add(msg.toMap());
            }

            Map<String, Object> genDict = config.toDict();
            String quantName = quantization == Quantization.AUTO ? null : quantization.name().toLowerCase();
            if (quantization == Quantization.AUTO) {
                quantName = "auto";
            }

            engine.put("_jpy_llm_chat_model", modelPath);
            engine.put("_jpy_llm_chat_adapter", adapterPath != null ? adapterPath : "");
            engine.put("_jpy_llm_chat_messages", msgList);
            engine.put("_jpy_llm_chat_gen", genDict);
            engine.put("_jpy_llm_chat_quant", quantName != null ? quantName : "");

            engine.exec(
                    "_jpy_llm_chat_result = jpy_llm_chat(" +
                    "_jpy_llm_chat_model, " +
                    "_jpy_llm_chat_adapter or None, " +
                    "_jpy_llm_chat_messages, " +
                    "_jpy_llm_chat_gen, " +
                    "quantization=_jpy_llm_chat_quant or None)"
            );

            @SuppressWarnings("unchecked")
            Map<String, Object> result = engine.eval("_jpy_llm_chat_result");

            return new ChatResponse(
                    (String) result.getOrDefault("content", ""),
                    ((Number) result.getOrDefault("prompt_tokens", 0)).intValue(),
                    ((Number) result.getOrDefault("completion_tokens", 0)).intValue(),
                    (String) result.getOrDefault("finish_reason", "stop")
            );
        } catch (Exception e) {
            throw new RuntimeException("Chat inference failed", e);
        }
    }

    // ==================== Merge ====================

    /**
     * Merge LoRA adapter into base model. Static utility.
     *
     * @param modelPath   base model path
     * @param adapterPath LoRA adapter path
     * @param outputPath  output path (null = auto)
     * @return path to merged model
     */
    public static String mergeAdapter(String modelPath, String adapterPath, String outputPath) {
        return mergeAdapterInternal(modelPath, adapterPath, outputPath);
    }

    static String mergeAdapterInternal(String adapterPath) {
        // Used from LLMTrainingResult — we don't know the base model path here
        // so we pass null and let Python figure it out from adapter config
        return mergeAdapterInternal(null, adapterPath, null);
    }

    private static String mergeAdapterInternal(String modelPath, String adapterPath, String outputPath) {
        DependencyManager.ensure("llm");
        PythonEngine engine = PythonEngine.getInstance();
        try {
            PythonScriptLoader.ensureLoaded(engine, "_jpy_llm_merge.py");

            engine.put("_jpy_llm_merge_model", modelPath != null ? modelPath : "");
            engine.put("_jpy_llm_merge_adapter", adapterPath);
            engine.put("_jpy_llm_merge_output", outputPath != null ? outputPath : "");

            engine.exec(
                    "_jpy_llm_merge_result = jpy_llm_merge(" +
                    "_jpy_llm_merge_model or None, " +
                    "_jpy_llm_merge_adapter, " +
                    "_jpy_llm_merge_output or None)"
            );

            @SuppressWarnings("unchecked")
            Map<String, Object> result = engine.eval("_jpy_llm_merge_result");
            return (String) result.get("merged_path");
        } catch (Exception e) {
            throw new RuntimeException("Adapter merge failed", e);
        }
    }

    // ==================== Accessors ====================

    String getModelPath() { return modelPath; }
    Quantization getQuantization() { return quantization; }
    PythonEngine getEngine() { return engine; }
}
