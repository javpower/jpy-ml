package io.github.javpower.jpyml.llm;

import io.github.javpower.jpyml.core.PythonEngine;
import io.github.javpower.jpyml.core.PythonScriptLoader;
import io.github.javpower.jpyml.llm.config.LLMTrainConfig;
import io.github.javpower.jpyml.llm.config.LoRAConfig;
import io.github.javpower.jpyml.ml.training.ProgressMonitor;
import io.github.javpower.jpyml.ml.training.TrainingCallback;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * LLM fine-tuning builder. Chain config methods then call run() or runAsync().
 */
public class LLMFineTuner {

    private static final Logger log = LoggerFactory.getLogger(LLMFineTuner.class);

    private final LLMModel model;
    private LoRAConfig loraConfig = LoRAConfig.create();
    private String datasetPath;
    private String validationDatasetPath;
    private LLMTrainConfig trainConfig = LLMTrainConfig.create();

    LLMFineTuner(LLMModel model) {
        this.model = model;
    }

    public LLMFineTuner lora(LoRAConfig config) { this.loraConfig = config; return this; }
    public LLMFineTuner dataset(String path) { this.datasetPath = path; return this; }
    public LLMFineTuner validationDataset(String path) { this.validationDatasetPath = path; return this; }
    public LLMFineTuner config(LLMTrainConfig config) { this.trainConfig = config; return this; }

    /**
     * Run fine-tuning synchronously with real-time callbacks.
     */
    public LLMTrainingResult run(TrainingCallback callback) {
        return runInternal(callback);
    }

    /**
     * Run fine-tuning synchronously without callback.
     */
    public LLMTrainingResult run() {
        return runInternal(null);
    }

    /**
     * Run fine-tuning asynchronously with real-time callbacks.
     */
    public CompletableFuture<LLMTrainingResult> runAsync(TrainingCallback callback) {
        return CompletableFuture.supplyAsync(() -> runInternal(callback));
    }

    @SuppressWarnings("unchecked")
    private LLMTrainingResult runInternal(TrainingCallback callback) {
        if (datasetPath == null) throw new IllegalStateException("dataset path is required");

        PythonEngine engine = model.getEngine();
        DependencyManager.ensure("llm");

        try {
            PythonScriptLoader.ensureLoaded(engine, "_jpy_progress.py");
            PythonScriptLoader.ensureLoaded(engine, "_jpy_llm_training.py");

            Map<String, Object> loraDict = loraConfig.toDict();
            Map<String, Object> trainDict = trainConfig.toDict();
            trainDict.put("quantization", model.getQuantization().name().toLowerCase());

            ProgressMonitor monitor = null;
            Path progressFile = null;
            Path cancelFile = null;

            if (callback != null) {
                progressFile = Files.createTempFile("jpy-llm-progress-", ".jsonl");
                cancelFile = Path.of(progressFile.toString() + ".cancel");
                Files.deleteIfExists(cancelFile);
                progressFile.toFile().deleteOnExit();

                monitor = new ProgressMonitor(progressFile, callback);
                monitor.start();
            }

            try {
                engine.put("_jpy_llm_model_path", model.getModelPath());
                engine.put("_jpy_llm_dataset", datasetPath);
                engine.put("_jpy_llm_lora_kwargs", loraDict);
                engine.put("_jpy_llm_train_kwargs", trainDict);
                engine.put("_jpy_llm_progress_file", progressFile != null ? progressFile.toString() : "");
                engine.put("_jpy_llm_cancel_file", cancelFile != null ? cancelFile.toString() : "");

                engine.exec(
                        "_jpy_llm_result = jpy_llm_train(" +
                        "_jpy_llm_model_path, _jpy_llm_dataset, " +
                        "_jpy_llm_lora_kwargs, _jpy_llm_train_kwargs, " +
                        "progress_file=_jpy_llm_progress_file or None, " +
                        "cancel_file=_jpy_llm_cancel_file or None)"
                );

                Map<String, Object> result = engine.eval("_jpy_llm_result");
                List<Map<String, Object>> logList =
                        (List<Map<String, Object>>) result.getOrDefault("log", List.of());

                String adapterPath = (String) result.getOrDefault("adapter_path", "");
                Object finalLoss = result.get("final_loss");

                return new LLMTrainingResult(
                        adapterPath,
                        logList,
                        finalLoss instanceof Number ? ((Number) finalLoss).floatValue() : -1f
                );
            } finally {
                if (monitor != null) {
                    String error = monitor.awaitCompletion();
                    if (callback != null) callback.onComplete(error);
                }
                if (progressFile != null) Files.deleteIfExists(progressFile);
                if (cancelFile != null) Files.deleteIfExists(cancelFile);
            }
        } catch (Exception e) {
            throw new RuntimeException("LLM fine-tuning failed", e);
        }
    }
}
