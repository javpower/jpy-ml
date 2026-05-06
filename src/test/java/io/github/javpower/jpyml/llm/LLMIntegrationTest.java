package io.github.javpower.jpyml.llm;

import io.github.javpower.jpyml.core.PythonRuntime;
import io.github.javpower.jpyml.llm.config.GenerationConfig;
import io.github.javpower.jpyml.llm.config.LoRAConfig;
import io.github.javpower.jpyml.llm.config.LLMTrainConfig;
import io.github.javpower.jpyml.llm.config.Quantization;
import io.github.javpower.jpyml.llm.data.ChatMessage;
import io.github.javpower.jpyml.llm.data.ChatResponse;
import io.github.javpower.jpyml.ml.training.TrainingCallback;
import org.junit.jupiter.api.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class LLMIntegrationTest {

    private static final String MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct";
    private static String modelPath;

    private static final Path PROJECT_ROOT = Path.of(System.getProperty("user.dir"));

    @BeforeAll
    static void init() throws Exception {
        Path pythonBin = PROJECT_ROOT.resolve(".venv/bin/python3");
        Path jepLib = PROJECT_ROOT.resolve(".venv/lib/python3.12/site-packages/jep/libjep.jnilib");
        PythonRuntime.init(pythonBin, jepLib);
    }

    @Test
    @Order(1)
    void testDownloadModel() {
        LLMModel model = LLMModel.download(MODEL_ID);
        assertNotNull(model);
        modelPath = model.getModelPath();
        assertNotNull(modelPath);
        assertTrue(Files.exists(Path.of(modelPath).resolve("config.json")),
                "Model config.json should exist at " + modelPath);
        System.out.println("Model downloaded to: " + modelPath);
    }

    @Test
    @Order(2)
    void testChatInference() {
        LLMModel model = LLMModel.load(modelPath)
                .quantize(Quantization.NONE); // macOS, no quantization

        ChatResponse resp = model.chat(
                ChatMessage.system("你是一个有用的助手"),
                ChatMessage.user("你好，请用一句话介绍你自己")
        );

        assertNotNull(resp.getContent());
        assertFalse(resp.getContent().isBlank());
        assertTrue(resp.getPromptTokens() > 0);
        System.out.println("Chat response: " + resp.getContent());
        System.out.println("Tokens: prompt=" + resp.getPromptTokens() + " completion=" + resp.getCompletionTokens());
    }

    @Test
    @Order(3)
    void testFineTuneWithCallback() throws IOException {
        // Create a tiny training dataset
        Path datasetFile = Files.createTempFile("jpy-llm-test-data-", ".jsonl");
        datasetFile.toFile().deleteOnExit();
        Files.writeString(datasetFile, String.join("\n",
                "{\"messages\": [{\"role\": \"user\", \"content\": \"1+1等于几\"}, {\"role\": \"assistant\", \"content\": \"1+1等于2\"}]}",
                "{\"messages\": [{\"role\": \"user\", \"content\": \"2+3等于几\"}, {\"role\": \"assistant\", \"content\": \"2+3等于5\"}]}",
                "{\"messages\": [{\"role\": \"user\", \"content\": \"天空是什么颜色\"}, {\"role\": \"assistant\", \"content\": \"天空是蓝色的\"}]}"
        ));

        LLMModel model = LLMModel.load(modelPath)
                .quantize(Quantization.NONE);

        AtomicInteger callbackCount = new AtomicInteger(0);
        long startTime = System.currentTimeMillis();

        LLMTrainingResult result = model.fineTune()
                .lora(LoRAConfig.create().rank(4).alpha(8))
                .dataset(datasetFile.toString())
                .config(LLMTrainConfig.create()
                        .epochs(2)
                        .batchSize(1)
                        .gradientAccumulation(1)
                        .learningRate(1e-4)
                        .maxSeqLength(256)
                        .loggingSteps(1)
                        .gradientCheckpointing(false))
                .run((epoch, log) -> {
                    int n = callbackCount.incrementAndGet();
                    long elapsed = System.currentTimeMillis() - startTime;
                    System.out.println("  [callback " + n + "] " + elapsed + "ms: " + log);
                });

        assertNotNull(result.getAdapterPath());
        assertFalse(result.getAdapterPath().isEmpty());
        System.out.println("Adapter saved to: " + result.getAdapterPath());
        System.out.println("Total callback events: " + callbackCount.get());
    }

    @Test
    @Order(4)
    void testAsyncFineTune() throws Exception {
        Path datasetFile = Files.createTempFile("jpy-llm-async-data-", ".jsonl");
        datasetFile.toFile().deleteOnExit();
        Files.writeString(datasetFile, String.join("\n",
                "{\"messages\": [{\"role\": \"user\", \"content\": \"你好\"}, {\"role\": \"assistant\", \"content\": \"你好！有什么可以帮你的吗？\"}]}"
        ));

        LLMModel model = LLMModel.load(modelPath)
                .quantize(Quantization.NONE);

        AtomicInteger count = new AtomicInteger(0);

        CompletableFuture<LLMTrainingResult> future = model.fineTune()
                .lora(LoRAConfig.create().rank(4).alpha(8))
                .dataset(datasetFile.toString())
                .config(LLMTrainConfig.create()
                        .epochs(1)
                        .batchSize(1)
                        .gradientAccumulation(1)
                        .learningRate(1e-4)
                        .maxSeqLength(128)
                        .loggingSteps(1)
                        .gradientCheckpointing(false))
                .runAsync((step, log) -> {
                    count.incrementAndGet();
                    System.out.println("  [async] " + log);
                });

        LLMTrainingResult result = future.get(10, java.util.concurrent.TimeUnit.MINUTES);
        assertNotNull(result.getAdapterPath());
        System.out.println("Async training done. Callbacks received: " + count.get());
    }
}
