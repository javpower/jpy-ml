package io.github.javpower.jpyml.cli;

import io.github.javpower.jpyml.core.PythonRuntime;
import io.github.javpower.jpyml.llm.LLMModel;
import io.github.javpower.jpyml.llm.config.GenerationConfig;
import io.github.javpower.jpyml.llm.config.Quantization;
import io.github.javpower.jpyml.llm.data.ChatMessage;
import io.github.javpower.jpyml.llm.data.ChatResponse;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.Callable;

@Command(name = "chat", description = "与 LLM 模型对话（单次或交互模式）")
public class ChatCommand implements Callable<Integer> {

    @Option(names = {"-m", "--model"}, description = "模型路径或 HuggingFace ID（如 Qwen/Qwen2.5-0.5B-Instruct）", required = true)
    String model;

    @Option(names = {"--message"}, description = "单次消息（省略则进入交互模式）")
    String message;

    @Option(names = {"--adapter"}, description = "LoRA 适配器路径")
    String adapter;

    @Option(names = {"--quantize"}, description = "量化: auto, nf4, int8, none（默认: auto）")
    String quantize = "auto";

    @Option(names = {"--max-tokens"}, description = "最大新 token 数（默认: 512）")
    int maxTokens = 512;

    @Option(names = {"--temperature"}, description = "采样温度（默认: 0.7）")
    double temperature = 0.7;

    @Option(names = {"--top-p"}, description = "Top-p 采样（默认: 0.9）")
    double topP = 0.9;

    @Option(names = {"--system"}, description = "系统提示")
    String systemPrompt;

    @Override
    public Integer call() throws Exception {
        CliRunner.initRuntime();

        LLMModel lm;
        if (model.startsWith("http") || model.contains("/")) {
            lm = LLMModel.download(model);
        } else {
            lm = LLMModel.load(model);
        }

        lm = lm.quantize(parseQuantization());
        if (adapter != null) {
            lm = lm.adapter(adapter);
        }

        GenerationConfig genConfig = GenerationConfig.create()
                .maxNewTokens(maxTokens)
                .temperature(temperature)
                .topP(topP);

        if (message != null) {
            return oneShot(lm, genConfig);
        }
        return interactive(lm, genConfig);
    }

    private int oneShot(LLMModel lm, GenerationConfig genConfig) {
        try {
            List<ChatMessage> messages = buildMessages(message);
            ChatResponse resp = lm.chat(messages, genConfig);
            System.out.println(resp.getContent());
            System.err.printf("\n[tokens: prompt=%d, completion=%d, reason=%s]%n",
                    resp.getPromptTokens(), resp.getCompletionTokens(), resp.getFinishReason());
            return 0;
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            return 1;
        }
    }

    private int interactive(LLMModel lm, GenerationConfig genConfig) {
        System.out.println("交互聊天模式。输入 /quit 退出。\n");
        List<ChatMessage> history = new ArrayList<>();
        if (systemPrompt != null) {
            history.add(ChatMessage.system(systemPrompt));
        }

        @SuppressWarnings("resource")
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("你> ");
            System.out.flush();
            if (!scanner.hasNextLine()) break;
            String line = scanner.nextLine().trim();
            if (line.isEmpty()) continue;
            if (line.equals("/quit") || line.equals("/exit")) break;

            history.add(ChatMessage.user(line));
            try {
                ChatResponse resp = lm.chat(history, genConfig);
                System.out.println("AI> " + resp.getContent());
                history.add(ChatMessage.assistant(resp.getContent()));
            } catch (Exception e) {
                System.err.println("Error: " + e.getMessage());
                history.remove(history.size() - 1);
            }
        }
        return 0;
    }

    private List<ChatMessage> buildMessages(String userMessage) {
        List<ChatMessage> messages = new ArrayList<>();
        if (systemPrompt != null) {
            messages.add(ChatMessage.system(systemPrompt));
        }
        messages.add(ChatMessage.user(userMessage));
        return messages;
    }

    private Quantization parseQuantization() {
        return switch (quantize.toLowerCase()) {
            case "nf4" -> Quantization.NF4;
            case "int8" -> Quantization.INT8;
            case "none" -> Quantization.NONE;
            default -> Quantization.AUTO;
        };
    }
}
