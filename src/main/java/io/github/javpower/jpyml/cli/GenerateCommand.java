package io.github.javpower.jpyml.cli;

import io.github.javpower.jpyml.flux.FluxConfig;
import io.github.javpower.jpyml.flux.FluxModel;
import io.github.javpower.jpyml.flux.FluxResult;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.util.concurrent.Callable;

@Command(name = "generate", description = "FLUX.1 文本生成图像")
public class GenerateCommand implements Callable<Integer> {

    @Option(names = {"-p", "--prompt"}, description = "文本提示", required = true)
    String prompt;

    @Option(names = {"-o", "--output"}, description = "输出图像路径", required = true)
    String output;

    @Option(names = {"-m", "--model"}, description = "模型 ID（默认: schnell）")
    String model = "schnell";

    @Option(names = {"--negative"}, description = "负面提示")
    String negativePrompt;

    @Option(names = {"--width"}, description = "图像宽度（默认: 1024）")
    int width = 1024;

    @Option(names = {"--height"}, description = "图像高度（默认: 1024）")
    int height = 1024;

    @Option(names = {"--steps"}, description = "推理步数")
    int steps = -1;

    @Option(names = {"--guidance"}, description = "引导强度")
    float guidance = -1;

    @Option(names = {"--seed"}, description = "随机种子（默认: -1 随机）")
    long seed = -1;

    @Option(names = {"--device"}, description = "设备（auto, cuda, mps, cpu）")
    String device = "auto";

    @Option(names = {"--dtype"}, description = "数据类型（auto, float16, bfloat16）")
    String dtype = "auto";

    @Override
    public Integer call() throws Exception {
        CliRunner.initRuntime();

        String modelId = resolveModelId(model);

        FluxConfig config = new FluxConfig()
                .modelId(modelId)
                .device(device)
                .dtype(dtype)
                .width(width)
                .height(height)
                .seed(seed);

        if (negativePrompt != null) {
            config.negativePrompt(negativePrompt);
        }

        // Schnell 默认 4 步，Dev 默认 20 步
        if (steps >= 0) {
            config.steps(steps);
        } else if (modelId.contains("schnell")) {
            config.steps(4);
        } else {
            config.steps(20);
        }

        if (guidance >= 0) {
            config.guidance(guidance);
        } else if (modelId.contains("schnell")) {
            config.guidance(0.0f);
        } else {
            config.guidance(3.5f);
        }

        System.out.println("FLUX.1 图像生成");
        System.out.printf("  模型: %s%n", modelId);
        System.out.printf("  提示: %s%n", prompt);
        System.out.printf("  尺寸: %dx%d%n", width, height);
        System.out.printf("  步数: %d, 引导: %.1f%n", config.getSteps(), config.getGuidance());
        if (seed >= 0) {
            System.out.printf("  种子: %d%n", seed);
        }
        System.out.printf("  输出: %s%n", output);
        System.out.println("  生成中...");

        try (FluxModel flux = new FluxModel(modelId, device, dtype, null)) {
            FluxResult result = flux.generate(prompt, output, config);

            if (result.isSuccess()) {
                System.out.println("\n生成完成!");
                System.out.printf("  输出: %s%n", result.getFirstOutputPath());
                System.out.printf("  耗时: %.2f 秒%n", result.getElapsedSeconds());
                if (result.getSeed() != null) {
                    System.out.printf("  种子: %d%n", result.getSeed());
                }
                return 0;
            } else {
                System.err.println("错误: " + result.getError());
                return 1;
            }
        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            return 1;
        }
    }

    private String resolveModelId(String name) {
        return switch (name.toLowerCase()) {
            case "schnell", "flux-schnell", "flux.1-schnell" ->
                    "black-forest-labs/FLUX.1-schnell";
            case "dev", "flux-dev", "flux.1-dev" ->
                    "black-forest-labs/FLUX.1-dev";
            default -> name; // 允许直接传 HuggingFace ID
        };
    }
}
