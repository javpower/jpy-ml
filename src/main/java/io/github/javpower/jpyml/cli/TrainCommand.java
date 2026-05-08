package io.github.javpower.jpyml.cli;

import io.github.javpower.jpyml.llm.LLMModel;
import io.github.javpower.jpyml.llm.LLMTrainingResult;
import io.github.javpower.jpyml.llm.config.LLMTrainConfig;
import io.github.javpower.jpyml.llm.config.LoRAConfig;
import io.github.javpower.jpyml.ml.training.TrainingCallback;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.util.concurrent.Callable;

@Command(name = "train", description = "使用 LoRA/QLoRA 微调 LLM")
public class TrainCommand implements Callable<Integer> {

    @Option(names = {"-m", "--model"}, description = "基础模型路径或 HuggingFace ID", required = true)
    String model;

    @Option(names = {"-d", "--dataset"}, description = "训练数据集路径（JSONL）", required = true)
    String dataset;

    @Option(names = {"--epochs"}, description = "训练轮数（默认: 3）")
    int epochs = 3;

    @Option(names = {"--rank"}, description = "LoRA rank（默认: 16）")
    int rank = 16;

    @Option(names = {"--alpha"}, description = "LoRA alpha（默认: 32）")
    int alpha = 32;

    @Option(names = {"--batch-size"}, description = "批次大小（默认: 4）")
    int batchSize = 4;

    @Option(names = {"--lr"}, description = "学习率（默认: 2e-4）")
    double learningRate = 2e-4;

    @Option(names = {"--max-seq-length"}, description = "最大序列长度（默认: 2048）")
    int maxSeqLength = 2048;

    @Option(names = {"--warmup-steps"}, description = "预热步数（默认: 100）")
    int warmupSteps = 100;

    @Option(names = {"--output"}, description = "输出目录")
    String outputDir;

    @Override
    public Integer call() throws Exception {
        CliRunner.initRuntime();

        LLMModel lm;
        if (model.contains("/")) {
            lm = LLMModel.download(model, progress -> System.err.println("  " + progress));
        } else {
            lm = LLMModel.load(model);
        }

        LoRAConfig loraConfig = LoRAConfig.create().rank(rank).alpha(alpha);
        LLMTrainConfig trainConfig = LLMTrainConfig.create()
                .epochs(epochs)
                .batchSize(batchSize)
                .learningRate(learningRate)
                .maxSeqLength(maxSeqLength)
                .warmupSteps(warmupSteps);
        if (outputDir != null) {
            trainConfig.outputDir(outputDir);
        }

        System.out.println("开始微调...");
        System.out.printf("  模型: %s%n", model);
        System.out.printf("  数据集: %s%n", dataset);
        System.out.printf("  LoRA rank=%d alpha=%d%n", rank, alpha);
        System.out.printf("  轮数=%d 批次=%d 学习率=%.1e%n", epochs, batchSize, learningRate);

        try {
            LLMTrainingResult result = lm.fineTune()
                    .lora(loraConfig)
                    .dataset(dataset)
                    .config(trainConfig)
                    .run(new TrainingCallback() {
                        @Override
                        public void onEpoch(int epoch, String logLine) {
                            System.out.printf("  Epoch %d: %s%n", epoch, logLine);
                        }

                        @Override
                        public void onComplete(String error) {
                            if (error != null) {
                            System.err.println("训练错误: " + error);
                            }
                        }
                    });

            System.out.println("\n训练完成!");
            System.out.printf("  适配器: %s%n", result.getAdapterPath());
            System.out.printf("  最终 loss: %.4f%n", result.getFinalLoss());
            System.out.printf("  日志条目: %d%n", result.getLog().size());
            return 0;
        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            return 1;
        }
    }
}
