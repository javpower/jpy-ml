package io.github.javpower.jpyml.cli;

import io.github.javpower.jpyml.ml.model.Device;
import io.github.javpower.jpyml.ml.model.Model;
import io.github.javpower.jpyml.ml.training.TrainingCallback;
import io.github.javpower.jpyml.ml.training.TrainingConfig;
import io.github.javpower.jpyml.ml.training.TrainingResult;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.util.concurrent.Callable;

@Command(name = "yolo-train", description = "训练 YOLO 模型（检测、分割、分类、姿态、OBB）")
public class YoloTrainCommand implements Callable<Integer> {

    @Option(names = {"-m", "--model"}, description = "基础模型路径（如 yolov8n.pt）", required = true)
    String model;

    @Option(names = {"-d", "--data"}, description = "数据配置文件路径（如 coco128.yaml）", required = true)
    String dataConfig;

    @Option(names = {"--epochs"}, description = "训练轮数（默认: 100）")
    int epochs = 100;

    @Option(names = {"--batch-size"}, description = "批次大小（默认: 16）")
    int batchSize = 16;

    @Option(names = {"--img-size"}, description = "输入图像尺寸（默认: 640）")
    int imgSize = 640;

    @Option(names = {"--device"}, description = "训练设备（如 cpu, 0, 0,1, mps）")
    String device;

    @Option(names = {"--workers"}, description = "数据加载线程数（默认: 8）")
    int workers = 8;

    @Option(names = {"--optimizer"}, description = "优化器（SGD, Adam, AdamW, AUTO）")
    String optimizer = "AUTO";

    @Option(names = {"--lr"}, description = "初始学习率（默认: 0.01）")
    float learningRate = 0.01f;

    @Option(names = {"--momentum"}, description = "SGD 动量（默认: 0.937）")
    float momentum = 0.937f;

    @Option(names = {"--weight-decay"}, description = "权重衰减（默认: 0.0005）")
    float weightDecay = 0.0005f;

    @Option(names = {"--patience"}, description = "早停耐心值（默认: 100）")
    int patience = 100;

    @Option(names = {"--seed"}, description = "随机种子（默认: 0）")
    int seed = 0;

    @Option(names = {"--no-amp"}, description = "禁用混合精度训练")
    boolean noAmp = false;

    @Option(names = {"--no-val"}, description = "禁用验证")
    boolean noVal = false;

    @Option(names = {"--no-plots"}, description = "禁用训练图表")
    boolean noPlots = false;

    @Option(names = {"--project"}, description = "项目目录（默认: runs/train）")
    String project = "runs/train";

    @Option(names = {"--name"}, description = "实验名称（默认: yolo_train）")
    String name = "yolo_train";

    @Option(names = {"--resume"}, description = "恢复训练")
    boolean resume = false;

    @Override
    public Integer call() throws Exception {
        CliRunner.initRuntime();

        System.out.println("开始 YOLO 模型训练...");
        System.out.printf("  模型: %s%n", model);
        System.out.printf("  数据: %s%n", dataConfig);
        System.out.printf("  轮数: %d, 批次: %d, 图像尺寸: %d%n", epochs, batchSize, imgSize);
        if (device != null) {
            System.out.printf("  设备: %s%n", device);
        }

        try (Model m = new Model(model)) {
            TrainingConfig config = new TrainingConfig()
                    .dataConfig(dataConfig)
                    .epochs(epochs)
                    .batchSize(batchSize)
                    .imageSize(imgSize)
                    .workers(workers)
                    .optimizer(parseOptimizer())
                    .initialLR(learningRate)
                    .momentum(momentum)
                    .weightDecay(weightDecay)
                    .patience(patience)
                    .seed(seed)
                    .amp(!noAmp)
                    .val(!noVal)
                    .plots(!noPlots)
                    .project(project)
                    .name(name)
                    .resume(resume);

            if (device != null) {
                config.device(device);
            }

            TrainingResult result = m.train(config, new TrainingCallback() {
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
            System.out.printf("  结果目录: %s/%s%n", project, name);
            System.out.printf("  Epoch 指标数: %d%n", result.getEpochMetrics().size());
            return 0;
        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            return 1;
        }
    }

    private io.github.javpower.jpyml.ml.training.OptimizerType parseOptimizer() {
        return switch (optimizer.toUpperCase()) {
            case "SGD" -> io.github.javpower.jpyml.ml.training.OptimizerType.SGD;
            case "ADAM" -> io.github.javpower.jpyml.ml.training.OptimizerType.ADAM;
            case "ADAMW" -> io.github.javpower.jpyml.ml.training.OptimizerType.ADAMW;
            default -> io.github.javpower.jpyml.ml.training.OptimizerType.AUTO;
        };
    }
}