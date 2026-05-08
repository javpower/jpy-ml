package io.github.javpower.jpyml.cli;

import io.github.javpower.jpyml.ml.model.Model;
import io.github.javpower.jpyml.ml.validation.PerClassMetric;
import io.github.javpower.jpyml.ml.validation.ValidationResult;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.util.concurrent.Callable;

@Command(name = "validate", description = "验证 YOLO 模型性能")
public class ValidateCommand implements Callable<Integer> {

    @Option(names = {"-m", "--model"}, description = "模型路径（如 yolov8n.pt）", required = true)
    String model;

    @Option(names = {"-d", "--data"}, description = "数据配置文件路径（如 coco128.yaml）")
    String dataConfig;

    @Option(names = {"--batch-size"}, description = "批次大小（默认: 16）")
    int batchSize = 16;

    @Option(names = {"--img-size"}, description = "输入图像尺寸（默认: 640）")
    int imgSize = 640;

    @Option(names = {"--device"}, description = "验证设备（如 cpu, 0, mps）")
    String device;

    @Option(names = {"--workers"}, description = "数据加载线程数（默认: 8）")
    int workers = 8;

    @Override
    public Integer call() throws Exception {
        CliRunner.initRuntime();

        System.out.println("开始模型验证...");
        System.out.printf("  模型: %s%n", model);
        if (dataConfig != null) {
            System.out.printf("  数据: %s%n", dataConfig);
        }

        try (Model m = new Model(model)) {
            ValidationResult result;
            if (dataConfig != null) {
                result = m.validate(dataConfig);
            } else {
                result = m.validate();
            }

            System.out.println("\n验证结果:");
            System.out.printf("  mAP50: %.4f%n", result.getMAP50());
            System.out.printf("  mAP50-95: %.4f%n", result.getMAP5095());
            System.out.printf("  精确率: %.4f%n", result.getPrecision());
            System.out.printf("  召回率: %.4f%n", result.getRecall());

            System.out.println("\n各类别指标:");
            for (PerClassMetric pc : result.getPerClassMetrics()) {
                System.out.printf("  %s (id=%d): mAP50-95=%.4f%n",
                        pc.className(), pc.classId(), pc.map5095());
            }

            return 0;
        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            return 1;
        }
    }
}