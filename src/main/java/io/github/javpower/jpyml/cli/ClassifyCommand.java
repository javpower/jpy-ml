package io.github.javpower.jpyml.cli;

import io.github.javpower.jpyml.ml.model.Model;
import io.github.javpower.jpyml.ml.model.ModelConfig;
import io.github.javpower.jpyml.ml.result.ClassificationResult;
import io.github.javpower.jpyml.ml.result.InferenceResult;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.util.List;
import java.util.concurrent.Callable;

@Command(name = "classify", description = "图像分类")
public class ClassifyCommand implements Callable<Integer> {

    @Option(names = {"-m", "--model"}, description = "分类模型路径（如 yolov8n-cls.pt）", required = true)
    String model;

    @Option(names = {"-s", "--source"}, description = "图像路径或 URL（可重复）", required = true, arity = "1..*")
    List<String> sources;

    @Option(names = {"--conf"}, description = "置信度阈值（默认: 0.25）")
    float conf = 0.25f;

    @Option(names = {"--img-size"}, description = "推理图像尺寸（默认: 224）")
    int imgSize = 224;

    @Option(names = {"--top-k"}, description = "显示前 K 个结果（默认: 5）")
    int topK = 5;

    @Override
    public Integer call() throws Exception {
        CliRunner.initRuntime();

        try (Model m = new Model(model)) {
            System.out.println("模型: " + model + " | 任务: 分类 | 类别数: " + m.getNumClasses());

            ModelConfig config = new ModelConfig().confidence(conf).imageSize(imgSize);
            List<InferenceResult> results = m.predict(sources, config);

            for (int i = 0; i < results.size(); i++) {
                InferenceResult r = results.get(i);
                System.out.printf("%n[%d] %s — %.1fms%n", i, r.getSourcePath(), r.getSpeed().inferenceMs());

                if (r instanceof ClassificationResult cr) {
                    System.out.printf("  Top-1: %s (%.2f%%)%n", cr.getTop1ClassName(), cr.getTop1Confidence() * 100);
                    // 显示更多结果
                    var allPreds = cr.getTopK();
                    for (int j = 0; j < Math.min(topK, allPreds.size()); j++) {
                        var pred = allPreds.get(j);
                        System.out.printf("  %d. %s (%.2f%%)%n", j + 1, pred.className(), pred.confidence() * 100);
                    }
                }
            }
            return 0;
        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            return 1;
        }
    }
}