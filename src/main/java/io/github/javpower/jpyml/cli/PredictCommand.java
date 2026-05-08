package io.github.javpower.jpyml.cli;

import io.github.javpower.jpyml.ml.annotation.ImageAnnotator;
import io.github.javpower.jpyml.ml.model.Model;
import io.github.javpower.jpyml.ml.model.ModelConfig;
import io.github.javpower.jpyml.ml.result.ClassPrediction;
import io.github.javpower.jpyml.ml.result.DetectionResult;
import io.github.javpower.jpyml.ml.result.InferenceResult;
import io.github.javpower.jpyml.ml.result.SegmentationResult;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.util.List;
import java.util.concurrent.Callable;

@Command(name = "predict", description = "运行目标检测/分割/分类")
public class PredictCommand implements Callable<Integer> {

    @Option(names = {"-m", "--model"}, description = "模型路径（如 yolov8n.pt）", required = true)
    String model;

    @Option(names = {"-s", "--source"}, description = "图像路径或 URL（可重复）", required = true, arity = "1..*")
    List<String> sources;

    @Option(names = {"-t", "--task"}, description = "任务类型: detect, segment, classify, pose, obb")
    String task;

    @Option(names = {"--conf"}, description = "置信度阈值（默认: 0.25）")
    float conf = 0.25f;

    @Option(names = {"--img-size"}, description = "推理图像尺寸（默认: 640）")
    int imgSize = 640;

    @Option(names = {"-o", "--output"}, description = "输出目录（用于保存标注图像）")
    String outputDir;

    @Override
    public Integer call() throws Exception {
        CliRunner.initRuntime();
        try (Model m = new Model(model)) {
            System.out.println("模型: " + model + " | 任务: " + m.getTaskType() + " | 类别数: " + m.getNumClasses());

            ModelConfig config = new ModelConfig().confidence(conf).imageSize(imgSize);
            List<InferenceResult> results = m.predict(sources, config);
            ImageAnnotator annotator = (outputDir != null) ? new ImageAnnotator() : null;

            for (int i = 0; i < results.size(); i++) {
                InferenceResult r = results.get(i);
                System.out.printf("%n[%d] %s — %d 个目标 (%.1fms)%n", i, r.getSourcePath(), r.count(), r.getSpeed().inferenceMs());

                if (r instanceof DetectionResult dr) {
                    for (ClassPrediction pred : dr.getBoxes()) {
                        System.out.printf("  %s %.2f%% [%.0f,%.0f,%.0f,%.0f]%n",
                                pred.className(), pred.confidence() * 100,
                                pred.box().x1(), pred.box().y1(), pred.box().x2(), pred.box().y2());
                    }
                } else if (r instanceof SegmentationResult sr) {
                    for (int j = 0; j < sr.getBoxes().size(); j++) {
                        ClassPrediction pred = sr.getBoxes().get(j);
                        System.out.printf("  %s %.2f%% (mask: %d points)%n",
                                pred.className(), pred.confidence() * 100,
                                sr.getMasks().get(j).getPointCount());
                    }
                }

                if (annotator != null) {
                    String path = annotator.annotate(r, outputDir + "/result_" + i + ".jpg");
                    System.out.println("  保存: " + path);
                }
            }
            return 0;
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            return 1;
        }
    }
}
