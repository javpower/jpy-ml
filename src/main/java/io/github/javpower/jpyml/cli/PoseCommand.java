package io.github.javpower.jpyml.cli;

import io.github.javpower.jpyml.ml.model.Model;
import io.github.javpower.jpyml.ml.model.ModelConfig;
import io.github.javpower.jpyml.ml.result.InferenceResult;
import io.github.javpower.jpyml.ml.result.Keypoint;
import io.github.javpower.jpyml.ml.result.KeypointCollection;
import io.github.javpower.jpyml.ml.result.PoseResult;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.util.List;
import java.util.concurrent.Callable;

@Command(name = "pose", description = "人体姿态估计")
public class PoseCommand implements Callable<Integer> {

    @Option(names = {"-m", "--model"}, description = "姿态模型路径（如 yolov8n-pose.pt）", required = true)
    String model;

    @Option(names = {"-s", "--source"}, description = "图像路径或 URL（可重复）", required = true, arity = "1..*")
    List<String> sources;

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
            System.out.println("模型: " + model + " | 任务: 姿态估计 | 类别数: " + m.getNumClasses());

            ModelConfig config = new ModelConfig().confidence(conf).imageSize(imgSize);
            List<InferenceResult> results = m.predict(sources, config);

            for (int i = 0; i < results.size(); i++) {
                InferenceResult r = results.get(i);
                System.out.printf("%n[%d] %s — %d 人, %.1fms%n",
                        i, r.getSourcePath(), r.count(), r.getSpeed().inferenceMs());

                if (r instanceof PoseResult pr) {
                    for (int j = 0; j < pr.personCount(); j++) {
                        KeypointCollection kpts = pr.getKeypoints().get(j);
                        System.out.printf("  人物 %d:%n", j + 1);
                        System.out.printf("    鼻子: (%.1f, %.1f) conf=%.2f%n",
                                kpts.getNose().x(), kpts.getNose().y(), kpts.getNose().confidence());
                        System.out.printf("    左眼: (%.1f, %.1f) conf=%.2f%n",
                                kpts.getLeftEye().x(), kpts.getLeftEye().y(), kpts.getLeftEye().confidence());
                        System.out.printf("    右眼: (%.1f, %.1f) conf=%.2f%n",
                                kpts.getRightEye().x(), kpts.getRightEye().y(), kpts.getRightEye().confidence());
                        // COCO 索引: 5=左肩, 6=右肩
                        if (kpts.size() > 6) {
                            Keypoint leftShoulder = kpts.get(5);
                            Keypoint rightShoulder = kpts.get(6);
                            System.out.printf("    左肩: (%.1f, %.1f) conf=%.2f%n",
                                    leftShoulder.x(), leftShoulder.y(), leftShoulder.confidence());
                            System.out.printf("    右肩: (%.1f, %.1f) conf=%.2f%n",
                                    rightShoulder.x(), rightShoulder.y(), rightShoulder.confidence());
                        }
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