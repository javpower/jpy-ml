package io.github.javpower.jpyml.cli;

import io.github.javpower.jpyml.ml.model.Model;
import io.github.javpower.jpyml.ml.model.ModelConfig;
import io.github.javpower.jpyml.ml.result.InferenceResult;
import io.github.javpower.jpyml.ml.result.OBBPrediction;
import io.github.javpower.jpyml.ml.result.OBBResult;
import io.github.javpower.jpyml.ml.result.RotatedBoundingBox;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.util.List;
import java.util.concurrent.Callable;

@Command(name = "obb", description = "旋转框目标检测")
public class OBBCommand implements Callable<Integer> {

    @Option(names = {"-m", "--model"}, description = "OBB 模型路径（如 yolov8n-obb.pt）", required = true)
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
            System.out.println("模型: " + model + " | 任务: 旋转框检测 | 类别数: " + m.getNumClasses());

            ModelConfig config = new ModelConfig().confidence(conf).imageSize(imgSize);
            List<InferenceResult> results = m.predict(sources, config);

            for (int i = 0; i < results.size(); i++) {
                InferenceResult r = results.get(i);
                System.out.printf("%n[%d] %s — %d 个目标, %.1fms%n",
                        i, r.getSourcePath(), r.count(), r.getSpeed().inferenceMs());

                if (r instanceof OBBResult obr) {
                    for (OBBPrediction pred : obr.getPredictions()) {
                        RotatedBoundingBox box = pred.box();
                        System.out.printf("  %s %.1f%% 中心=(%.1f, %.1f) 尺寸=(%.1f, %.1f) 角度=%.1f°%n",
                                pred.className(), pred.confidence() * 100,
                                box.centerX(), box.centerY(), box.width(), box.height(), box.angleDegrees());
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