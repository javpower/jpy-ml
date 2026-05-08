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

@Command(name = "batch", description = "批量图像推理")
public class BatchCommand implements Callable<Integer> {

    @Option(names = {"-m", "--model"}, description = "模型路径（如 yolov8n.pt）", required = true)
    String model;

    @Option(names = {"-d", "--dir"}, description = "图像目录路径", required = true)
    String imageDir;

    @Option(names = {"--conf"}, description = "置信度阈值（默认: 0.25）")
    float conf = 0.25f;

    @Option(names = {"--img-size"}, description = "推理图像尺寸（默认: 640）")
    int imgSize = 640;

    @Option(names = {"-o", "--output"}, description = "输出目录（用于保存标注图像）")
    String outputDir;

    @Option(names = {"--ext"}, description = "图像扩展名过滤（默认: jpg,jpeg,png）")
    String ext = "jpg,jpeg,png";

    @Override
    public Integer call() throws Exception {
        CliRunner.initRuntime();

        try (Model m = new Model(model)) {
            System.out.println("模型: " + model + " | 任务: 批量推理");
            System.out.printf("  目录: %s%n", imageDir);
            System.out.printf("  扩展名: %s%n", ext);

            // 获取目录中的图像文件
            List<String> images = getImagesInDirectory(imageDir, ext);
            System.out.printf("  找到 %d 张图像%n", images.size());

            if (images.isEmpty()) {
                System.out.println("没有找到图像文件");
                return 0;
            }

            ModelConfig config = new ModelConfig().confidence(conf).imageSize(imgSize);
            ImageAnnotator annotator = (outputDir != null) ? new ImageAnnotator() : null;

            int totalObjects = 0;
            long totalTime = 0;

            for (int i = 0; i < images.size(); i++) {
                String imagePath = images.get(i);
                InferenceResult r = m.predict(imagePath, config);

                System.out.printf("[%d/%d] %s — %d 个目标 (%.1fms)%n",
                        i + 1, images.size(), imagePath, r.count(), r.getSpeed().inferenceMs());

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

                totalObjects += r.count();
                totalTime += r.getSpeed().inferenceMs();
            }

            System.out.printf("%n批量推理完成!%n");
            System.out.printf("  总图像数: %d%n", images.size());
            System.out.printf("  总目标数: %d%n", totalObjects);
            System.out.printf("  平均耗时: %.1f ms/张%n", totalTime / images.size());
            return 0;
        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            return 1;
        }
    }

    private List<String> getImagesInDirectory(String dir, String extensions) {
        java.io.File directory = new java.io.File(dir);
        if (!directory.exists() || !directory.isDirectory()) {
            return List.of();
        }

        String[] extArray = extensions.split(",");
        return java.util.Arrays.stream(directory.listFiles())
                .filter(f -> f.isFile())
                .filter(f -> {
                    String name = f.getName().toLowerCase();
                    for (String ext : extArray) {
                        if (name.endsWith("." + ext.trim().toLowerCase())) {
                            return true;
                        }
                    }
                    return false;
                })
                .map(java.io.File::getAbsolutePath)
                .sorted()
                .toList();
    }
}