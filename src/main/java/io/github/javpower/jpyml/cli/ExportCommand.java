package io.github.javpower.jpyml.cli;

import io.github.javpower.jpyml.ml.export.ExportConfig;
import io.github.javpower.jpyml.ml.export.ExportFormat;
import io.github.javpower.jpyml.ml.export.ExportResult;
import io.github.javpower.jpyml.ml.model.Model;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.util.concurrent.Callable;

@Command(name = "export", description = "导出 YOLO 模型为其他格式")
public class ExportCommand implements Callable<Integer> {

    @Option(names = {"-m", "--model"}, description = "模型路径（如 yolov8n.pt）", required = true)
    String model;

    @Option(names = {"-f", "--format"}, description = "导出格式（onnx, torchscript, coreml, tflite, engine, openvino, ncnn, paddle）")
    String format = "onnx";

    @Option(names = {"--half"}, description = "FP16 半精度导出")
    boolean half = false;

    @Option(names = {"--dynamic"}, description = "动态输入尺寸（ONNX）")
    boolean dynamic = false;

    @Option(names = {"--no-simplify"}, description = "禁用 ONNX 简化")
    boolean noSimplify = false;

    @Option(names = {"--opset"}, description = "ONNX opset 版本（默认: 17）")
    int opset = 17;

    @Option(names = {"--img-size"}, description = "导出图像尺寸（默认: 640）")
    int imgSize = 640;

    @Override
    public Integer call() throws Exception {
        CliRunner.initRuntime();

        System.out.println("开始模型导出...");
        System.out.printf("  模型: %s%n", model);
        System.out.printf("  格式: %s%n", format);

        try (Model m = new Model(model)) {
            ExportConfig config = new ExportConfig()
                    .format(parseFormat())
                    .half(half)
                    .dynamic(dynamic)
                    .simplify(!noSimplify)
                    .opset(opset)
                    .imgsz(imgSize);

            ExportResult result = m.export(config);

            System.out.println("\n导出完成!");
            System.out.printf("  输出路径: %s%n", result.getOutputPath());
            System.out.printf("  格式: %s%n", result.getFormat());
            return 0;
        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            return 1;
        }
    }

    private ExportFormat parseFormat() {
        return switch (format.toLowerCase()) {
            case "torchscript" -> ExportFormat.TORCHSCRIPT;
            case "onnx" -> ExportFormat.ONNX;
            case "openvino" -> ExportFormat.OPENVINO;
            case "coreml" -> ExportFormat.COREML;
            case "tflite" -> ExportFormat.TFLITE;
            case "engine", "tensorrt" -> ExportFormat.TENSORRT;
            case "ncnn" -> ExportFormat.NCNN;
            case "paddle" -> ExportFormat.PADDLE;
            default -> throw new IllegalArgumentException("不支持的导出格式: " + format);
        };
    }
}