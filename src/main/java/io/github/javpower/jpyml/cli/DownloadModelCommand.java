package io.github.javpower.jpyml.cli;

import io.github.javpower.jpyml.ml.model.ModelHub;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.util.concurrent.Callable;

@Command(name = "download", description = "下载模型")
public class DownloadModelCommand implements Callable<Integer> {

    @Option(names = {"-n", "--name"}, description = "模型名称（如 yolov8n.pt）", required = true)
    String modelName;

    @Override
    public Integer call() throws Exception {
        CliRunner.initRuntime();

        System.out.println("开始下载模型...");
        System.out.printf("  模型: %s%n", modelName);

        try {
            java.nio.file.Path modelPath = ModelHub.download(modelName);
            System.out.println("\n下载完成!");
            System.out.printf("  路径: %s%n", modelPath);
            return 0;
        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            return 1;
        }
    }
}