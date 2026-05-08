package io.github.javpower.jpyml.cli;

import io.github.javpower.jpyml.ml.model.Model;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.util.concurrent.Callable;

@Command(name = "info", description = "显示模型信息")
public class InfoCommand implements Callable<Integer> {

    @Option(names = {"-m", "--model"}, description = "模型路径（如 yolov8n.pt）", required = true)
    String model;

    @Override
    public Integer call() throws Exception {
        CliRunner.initRuntime();

        try (Model m = new Model(model)) {
            System.out.println("模型信息:");
            System.out.printf("  路径: %s%n", model);
            System.out.printf("  任务类型: %s%n", m.getTaskType());
            System.out.printf("  类别数: %d%n", m.getNumClasses());
            System.out.printf("  类别名称: %s%n", m.getClassNames());
            return 0;
        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            return 1;
        }
    }
}