package io.github.javpower.jpyml.cli;

import io.github.javpower.jpyml.llm.LLMModel;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.util.concurrent.Callable;

@Command(name = "merge-adapter", description = "将 LoRA 适配器合并到基座模型")
public class MergeAdapterCommand implements Callable<Integer> {

    @Option(names = {"-m", "--model"}, description = "基座模型路径", required = true)
    String model;

    @Option(names = {"-a", "--adapter"}, description = "LoRA 适配器路径", required = true)
    String adapter;

    @Option(names = {"-o", "--output"}, description = "输出路径", required = true)
    String output;

    @Override
    public Integer call() throws Exception {
        CliRunner.initRuntime();

        System.out.println("开始合并适配器...");
        System.out.printf("  基座模型: %s%n", model);
        System.out.printf("  适配器: %s%n", adapter);
        System.out.printf("  输出: %s%n", output);

        try {
            String mergedPath = LLMModel.mergeAdapter(model, adapter, output);
            System.out.println("\n合并完成!");
            System.out.printf("  输出路径: %s%n", mergedPath);
            return 0;
        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            return 1;
        }
    }
}