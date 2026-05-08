package io.github.javpower.jpyml.cli;

import io.github.javpower.jpyml.ml.model.ModelHub;
import picocli.CommandLine.Command;

import java.util.concurrent.Callable;

@Command(name = "list-models", description = "列出所有可用模型", mixinStandardHelpOptions = true)
public class ListModelsCommand implements Callable<Integer> {

    @Override
    public Integer call() throws Exception {
        CliRunner.initRuntime();

        System.out.println("可用模型:");
        System.out.println("--------");

        var models = ModelHub.listAvailable();
        for (var model : models) {
            String cached = ModelHub.isCached(model.getName()) ? " [已缓存]" : "";
            System.out.printf("  %-20s %s (%s, %s)%n",
                    model.getName(),
                    model.getSize(),
                    model.getTask() != null ? model.getTask() : "SAM",
                    cached);
        }

        System.out.printf("%n共 %d 个模型%n", models.size());
        return 0;
    }
}