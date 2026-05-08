package io.github.javpower.jpyml.cli;

import io.github.javpower.jpyml.core.PythonRuntime;
import picocli.CommandLine;
import picocli.CommandLine.Command;

import java.nio.file.Files;
import java.nio.file.Path;

@Command(
        name = "jpy-ml",
        description = "Java ML 框架 - 将 Python ML 生态系统桥接到 Java",
        version = "jpy-ml 1.4.0",
        subcommands = {
                PredictCommand.class,
                SegmentCommand.class,
                ChatCommand.class,
                TrainCommand.class,
                YoloTrainCommand.class,
                ValidateCommand.class,
                ExportCommand.class,
                ClassifyCommand.class,
                PoseCommand.class,
                OBBCommand.class,
                VideoCommand.class,
                InfoCommand.class,
                BatchCommand.class,
                MergeAdapterCommand.class,
                DownloadModelCommand.class,
                ListModelsCommand.class,
                GenerateCommand.class,
        },
        mixinStandardHelpOptions = true
)
public class CliRunner implements Runnable {

    private static volatile boolean initialized = false;

    @Override
    public void run() {
        CommandLine.usage(this, System.out);
    }

    public static void main(String[] args) {
        int exitCode = new CommandLine(new CliRunner()).execute(args);
        System.exit(exitCode);
    }

    static void initRuntime() throws Exception {
        if (initialized) return;
        synchronized (CliRunner.class) {
            if (initialized) return;

            Path cwd = Path.of("").toAbsolutePath();
            Path pythonBin = cwd.resolve(".venv/bin/python3");
            Path jepLib = cwd.resolve(".venv/lib/python3.12/site-packages/jep/libjep.jnilib");

            if (Files.exists(pythonBin) && Files.exists(jepLib)) {
                PythonRuntime.init(pythonBin, jepLib);
            } else {
                PythonRuntime.init();
            }
            initialized = true;
        }
    }
}
