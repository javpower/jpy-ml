package io.github.javpower.jpyml.cli;

import io.github.javpower.jpyml.ml.model.Model;
import io.github.javpower.jpyml.ml.model.ModelConfig;
import io.github.javpower.jpyml.ml.result.DetectionResult;
import io.github.javpower.jpyml.ml.result.InferenceResult;
import io.github.javpower.jpyml.ml.result.StreamFrame;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;

@Command(name = "video", description = "视频/摄像头实时推理")
public class VideoCommand implements Callable<Integer> {

    @Option(names = {"-m", "--model"}, description = "模型路径（如 yolov8n.pt）", required = true)
    String model;

    @Option(names = {"-s", "--source"}, description = "视频路径或摄像头索引（默认: 0）")
    String source = "0";

    @Option(names = {"--conf"}, description = "置信度阈值（默认: 0.25）")
    float conf = 0.25f;

    @Option(names = {"--img-size"}, description = "推理图像尺寸（默认: 640）")
    int imgSize = 640;

    @Option(names = {"--max-frames"}, description = "最大处理帧数（默认: 0=无限）")
    int maxFrames = 0;

    @Option(names = {"--show-fps"}, description = "显示 FPS 信息")
    boolean showFps = false;

    @Override
    public Integer call() throws Exception {
        CliRunner.initRuntime();

        try (Model m = new Model(model)) {
            System.out.println("模型: " + model + " | 任务: 视频推理");
            System.out.printf("  来源: %s%n", source);
            System.out.printf("  置信度: %.2f, 图像尺寸: %d%n", conf, imgSize);

            ModelConfig config = new ModelConfig().confidence(conf).imageSize(imgSize);
            AtomicInteger frameCount = new AtomicInteger(0);
            long startTime = System.currentTimeMillis();

            // 判断是摄像头还是视频文件
            boolean isCamera = source.matches("\\d+");
            if (isCamera) {
                int cameraIndex = Integer.parseInt(source);
                System.out.println("启动摄像头推理... (按 Ctrl+C 停止)");
                m.predictStream(cameraIndex, config, frame -> {
                    processStreamFrame(frame, frameCount, startTime, showFps);
                    if (maxFrames > 0 && frameCount.get() >= maxFrames) {
                        m.stopStream();
                    }
                });
            } else {
                System.out.println("开始视频推理...");
                m.predictVideo(source, config, frame -> {
                    processInferenceFrame(frame, frameCount, startTime, showFps);
                    if (maxFrames > 0 && frameCount.get() >= maxFrames) {
                        m.stopStream();
                    }
                });
            }

            long endTime = System.currentTimeMillis();
            System.out.printf("%n处理完成! 总帧数: %d, 耗时: %.1f 秒%n",
                    frameCount.get(), (endTime - startTime) / 1000.0);
            return 0;
        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            return 1;
        }
    }

    private void processStreamFrame(StreamFrame frame, AtomicInteger frameCount, long startTime, boolean showFps) {
        int count = frameCount.incrementAndGet();
        InferenceResult result = frame.getResult();
        if (showFps && count % 10 == 0) {
            long elapsed = System.currentTimeMillis() - startTime;
            double fps = count * 1000.0 / elapsed;
            System.out.printf("  帧 %d: %d 个目标, FPS: %.1f%n", count, result.count(), fps);
        } else {
            System.out.printf("  帧 %d: %d 个目标%n", count, result.count());
        }
    }

    private void processInferenceFrame(InferenceResult frame, AtomicInteger frameCount, long startTime, boolean showFps) {
        int count = frameCount.incrementAndGet();
        if (showFps && count % 10 == 0) {
            long elapsed = System.currentTimeMillis() - startTime;
            double fps = count * 1000.0 / elapsed;
            System.out.printf("  帧 %d: %d 个目标, FPS: %.1f%n", count, frame.count(), fps);
        } else {
            System.out.printf("  帧 %d: %d 个目标%n", count, frame.count());
        }
    }
}