package io.github.javpower.jpyml.cli;

import io.github.javpower.jpyml.ml.model.Prompt;
import io.github.javpower.jpyml.ml.model.SAM2Model;
import io.github.javpower.jpyml.ml.model.SAM3Model;
import io.github.javpower.jpyml.ml.result.Mask;
import io.github.javpower.jpyml.ml.result.SAM2Result;
import io.github.javpower.jpyml.ml.result.SAM3Result;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

@Command(name = "segment", description = "运行 SAM2/SAM3 图像分割")
public class SegmentCommand implements Callable<Integer> {

    @Option(names = {"-m", "--model"}, description = "SAM 模型路径（如 sam2.1_hiera_base.pt）", required = true)
    String model;

    @Option(names = {"-i", "--image"}, description = "输入图像路径", required = true)
    String image;

    @Option(names = {"-p", "--point"}, description = "点提示 x,y（可重复）")
    List<String> points;

    @Option(names = {"-b", "--box"}, description = "框提示 x1,y1,x2,y2（可重复）")
    List<String> boxes;

    @Option(names = {"--text"}, description = "文本提示（SAM3，可重复）")
    List<String> texts;

    @Option(names = {"-o", "--output"}, description = "输出图像路径（用于掩膜可视化）")
    String output;

    @Option(names = {"--sam3"}, description = "使用 SAM3 模型")
    boolean sam3;

    @Override
    public Integer call() throws Exception {
        CliRunner.initRuntime();
        try {
            if (sam3) {
                return runSAM3();
            }
            return runSAM2();
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            return 1;
        }
    }

    private int runSAM2() throws Exception {
        try (SAM2Model sam = new SAM2Model(model)) {
            List<Prompt> prompts = buildPrompts();
            if (prompts.isEmpty()) {
                System.err.println("错误: SAM2 需要至少一个 --point 或 --box");
                return 1;
            }

            SAM2Result result = sam.predict(image, prompts.toArray(new Prompt[0]));
            printResult(result.sourcePath(), result.masks(), result.scores());
            return 0;
        }
    }

    private int runSAM3() throws Exception {
        try (SAM3Model sam = new SAM3Model(model)) {
            if (texts != null && !texts.isEmpty()) {
                SAM3Result result = sam.predictText(image, texts.toArray(new String[0]));
                printResult(result.sourcePath(), result.masks(), result.scores());
            } else {
                List<Prompt> prompts = buildPrompts();
                if (prompts.isEmpty()) {
                    System.err.println("错误: SAM3 需要至少一个 --point、--box 或 --text");
                    return 1;
                }
                System.err.println("注意: SAM3 使用点/框提示需要参考图像，请使用 --text 进行文本分割。");
                return 1;
            }
            return 0;
        }
    }

    private List<Prompt> buildPrompts() {
        List<Prompt> prompts = new ArrayList<>();
        if (points != null) {
            for (String p : points) {
                String[] parts = p.split(",");
                prompts.add(Prompt.point(Integer.parseInt(parts[0].trim()), Integer.parseInt(parts[1].trim())));
            }
        }
        if (boxes != null) {
            for (String b : boxes) {
                String[] parts = b.split(",");
                prompts.add(Prompt.box(
                        Integer.parseInt(parts[0].trim()), Integer.parseInt(parts[1].trim()),
                        Integer.parseInt(parts[2].trim()), Integer.parseInt(parts[3].trim())
                ));
            }
        }
        return prompts;
    }

    private void printResult(String sourcePath, List<Mask> masks, List<Float> scores) {
        System.out.printf("图像: %s — %d 个掩膜%n", sourcePath, masks.size());
        for (int i = 0; i < masks.size(); i++) {
            System.out.printf("  掩膜 %d: 置信度=%.4f 点数=%d%n", i, scores.get(i), masks.get(i).getPointCount());
        }
        if (output != null && !masks.isEmpty()) {
            System.out.println("输出: " + output + " (CLI 掩膜保存尚未实现)");
        }
    }
}
