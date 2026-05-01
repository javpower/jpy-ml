package io.github.javpower.jpyml.ml.training;

import java.util.List;
import java.util.Map;

public class TrainingResult {
    private final String bestModelPath;
    private final String lastModelPath;
    private final String saveDirectory;
    private final int completedEpochs;
    private final float bestFitness;
    private final List<EpochMetric> epochMetrics;

    public TrainingResult(String bestModelPath, String lastModelPath, String saveDirectory,
                          int completedEpochs, float bestFitness) {
        this(bestModelPath, lastModelPath, saveDirectory, completedEpochs, bestFitness, List.of());
    }

    public TrainingResult(String bestModelPath, String lastModelPath, String saveDirectory,
                          int completedEpochs, float bestFitness,
                          List<Map<String, Object>> rawEpochLog) {
        this.bestModelPath = bestModelPath;
        this.lastModelPath = lastModelPath;
        this.saveDirectory = saveDirectory;
        this.completedEpochs = completedEpochs;
        this.bestFitness = bestFitness;
        this.epochMetrics = parseEpochMetrics(rawEpochLog);
    }

    public String getBestModelPath() { return bestModelPath; }
    public String getLastModelPath() { return lastModelPath; }
    public String getSaveDirectory() { return saveDirectory; }
    public int getCompletedEpochs() { return completedEpochs; }
    public float getBestFitness() { return bestFitness; }
    public List<EpochMetric> getEpochMetrics() { return epochMetrics; }

    private static List<EpochMetric> parseEpochMetrics(List<Map<String, Object>> raw) {
        if (raw == null) return List.of();
        return raw.stream().map(m -> new EpochMetric(
            ((Number) m.getOrDefault("epoch", 0)).intValue(),
            ((Number) m.getOrDefault("box_loss", 0.0)).floatValue(),
            ((Number) m.getOrDefault("cls_loss", 0.0)).floatValue(),
            ((Number) m.getOrDefault("dfl_loss", 0.0)).floatValue(),
            ((Number) m.getOrDefault("fitness", 0.0)).floatValue()
        )).toList();
    }

    @Override
    public String toString() {
        return String.format("TrainingResult{best='%s', epochs=%d, fitness=%.4f, metrics=%d}",
                bestModelPath, completedEpochs, bestFitness, epochMetrics.size());
    }
}
