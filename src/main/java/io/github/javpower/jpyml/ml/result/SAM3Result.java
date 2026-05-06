package io.github.javpower.jpyml.ml.result;

import io.github.javpower.jpyml.exception.InferenceException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Result of SAM 3 concept-level segmentation.
 * Contains masks, confidence scores, and optional class IDs.
 *
 * @param sourcePath path to the source image
 * @param masks      list of segmentation masks (polygons)
 * @param scores     confidence scores for each mask
 * @param classIds   class IDs for each mask (from text prompts)
 */
public record SAM3Result(
        String sourcePath,
        List<Mask> masks,
        List<Float> scores,
        List<Integer> classIds
) {
    public SAM3Result {
        if (masks.size() != scores.size()) {
            throw new IllegalArgumentException(
                    "masks and scores must have same size: " + masks.size() + " vs " + scores.size());
        }
        if (masks.size() != classIds.size()) {
            throw new IllegalArgumentException(
                    "masks and classIds must have same size: " + masks.size() + " vs " + classIds.size());
        }
        masks = Collections.unmodifiableList(masks);
        scores = Collections.unmodifiableList(scores);
        classIds = Collections.unmodifiableList(classIds);
    }

    /**
     * Get the number of masks.
     */
    public int count() {
        return masks.size();
    }

    /**
     * Get the mask with the highest score.
     */
    public Mask bestMask() {
        if (masks.isEmpty()) {
            throw new InferenceException("No masks in result");
        }
        int bestIdx = 0;
        float bestScore = scores.get(0);
        for (int i = 1; i < scores.size(); i++) {
            if (scores.get(i) > bestScore) {
                bestScore = scores.get(i);
                bestIdx = i;
            }
        }
        return masks.get(bestIdx);
    }

    /**
     * Get the best score.
     */
    public float bestScore() {
        if (scores.isEmpty()) {
            return 0;
        }
        return Collections.max(scores);
    }

    /**
     * Filter masks by minimum score.
     */
    public SAM3Result filterByScore(float minScore) {
        List<Mask> filteredMasks = new ArrayList<>();
        List<Float> filteredScores = new ArrayList<>();
        List<Integer> filteredClassIds = new ArrayList<>();

        for (int i = 0; i < scores.size(); i++) {
            if (scores.get(i) >= minScore) {
                filteredMasks.add(masks.get(i));
                filteredScores.add(scores.get(i));
                filteredClassIds.add(classIds.get(i));
            }
        }

        return new SAM3Result(sourcePath, filteredMasks, filteredScores, filteredClassIds);
    }

    public String toJson() { return ResultSerializer.toJson(this); }
    public Map<String, Object> toMap() { return ResultSerializer.toMap(this); }
}
