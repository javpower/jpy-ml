package io.github.javpower.jpyml.ml.result;

import io.github.javpower.jpyml.exception.InferenceException;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Result of SAM 2 interactive segmentation.
 * Contains masks and their confidence scores.
 *
 * @param sourcePath path to the source image
 * @param masks      list of segmentation masks (polygons)
 * @param scores     confidence scores for each mask
 */
public record SAM2Result(
        String sourcePath,
        List<Mask> masks,
        List<Float> scores
) {
    public SAM2Result {
        masks = Collections.unmodifiableList(masks);
        scores = Collections.unmodifiableList(scores);
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

    public String toJson() { return ResultSerializer.toJson(this); }
    public Map<String, Object> toMap() { return ResultSerializer.toMap(this); }
}
