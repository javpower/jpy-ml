package io.github.javpower.jpyml.ml.result;

import java.util.List;

public final class ResultParseUtil {

    private ResultParseUtil() {}

    public static float[][] parsePolygon(List<List<Number>> polygon) {
        if (polygon == null) {
            return new float[0][2];
        }
        float[][] result = new float[polygon.size()][2];
        for (int i = 0; i < polygon.size(); i++) {
            result[i][0] = polygon.get(i).get(0).floatValue();
            result[i][1] = polygon.get(i).get(1).floatValue();
        }
        return result;
    }
}
