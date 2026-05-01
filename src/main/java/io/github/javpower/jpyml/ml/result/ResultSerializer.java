package io.github.javpower.jpyml.ml.result;

import io.github.javpower.jpyml.ml.model.TaskType;

import java.util.*;

/**
 * Lightweight JSON and Map serializer for inference results.
 * No external dependencies — uses StringBuilder for JSON output.
 */
public final class ResultSerializer {

    private ResultSerializer() {}

    // ── JSON ──────────────────────────────────────────────────────────────

    public static String toJson(InferenceResult result) {
        Map<String, Object> map = toMap(result);
        return mapToJson(map);
    }

    public static String toJson(SAM2Result result) {
        return mapToJson(toMap(result));
    }

    public static String toJson(SAM3Result result) {
        return mapToJson(toMap(result));
    }

    // ── Map ───────────────────────────────────────────────────────────────

    public static Map<String, Object> toMap(InferenceResult result) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("task", result.getTaskType().name().toLowerCase());
        m.put("source", result.getSourcePath());
        m.put("width", result.getOriginalWidth());
        m.put("height", result.getOriginalHeight());
        m.put("count", result.count());

        if (result.getSpeed() != null) {
            Map<String, Object> speed = new LinkedHashMap<>();
            speed.put("preprocess_ms", round2(result.getSpeed().preprocessMs()));
            speed.put("inference_ms", round2(result.getSpeed().inferenceMs()));
            speed.put("postprocess_ms", round2(result.getSpeed().postprocessMs()));
            m.put("speed", speed);
        }

        switch (result.getTaskType()) {
            case DETECT -> m.put("boxes", boxList(((DetectionResult) result).getBoxes()));
            case SEGMENT -> {
                SegmentationResult sr = (SegmentationResult) result;
                m.put("boxes", boxList(sr.getBoxes()));
                m.put("masks", maskList(sr.getMasks()));
            }
            case CLASSIFY -> m.put("predictions", classList(((ClassificationResult) result).getTopK()));
            case POSE -> {
                PoseResult pr = (PoseResult) result;
                m.put("boxes", boxList(pr.getBoxes()));
                m.put("keypoints", keypointList(pr.getKeypoints()));
            }
            case OBB -> m.put("predictions", obbList(((OBBResult) result).getPredictions()));
        }
        return m;
    }

    public static Map<String, Object> toMap(SAM2Result result) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("task", "sam2");
        m.put("source", result.sourcePath());
        m.put("count", result.count());
        m.put("masks", maskList(result.masks()));
        List<Object> scores = new ArrayList<>();
        for (Float s : result.scores()) scores.add(round2(s));
        m.put("scores", scores);
        return m;
    }

    public static Map<String, Object> toMap(SAM3Result result) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("task", "sam3");
        m.put("source", result.sourcePath());
        m.put("count", result.count());
        m.put("masks", maskList(result.masks()));
        List<Object> scores = new ArrayList<>();
        for (Float s : result.scores()) scores.add(round2(s));
        m.put("scores", scores);
        m.put("class_ids", result.classIds());
        return m;
    }

    // ── helpers ───────────────────────────────────────────────────────────

    private static List<Object> boxList(List<ClassPrediction> boxes) {
        List<Object> list = new ArrayList<>();
        for (ClassPrediction p : boxes) {
            Map<String, Object> b = new LinkedHashMap<>();
            b.put("class", p.className());
            b.put("confidence", round2(p.confidence()));
            b.put("x1", round2(p.box().x1()));
            b.put("y1", round2(p.box().y1()));
            b.put("x2", round2(p.box().x2()));
            b.put("y2", round2(p.box().y2()));
            list.add(b);
        }
        return list;
    }

    private static List<Object> classList(List<ClassPrediction> preds) {
        List<Object> list = new ArrayList<>();
        for (ClassPrediction p : preds) {
            Map<String, Object> b = new LinkedHashMap<>();
            b.put("class", p.className());
            b.put("confidence", round2(p.confidence()));
            b.put("class_id", p.classId());
            list.add(b);
        }
        return list;
    }

    private static List<Object> maskList(List<Mask> masks) {
        List<Object> list = new ArrayList<>();
        for (Mask mask : masks) {
            List<List<Object>> pts = new ArrayList<>();
            for (float[] p : mask.getPolygon()) {
                pts.add(List.of(round2(p[0]), round2(p[1])));
            }
            list.add(Map.of("points", pts, "count", pts.size()));
        }
        return list;
    }

    private static List<Object> keypointList(List<KeypointCollection> persons) {
        List<Object> list = new ArrayList<>();
        for (KeypointCollection kc : persons) {
            List<Object> kpts = new ArrayList<>();
            for (Keypoint k : kc.getAll()) {
                kpts.add(Map.of("x", round2(k.x()), "y", round2(k.y()), "confidence", round2(k.confidence())));
            }
            list.add(kpts);
        }
        return list;
    }

    private static List<Object> obbList(List<OBBPrediction> preds) {
        List<Object> list = new ArrayList<>();
        for (OBBPrediction p : preds) {
            Map<String, Object> b = new LinkedHashMap<>();
            b.put("class", p.className());
            b.put("confidence", round2(p.confidence()));
            b.put("cx", round2(p.box().centerX()));
            b.put("cy", round2(p.box().centerY()));
            b.put("width", round2(p.box().width()));
            b.put("height", round2(p.box().height()));
            b.put("angle", round2(p.box().angleDegrees()));
            list.add(b);
        }
        return list;
    }

    // ── JSON writer ───────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private static String mapToJson(Object value) {
        if (value == null) return "null";
        if (value instanceof Number) return value.toString();
        if (value instanceof Boolean) return value.toString();
        if (value instanceof String) return escapeString((String) value);
        if (value instanceof Map) {
            Map<String, Object> map = (Map<String, Object>) value;
            StringBuilder sb = new StringBuilder("{");
            boolean first = true;
            for (Map.Entry<String, Object> e : map.entrySet()) {
                if (!first) sb.append(',');
                sb.append(escapeString(e.getKey())).append(':').append(mapToJson(e.getValue()));
                first = false;
            }
            sb.append('}');
            return sb.toString();
        }
        if (value instanceof List) {
            List<?> list = (List<?>) value;
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) sb.append(',');
                sb.append(mapToJson(list.get(i)));
            }
            sb.append(']');
            return sb.toString();
        }
        return escapeString(value.toString());
    }

    private static String escapeString(String s) {
        StringBuilder sb = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> sb.append(c);
            }
        }
        return sb.append('"').toString();
    }

    private static double round2(float v) {
        return Math.round(v * 100.0) / 100.0;
    }
}
