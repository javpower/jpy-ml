package io.github.javpower.jpyml.ml.annotation;

import io.github.javpower.jpyml.ml.model.TaskType;
import io.github.javpower.jpyml.ml.result.*;
import io.github.javpower.jpyml.core.PythonEngine;
import io.github.javpower.jpyml.core.PythonScriptLoader;
import jep.JepException;

import java.awt.*;

/**
 * Draws inference results onto images. Supports all task types.
 */
public class ImageAnnotator {

    private static final Color[] PALETTE = {
            Color.RED, Color.GREEN, Color.BLUE, Color.ORANGE, Color.MAGENTA,
            Color.CYAN, Color.PINK, Color.YELLOW, new Color(128, 0, 255), new Color(0, 128, 128)
    };

    private final PythonEngine engine;
    private boolean scriptsLoaded = false;

    public ImageAnnotator() throws JepException {
        this.engine = PythonEngine.getInstance();
    }

    private void ensureScripts() throws JepException {
        if (!scriptsLoaded) {
            PythonScriptLoader.ensureLoaded(engine, "_jpy_annotation.py");
            scriptsLoaded = true;
        }
    }

    public String annotate(InferenceResult result, String outputPath) {
        try {
            ensureScripts();
            String inputPath = result.getSourcePath();
            engine.put("_jpy_ann_input", inputPath);
            engine.put("_jpy_ann_output", outputPath);
            engine.put("_jpy_ann_task", result.getTaskType().getKey());

            String boxesPy = buildBoxesPython(result);
            engine.exec("_jpy_ann_boxes = " + boxesPy);

            engine.exec("_jpy_ann_result = jpy_annotate(_jpy_ann_input, _jpy_ann_output, _jpy_ann_boxes, _jpy_ann_task)");
            return engine.eval("_jpy_ann_result");
        } catch (JepException e) {
            throw new RuntimeException("Annotation failed", e);
        }
    }

    private String buildBoxesPython(InferenceResult result) {
        StringBuilder sb = new StringBuilder("[");

        switch (result.getTaskType()) {
            case DETECT -> {
                DetectionResult dr = (DetectionResult) result;
                for (int i = 0; i < dr.getBoxes().size(); i++) {
                    if (i > 0) sb.append(", ");
                    sb.append(boxDict(dr.getBoxes().get(i)));
                }
            }
            case SEGMENT -> {
                SegmentationResult sr = (SegmentationResult) result;
                for (int i = 0; i < sr.getBoxes().size(); i++) {
                    if (i > 0) sb.append(", ");
                    ClassPrediction p = sr.getBoxes().get(i);
                    if (i < sr.getMasks().size()) {
                        Mask m = sr.getMasks().get(i);
                        sb.append(boxWithPolygonDict(p, m));
                    } else {
                        sb.append(boxDict(p));
                    }
                }
            }
            case CLASSIFY -> {
                ClassificationResult cr = (ClassificationResult) result;
                for (int i = 0; i < cr.getTopK().size(); i++) {
                    if (i > 0) sb.append(", ");
                    ClassPrediction p = cr.getTopK().get(i);
                    sb.append("{'class_id': ").append(p.classId())
                            .append(", 'confidence': ").append(p.confidence())
                            .append(", 'class_name': '").append(p.className()).append("'}");
                }
            }
            case POSE -> {
                PoseResult pr = (PoseResult) result;
                for (int i = 0; i < pr.getBoxes().size(); i++) {
                    if (i > 0) sb.append(", ");
                    ClassPrediction p = pr.getBoxes().get(i);
                    if (i < pr.getKeypoints().size()) {
                        KeypointCollection kc = pr.getKeypoints().get(i);
                        sb.append(boxWithKeypointsDict(p, kc));
                    } else {
                        sb.append(boxDict(p));
                    }
                }
            }
            case OBB -> {
                OBBResult or = (OBBResult) result;
                for (int i = 0; i < or.getPredictions().size(); i++) {
                    if (i > 0) sb.append(", ");
                    sb.append(obbDict(or.getPredictions().get(i)));
                }
            }
        }

        sb.append("]");
        return sb.toString();
    }

    private String boxDict(ClassPrediction p) {
        BoundingBox b = p.box();
        return "{'x1': " + b.x1() + ", 'y1': " + b.y1()
                + ", 'x2': " + b.x2() + ", 'y2': " + b.y2()
                + ", 'confidence': " + p.confidence()
                + ", 'class_id': " + p.classId()
                + ", 'class_name': '" + p.className() + "'}";
    }

    private String boxWithPolygonDict(ClassPrediction p, Mask m) {
        String base = boxDict(p);
        StringBuilder sb = new StringBuilder(base);
        sb.deleteCharAt(sb.length() - 1); // remove trailing }
        sb.append(", 'polygon': [");
        float[][] poly = m.getPolygon();
        for (int i = 0; i < poly.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append("[").append(poly[i][0]).append(", ").append(poly[i][1]).append("]");
        }
        sb.append("]}");
        return sb.toString();
    }

    private String boxWithKeypointsDict(ClassPrediction p, KeypointCollection kc) {
        String base = boxDict(p);
        StringBuilder sb = new StringBuilder(base);
        sb.deleteCharAt(sb.length() - 1); // remove trailing }
        sb.append(", 'keypoints': [");
        for (int i = 0; i < kc.size(); i++) {
            if (i > 0) sb.append(", ");
            Keypoint k = kc.get(i);
            sb.append("[").append(k.x()).append(", ").append(k.y()).append(", ").append(k.confidence()).append("]");
        }
        sb.append("]}");
        return sb.toString();
    }

    private String obbDict(OBBPrediction p) {
        RotatedBoundingBox rb = p.box();
        return "{'cx': " + rb.centerX()
                + ", 'cy': " + rb.centerY()
                + ", 'w': " + rb.width()
                + ", 'h': " + rb.height()
                + ", 'angle': " + rb.angleDegrees()
                + ", 'confidence': " + p.confidence()
                + ", 'class_id': " + p.classId()
                + ", 'class_name': '" + p.className() + "'}";
    }
}
