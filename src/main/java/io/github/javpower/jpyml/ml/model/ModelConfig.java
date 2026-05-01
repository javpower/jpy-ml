package io.github.javpower.jpyml.ml.model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class ModelConfig {
    private float confidence = 0.25f;
    private float iouThreshold = 0.7f;
    private int imageSize = 640;
    private int maxDetections = 300;
    private Device device = null;
    private boolean augment = false;
    private boolean agnosticNms = false;
    private List<Integer> classes = null;
    private int vidStride = 1;
    private boolean retinaMasks = false;
    private boolean half = false;
    private List<Integer> embed = null;
    private boolean verbose = true;
    private boolean save = false;
    private boolean saveTxt = false;
    private boolean saveConf = false;
    private boolean saveCrop = false;
    private boolean show = false;
    private boolean showLabels = true;
    private boolean showConf = true;
    private boolean showBoxes = true;
    private int lineWidth = -1;

    public ModelConfig confidence(float v) { confidence = v; return this; }
    public ModelConfig iouThreshold(float v) { iouThreshold = v; return this; }
    public ModelConfig imageSize(int v) { imageSize = v; return this; }
    public ModelConfig maxDetections(int v) { maxDetections = v; return this; }
    public ModelConfig device(Device v) { device = v; return this; }
    public ModelConfig device(int v) { device = Device.gpu(v); return this; }
    public ModelConfig augment(boolean v) { augment = v; return this; }
    public ModelConfig agnosticNms(boolean v) { agnosticNms = v; return this; }
    public ModelConfig filterClasses(int... ids) {
        classes = new ArrayList<>();
        for (int id : ids) classes.add(id);
        return this;
    }
    public ModelConfig vidStride(int v) { vidStride = v; return this; }
    public ModelConfig retinaMasks(boolean v) { retinaMasks = v; return this; }
    public ModelConfig half(boolean v) { half = v; return this; }
    public ModelConfig embed(int... layers) {
        this.embed = Arrays.stream(layers).boxed().collect(Collectors.toList());
        return this;
    }
    public ModelConfig verbose(boolean v) { verbose = v; return this; }
    public ModelConfig save(boolean v) { save = v; return this; }
    public ModelConfig saveTxt(boolean v) { saveTxt = v; return this; }
    public ModelConfig saveConf(boolean v) { saveConf = v; return this; }
    public ModelConfig saveCrop(boolean v) { saveCrop = v; return this; }
    public ModelConfig show(boolean v) { show = v; return this; }
    public ModelConfig showLabels(boolean v) { showLabels = v; return this; }
    public ModelConfig showConf(boolean v) { showConf = v; return this; }
    public ModelConfig showBoxes(boolean v) { showBoxes = v; return this; }
    public ModelConfig lineWidth(int v) { lineWidth = v; return this; }

    public Map<String, Object> toPythonKwargs() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("conf", confidence);
        m.put("iou", iouThreshold);
        m.put("imgsz", imageSize);
        m.put("max_det", maxDetections);
        if (device != null) m.put("device", device.toPython());
        if (augment) m.put("augment", true);
        if (agnosticNms) m.put("agnostic_nms", true);
        if (classes != null) m.put("classes", classes);
        if (vidStride != 1) m.put("vid_stride", vidStride);
        if (retinaMasks) m.put("retina_masks", true);
        if (half) m.put("half", true);
        if (embed != null) m.put("embed", embed);
        if (!verbose) m.put("verbose", false);
        if (save) m.put("save", true);
        if (saveTxt) m.put("save_txt", true);
        if (saveConf) m.put("save_conf", true);
        if (saveCrop) m.put("save_crop", true);
        if (show) m.put("show", true);
        if (!showLabels) m.put("show_labels", false);
        if (!showConf) m.put("show_conf", false);
        if (!showBoxes) m.put("show_boxes", false);
        if (lineWidth >= 0) m.put("line_width", lineWidth);
        return m;
    }
}
