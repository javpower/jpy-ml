package io.github.javpower.jpyml.ml.training;
import java.util.LinkedHashMap;
import java.util.Map;
public class AugmentationConfig {
    private float hsvH = 0.015f, hsvS = 0.7f, hsvV = 0.4f;
    private float degrees = 0, translate = 0.1f, scale = 0.5f, shear = 0, perspective = 0;
    private float flipUD = 0, flipLR = 0.5f;
    private float mosaic = 1.0f, mixup = 0, copyPaste = 0;
    private int closeMosaic = 10;
    public AugmentationConfig mosaic(float v) { mosaic=v; return this; }
    public AugmentationConfig mixup(float v) { mixup=v; return this; }
    public AugmentationConfig flipLR(float v) { flipLR=v; return this; }
    public AugmentationConfig flipUD(float v) { flipUD=v; return this; }
    public AugmentationConfig scale(float v) { scale=v; return this; }
    public AugmentationConfig degrees(float v) { degrees=v; return this; }
    public AugmentationConfig hsvH(float v) { hsvH=v; return this; }
    public AugmentationConfig hsvS(float v) { hsvS=v; return this; }
    public AugmentationConfig hsvV(float v) { hsvV=v; return this; }
    public AugmentationConfig translate(float v) { translate=v; return this; }
    public AugmentationConfig shear(float v) { shear=v; return this; }
    public AugmentationConfig perspective(float v) { perspective=v; return this; }
    public AugmentationConfig copyPaste(float v) { copyPaste=v; return this; }
    public AugmentationConfig closeMosaic(int v) { closeMosaic=v; return this; }
    public Map<String,Object> toMap() {
        Map<String,Object> m = new LinkedHashMap<>();
        m.put("hsv_h", hsvH); m.put("hsv_s", hsvS); m.put("hsv_v", hsvV);
        m.put("degrees", degrees); m.put("translate", translate); m.put("scale", scale);
        m.put("shear", shear); m.put("perspective", perspective);
        m.put("flipud", flipUD); m.put("fliplr", flipLR);
        m.put("mosaic", mosaic); m.put("mixup", mixup); m.put("copy_paste", copyPaste);
        m.put("close_mosaic", closeMosaic);
        return m;
    }
}
