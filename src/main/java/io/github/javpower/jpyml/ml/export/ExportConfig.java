package io.github.javpower.jpyml.ml.export;

public class ExportConfig {
    private ExportFormat format = ExportFormat.ONNX;
    private boolean half = false;
    private boolean dynamic = false;
    private boolean simplify = true;
    private int opset = 17;
    private int imgsz = 640;

    public ExportConfig format(ExportFormat v) { format = v; return this; }
    public ExportConfig half(boolean v) { half = v; return this; }
    public ExportConfig dynamic(boolean v) { dynamic = v; return this; }
    public ExportConfig simplify(boolean v) { simplify = v; return this; }
    public ExportConfig opset(int v) { opset = v; return this; }
    public ExportConfig imgsz(int v) { imgsz = v; return this; }

    public ExportFormat getFormat() { return format; }
    public boolean isHalf() { return half; }
    public boolean isDynamic() { return dynamic; }
    public boolean isSimplify() { return simplify; }
    public int getOpset() { return opset; }
    public int getImgsz() { return imgsz; }
}
