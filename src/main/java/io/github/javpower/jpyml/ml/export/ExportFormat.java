package io.github.javpower.jpyml.ml.export;
public enum ExportFormat {
    TORCHSCRIPT("torchscript"), ONNX("onnx"), OPENVINO("openvino"), COREML("coreml"),
    TFLITE("tflite"), TENSORRT("engine"), NCNN("ncnn"), PADDLE("paddle");
    private final String key;
    ExportFormat(String key) { this.key = key; }
    public String getKey() { return key; }
}
