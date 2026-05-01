package io.github.javpower.jpyml.ml.export;

public class ExportResult {
    private final String outputPath;
    private final ExportFormat format;
    private final long fileSizeBytes;

    public ExportResult(String outputPath, ExportFormat format, long fileSizeBytes) {
        this.outputPath = outputPath;
        this.format = format;
        this.fileSizeBytes = fileSizeBytes;
    }

    public String getOutputPath() { return outputPath; }
    public ExportFormat getFormat() { return format; }
    public long getFileSizeBytes() { return fileSizeBytes; }
    public String getFileSizeMB() { return String.format("%.1f MB", fileSizeBytes / (1024.0 * 1024.0)); }

    @Override
    public String toString() {
        return String.format("ExportResult{path='%s', format=%s, size=%s}", outputPath, format, getFileSizeMB());
    }
}
