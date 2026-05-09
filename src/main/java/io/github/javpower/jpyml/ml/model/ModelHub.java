package io.github.javpower.jpyml.ml.model;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Model registry with auto-download capabilities.
 * Caches models in {@code ~/.jpy-ml/models/}.
 */
public final class ModelHub {

    private static final Logger log = LoggerFactory.getLogger(ModelHub.class);
    private static final Path CACHE_DIR = Path.of(System.getProperty("user.home"), ".jpy-ml", "models");
    private static final String DEFAULT_BASE_URL = "https://github.com/ultralytics/assets/releases/download";

    private static String getBaseUrl() {
        String url = System.getProperty("jpy.model.base-url");
        if (url == null || url.isEmpty()) {
            url = System.getenv("JPY_MODEL_BASE_URL");
        }
        if (url == null || url.isEmpty()) {
            // 快捷代理模式下自动使用镜像
            String quickProxy = System.getProperty("jpy.proxy");
            if ("true".equalsIgnoreCase(quickProxy)) {
                url = "https://mirror.ghproxy.com/https://github.com/ultralytics/assets/releases/download";
            }
        }
        if (url == null || url.isEmpty()) {
            url = DEFAULT_BASE_URL;
        }
        if (!url.endsWith("/")) {
            url += "/";
        }
        return url;
    }

    private static Proxy getDownloadProxy() {
        // 快捷模式: -Djpy.proxy=true 使用默认代理 (http://127.0.0.1:7890)
        String quickProxy = System.getProperty("jpy.proxy");
        if ("true".equalsIgnoreCase(quickProxy)) {
            return new Proxy(Proxy.Type.HTTP, new InetSocketAddress("127.0.0.1", 7890));
        }

        String proxy = System.getProperty("jpy.download.proxy");
        if (proxy == null || proxy.isEmpty()) {
            proxy = System.getenv("JPY_DOWNLOAD_PROXY");
        }
        if (proxy != null && !proxy.isEmpty()) {
            try {
                URI uri = new URI(proxy);
                String host = uri.getHost();
                int port = uri.getPort();
                if (port == -1) {
                    port = "https".equals(uri.getScheme()) ? 443 : 80;
                }
                return new Proxy(Proxy.Type.HTTP, new InetSocketAddress(host, port));
            } catch (URISyntaxException e) {
                log.warn("Invalid proxy URL: {}, ignoring", proxy);
            }
        }
        return Proxy.NO_PROXY;
    }

    private static final Map<String, ModelEntry> REGISTRY = new LinkedHashMap<>();

    static {
        // YOLOv8
        reg("yolov8n", "v8.4.45", "yolov8n.pt", TaskType.DETECT, "6.2 MB");
        reg("yolov8s", "v8.4.45", "yolov8s.pt", TaskType.DETECT, "22.5 MB");
        reg("yolov8m", "v8.4.45", "yolov8m.pt", TaskType.DETECT, "52.0 MB");
        reg("yolov8l", "v8.4.45", "yolov8l.pt", TaskType.DETECT, "83.6 MB");
        reg("yolov8x", "v8.4.45", "yolov8x.pt", TaskType.DETECT, "130.5 MB");
        reg("yolov8n-seg", "v8.4.45", "yolov8n-seg.pt", TaskType.SEGMENT, "6.7 MB");
        reg("yolov8s-seg", "v8.4.45", "yolov8s-seg.pt", TaskType.SEGMENT, "23.7 MB");
        reg("yolov8n-cls", "v8.4.45", "yolov8n-cls.pt", TaskType.CLASSIFY, "5.4 MB");
        reg("yolov8n-pose", "v8.4.45", "yolov8n-pose.pt", TaskType.POSE, "6.5 MB");
        reg("yolov8n-obb", "v8.4.45", "yolov8n-obb.pt", TaskType.OBB, "6.3 MB");

        // YOLO11
        reg("yolo11n", "v8.4.45", "yolo11n.pt", TaskType.DETECT, "5.4 MB");
        reg("yolo11s", "v8.4.45", "yolo11s.pt", TaskType.DETECT, "18.4 MB");
        reg("yolo11n-seg", "v8.4.45", "yolo11n-seg.pt", TaskType.SEGMENT, "5.9 MB");
        reg("yolo11n-cls", "v8.4.45", "yolo11n-cls.pt", TaskType.CLASSIFY, "5.3 MB");
        reg("yolo11n-pose", "v8.4.45", "yolo11n-pose.pt", TaskType.POSE, "5.7 MB");
        reg("yolo11n-obb", "v8.4.45", "yolo11n-obb.pt", TaskType.OBB, "5.5 MB");

        // YOLO26
        reg("yolo26n", "v8.4.45", "yolo26n.pt", TaskType.DETECT, "8.2 MB");

        // RT-DETR
        reg("rtdetr-l", "v8.4.45", "rtdetr-l.pt", TaskType.DETECT, "44.0 MB");
        reg("rtdetr-x", "v8.4.45", "rtdetr-x.pt", TaskType.DETECT, "71.0 MB");

        // SAM 2.1
        reg("sam2.1_t", "v8.4.45", "sam2.1_t.pt", null, "39.0 MB");
        reg("sam2.1_s", "v8.4.45", "sam2.1_s.pt", null, "91.0 MB");
        reg("sam2.1_b", "v8.4.45", "sam2.1_b.pt", null, "185.0 MB");
        reg("sam2.1_l", "v8.4.45", "sam2.1_l.pt", null, "361.0 MB");
    }

    private ModelHub() {}

    private static void reg(String name, String version, String filename, TaskType task, String size) {
        REGISTRY.put(name, new ModelEntry(name, version, filename, task, size));
    }

    /**
     * Ensure model is downloaded. Returns local path.
     * Downloads from Ultralytics assets if not cached.
     */
    public static Path ensure(String modelName) throws IOException {
        Path cached = getCachedPath(modelName);
        if (cached != null && Files.exists(cached)) {
            return cached;
        }
        return download(modelName);
    }

    /**
     * Download model to cache directory. Returns local path.
     */
    public static Path download(String modelName) throws IOException {
        ModelEntry entry = REGISTRY.get(modelName);
        if (entry == null) {
            throw new IllegalArgumentException("Unknown model: " + modelName +
                    ". Available: " + String.join(", ", REGISTRY.keySet()));
        }

        Files.createDirectories(CACHE_DIR);
        Path target = CACHE_DIR.resolve(entry.filename);

        if (Files.exists(target)) {
            log.info("[jpy-ml] Model {} already cached at {}", modelName, target);
            return target;
        }

        String url = getBaseUrl() + entry.version + "/" + entry.filename;
        log.info("[jpy-ml] Downloading {} ({}) ...", modelName, entry.size);

        Path tempFile = CACHE_DIR.resolve(entry.filename + ".tmp");
        try {
            downloadFile(url, tempFile);
            Files.move(tempFile, target);
            log.info("[jpy-ml] Downloaded {} -> {}", modelName, target);
            return target;
        } catch (Exception e) {
            Files.deleteIfExists(tempFile);
            throw new IOException("Failed to download " + modelName + ": " + e.getMessage(), e);
        }
    }

    /**
     * Get cached model path, or null if not cached.
     */
    public static Path getCachedPath(String modelName) {
        ModelEntry entry = REGISTRY.get(modelName);
        if (entry == null) return null;
        return CACHE_DIR.resolve(entry.filename);
    }

    /**
     * Check if a model is registered.
     */
    public static boolean isRegistered(String modelName) {
        return REGISTRY.containsKey(modelName);
    }

    /**
     * Check if a model is already downloaded.
     */
    public static boolean isCached(String modelName) {
        Path p = getCachedPath(modelName);
        return p != null && Files.exists(p);
    }

    /**
     * List all registered models.
     */
    public static List<ModelEntry> listAvailable() {
        return List.copyOf(REGISTRY.values());
    }

    private static void downloadFile(String urlStr, Path target) throws IOException {
        Proxy proxy = getDownloadProxy();
        URL url = new URL(urlStr);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection(proxy);
        conn.setRequestProperty("User-Agent", "jpy-ml");
        conn.setConnectTimeout(30000);
        conn.setReadTimeout(600000);
        conn.setInstanceFollowRedirects(true);

        // Follow GitHub redirect
        int status = conn.getResponseCode();
        if (status == HttpURLConnection.HTTP_MOVED_PERM || status == HttpURLConnection.HTTP_MOVED_TEMP) {
            String redirectUrl = conn.getHeaderField("Location");
            conn.disconnect();
            url = new URL(redirectUrl);
            conn = (HttpURLConnection) url.openConnection(proxy);
            conn.setRequestProperty("User-Agent", "jpy-ml");
            conn.setConnectTimeout(30000);
            conn.setReadTimeout(600000);
        }

        long contentLength = conn.getContentLengthLong();
        try (InputStream in = conn.getInputStream();
             OutputStream out = Files.newOutputStream(target)) {
            byte[] buffer = new byte[8192];
            long totalRead = 0;
            int lastPercent = -1;
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
                totalRead += read;
                if (contentLength > 0) {
                    int percent = (int) (totalRead * 100 / contentLength);
                    if (percent != lastPercent && percent % 10 == 0) {
                        log.info("[jpy-ml]   {}% ({}/{} MB)",
                                percent,
                                totalRead / (1024 * 1024),
                                contentLength / (1024 * 1024));
                        lastPercent = percent;
                    }
                }
            }
        } finally {
            conn.disconnect();
        }
    }

    /**
     * Model registry entry.
     */
    public static final class ModelEntry {
        private final String name;
        private final String version;
        private final String filename;
        private final TaskType task;
        private final String size;

        ModelEntry(String name, String version, String filename, TaskType task, String size) {
            this.name = name; this.version = version; this.filename = filename;
            this.task = task; this.size = size;
        }

        public String getName() { return name; }
        public String getVersion() { return version; }
        public String getFilename() { return filename; }
        public TaskType getTask() { return task; }
        public String getSize() { return size; }
        public String getUrl() { return getBaseUrl() + version + "/" + filename; }

        @Override
        public String toString() {
            return name + " (" + size + ", " + (task != null ? task : "SAM") + ")";
        }
    }
}
