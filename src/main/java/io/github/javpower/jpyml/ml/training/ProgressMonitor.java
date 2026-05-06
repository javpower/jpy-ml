package io.github.javpower.jpyml.ml.training;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Daemon thread that monitors a progress file written by Python during training.
 * Reads JSON lines and dispatches epoch events to a TrainingCallback in real-time.
 * <p>
 * This class performs NO Jep operations — only file I/O. The callback is invoked
 * on the monitor thread, so implementors must synchronize any shared state access.
 */
public class ProgressMonitor {

    private static final Logger log = LoggerFactory.getLogger(ProgressMonitor.class);

    private final Path progressFile;
    private final TrainingCallback callback;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private volatile Thread monitorThread;
    private volatile String errorMessage;
    private volatile boolean completed;

    public ProgressMonitor(Path progressFile, TrainingCallback callback) {
        this.progressFile = progressFile;
        this.callback = callback;
    }

    /**
     * Start monitoring in a daemon thread.
     */
    public void start() {
        if (!running.compareAndSet(false, true)) {
            throw new IllegalStateException("Monitor already running");
        }
        completed = false;
        errorMessage = null;
        monitorThread = new Thread(this::runMonitor, "jpy-training-monitor");
        monitorThread.setDaemon(true);
        monitorThread.start();
    }

    /**
     * Block until the monitor receives a "done" event or is stopped.
     *
     * @return error message from the "done" event, or null if successful
     */
    public String awaitCompletion() throws InterruptedException {
        if (monitorThread != null) {
            monitorThread.join();
        }
        return errorMessage;
    }

    /**
     * Block until the monitor receives a "done" event, is stopped, or timeout elapses.
     *
     * @param timeoutSeconds max seconds to wait
     * @return error message from the "done" event, null if successful, or null on timeout
     */
    public String awaitCompletion(long timeoutSeconds) throws InterruptedException {
        if (monitorThread != null) {
            monitorThread.join(timeoutSeconds * 1000);
        }
        return errorMessage;
    }

    /**
     * Signal the monitor to stop polling.
     */
    public void stop() {
        running.set(false);
        if (monitorThread != null) {
            monitorThread.interrupt();
        }
    }

    public boolean isCompleted() {
        return completed;
    }

    private void runMonitor() {
        long filePosition = 0;
        try {
            // Wait for the progress file to be created by Python (up to 30s)
            long deadline = System.currentTimeMillis() + 30_000;
            while (running.get() && !Files.exists(progressFile)) {
                if (System.currentTimeMillis() > deadline) {
                    log.warn("Progress file not created after 30s: {}", progressFile);
                    return;
                }
                Thread.sleep(200);
            }

            while (running.get()) {
                boolean foundData = false;
                try (RandomAccessFile raf = new RandomAccessFile(progressFile.toFile(), "r")) {
                    raf.seek(filePosition);
                    String line;
                    while ((line = raf.readLine()) != null) {
                        foundData = true;
                        if (!line.isBlank()) {
                            processLine(line.trim());
                            if (completed) return;
                        }
                    }
                    filePosition = raf.getFilePointer();
                } catch (IOException e) {
                    log.debug("Error reading progress file: {}", e.getMessage());
                }
                if (foundData) continue;
                Thread.sleep(250);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (Exception e) {
            log.error("Progress monitor error", e);
        } finally {
            running.set(false);
        }
    }

    private void processLine(String line) {
        String type = extractStringValue(line, "type");
        if (type == null) return;

        switch (type) {
            case "epoch", "step" -> {
                int step = extractIntValue(line, "step");
                if (step == 0) step = extractIntValue(line, "epoch");
                callback.onEpoch(step, line);
            }
            case "info" -> log.info("Training: {}", extractStringValue(line, "status"));
            case "done" -> {
                String error = extractStringValue(line, "error");
                if (error != null && !error.isEmpty()) {
                    errorMessage = error;
                }
                completed = true;
            }
            default -> log.debug("Unknown progress event type: {}", type);
        }
    }

    static String extractStringValue(String json, String key) {
        String search = "\"" + key + "\":\"";
        int start = json.indexOf(search);
        if (start < 0) {
            String nullSearch = "\"" + key + "\":null";
            if (json.contains(nullSearch)) return null;
            return null;
        }
        start += search.length();
        StringBuilder sb = new StringBuilder();
        for (int i = start; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '"') break;
            if (c == '\\' && i + 1 < json.length()) {
                char next = json.charAt(i + 1);
                if (next == '"' || next == '\\') { sb.append(next); i++; continue; }
                if (next == 'n') { sb.append('\n'); i++; continue; }
                if (next == 't') { sb.append('\t'); i++; continue; }
            }
            sb.append(c);
        }
        return sb.toString();
    }

    static int extractIntValue(String json, String key) {
        String search = "\"" + key + "\":";
        int start = json.indexOf(search);
        if (start < 0) return 0;
        start += search.length();
        int end = start;
        while (end < json.length() && (Character.isDigit(json.charAt(end)) || json.charAt(end) == '-')) {
            end++;
        }
        try {
            return Integer.parseInt(json.substring(start, end));
        } catch (NumberFormatException e) {
            return 0;
        }
    }
}
