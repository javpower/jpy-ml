package io.github.javpower.jpyml.core;

import io.github.javpower.jpyml.exception.PythonException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Manages an embedded, self-contained Python runtime.
 * <p>
 * On first use, downloads python-build-standalone for the current platform,
 * extracts it, and installs the jep bridge package. Subsequent runs reuse
 * the cached runtime.
 */
public class PythonRuntime {

    private static final Logger log = LoggerFactory.getLogger(PythonRuntime.class);
    private static final String RUNTIME_DIR_NAME = ".jpy-ml";
    private static final String PYTHON_VERSION = "3.12.9";
    private static final String PYTHON_RELEASE_TAG = "20250317";
    private static final AtomicBoolean initialized = new AtomicBoolean(false);
    private static volatile Path runtimeRoot;
    private static volatile Path pythonHome;
    private static volatile Path jepNativeLib;
    private static volatile Path sitePackagesPath;

    private PythonRuntime() {
    }

    /**
     * Returns the runtime root directory (~/.jpy-ml by default).
     */
    public static Path getRuntimeRoot() {
        if (runtimeRoot == null) {
            runtimeRoot = Paths.get(System.getProperty("user.home"), RUNTIME_DIR_NAME);
        }
        return runtimeRoot;
    }

    /**
     * Set a custom runtime root directory. Must be called before init().
     */
    public static void setRuntimeRoot(Path root) {
        if (initialized.get()) {
            throw new IllegalStateException("Runtime already initialized");
        }
        runtimeRoot = root;
    }

    /**
     * Returns the Python home directory.
     */
    public static Path getPythonHome() {
        return pythonHome;
    }

    /**
     * Returns the Python executable path.
     */
    public static Path getPythonExecutable() {
        if (pythonHome == null) return null;
        if (isWindows()) {
            return pythonHome.resolve("python.exe");
        }
        return pythonHome.resolve("bin").resolve("python3");
    }

    /**
     * Initialize with an existing Python installation (e.g. a venv).
     * Skips the python-build-standalone download. Useful for development.
     *
     * @param pythonPath       path to the Python executable
     * @param jepLibraryPath   path to libjep.jnilib / libjep.so / jep.dll
     */
    public static synchronized void init(Path pythonPath, Path jepLibraryPath) throws IOException {
        if (initialized.get()) return;

        configureEnvironment();

        if (!Files.exists(pythonPath)) {
            throw new IOException("Python executable not found: " + pythonPath);
        }
        if (!Files.exists(jepLibraryPath)) {
            throw new IOException("Jep native library not found: " + jepLibraryPath);
        }

        // Derive pythonHome from the python executable
        // venv (Unix):   venv/bin/python3       -> pythonHome = venv
        // venv (Win):    venv\Scripts\python.exe -> pythonHome = venv
        // standalone (Unix): python/bin/python3  -> pythonHome = python
        // standalone (Win):  dir\python.exe      -> pythonHome = dir
        Path parent = pythonPath.getParent();
        if (isWindows() && parent.getFileName() != null &&
                parent.getFileName().toString().equalsIgnoreCase("Scripts")) {
            pythonHome = parent.getParent();
        } else {
            pythonHome = parent.getParent() != null ? parent.getParent() : parent;
        }
        jepNativeLib = jepLibraryPath;

        // Set runtimeRoot for pip install working directory
        if (runtimeRoot == null) {
            runtimeRoot = pythonHome;
        }

        // Find site-packages
        try (var stream = Files.walk(pythonHome, 5)) {
            sitePackagesPath = stream
                    .filter(p -> p.getFileName().toString().equals("site-packages") && Files.isDirectory(p))
                    .findFirst()
                    .orElse(null);
        }

        log.info("Using existing Python: {}", pythonPath);
        log.info("Jep native lib: {}", jepNativeLib);
        log.info("Site-packages: {}", sitePackagesPath);

        initialized.set(true);
    }

    /**
     * Initialize the embedded Python runtime (auto-download mode).
     * Downloads and extracts python-build-standalone if not already present.
     */
    public static synchronized void init() throws IOException {
        if (initialized.get()) return;

        configureEnvironment();

        getRuntimeRoot();
        Files.createDirectories(runtimeRoot);

        String platformKey = getPlatformKey();
        pythonHome = runtimeRoot.resolve("python").resolve(platformKey);

        if (!Files.exists(getPythonExecutable())) {
            log.info("First-time setup: downloading Python {} for {}...", PYTHON_VERSION, platformKey);
            downloadAndExtractPython(platformKey);
            log.info("Python runtime ready at {}", pythonHome);
        } else {
            log.info("Using cached Python runtime at {}", pythonHome);
        }

        // Ensure packages are installed even if Python was cached without them
        sitePackagesPath = computeSitePackages();
        jepNativeLib = findJepNativeLibrary();
        if (jepNativeLib == null || !Files.exists(jepNativeLib)) {
            log.info("Jep not found in cached runtime, installing packages...");
            installBundledRequirements();
            sitePackagesPath = computeSitePackages();
            jepNativeLib = findJepNativeLibrary();
            if (jepNativeLib == null || !Files.exists(jepNativeLib)) {
                throw new IOException("Jep native library not found after installation. " +
                        "Expected at: " + computeSitePackages().resolve("jep"));
            }
        }

        initialized.set(true);
    }

    /**
     * Install Python packages via pip into the embedded environment.
     */
    public static int pipInstall(String... packages) throws IOException, InterruptedException {
        ensureInitialized();
        Path pip = resolvePipExecutable();
        if (pip == null) throw new IOException("Python executable not found");

        String[] cmd = new String[packages.length + 4];
        cmd[0] = pip.toString();
        cmd[1] = "-m";
        cmd[2] = "pip";
        cmd[3] = "install";
        System.arraycopy(packages, 0, cmd, 4, packages.length);

        log.info("[pip] Running: {}", String.join(" ", cmd));
        log.info("[pip] Working directory: {}", runtimeRoot);

        ProcessBuilder pb = new ProcessBuilder(cmd)
                .redirectErrorStream(true)
                .directory(runtimeRoot.toFile());

        Process p = pb.start();
        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
                log.debug("[pip] {}", line);
            }
        }
        int exitCode = p.waitFor();
        if (exitCode != 0) {
            log.warn("[pip] install failed (exit={}):\n{}", exitCode, output);
        }
        return exitCode;
    }

    /**
     * Install a list of packages from a requirements.txt file.
     */
    public static int pipInstallFromRequirements(Path requirementsFile) throws IOException, InterruptedException {
        ensureInitialized();
        Path pip = resolvePipExecutable();
        if (pip == null) throw new IOException("Python executable not found");

        ProcessBuilder pb = new ProcessBuilder(
                pip.toString(), "-m", "pip", "install", "-r", requirementsFile.toString()
        ).redirectErrorStream(true);

        Process p = pb.start();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                log.debug("[pip] {}", line);
            }
        }
        return p.waitFor();
    }

    /**
     * Check if the runtime is initialized.
     */
    public static boolean isInitialized() {
        return initialized.get();
    }

    /**
     * Shutdown the runtime: close the PythonEngine singleton and reset state.
     */
    public static synchronized void shutdown() {
        if (!initialized.get()) return;
        try {
            PythonEngine.shutdown();
        } catch (Exception e) {
            log.debug("Error during engine shutdown: {}", e.getMessage());
        }
        initialized.set(false);
        pythonHome = null;
        jepNativeLib = null;
        sitePackagesPath = null;
        log.info("PythonRuntime shut down");
    }

    /**
     * Check if the runtime is ready (Python + jep + all required packages installed).
     */
    public static boolean isReady() {
        if (!initialized.get()) return false;
        if (jepNativeLib == null || !Files.exists(jepNativeLib)) return false;
        return true;
    }

    /**
     * Ensure Python packages are installed.
     */
    public static int ensurePackages(String... packages) throws IOException, InterruptedException {
        ensureInitialized();
        List<String> missing = new ArrayList<>();
        for (String pkg : packages) {
            if (!checkPackageInstalled(pkg)) {
                missing.add(pkg);
            }
        }
        if (missing.isEmpty()) return 0;
        log.info("Installing missing packages: {}", missing);
        return pipInstall(missing.toArray(new String[0]));
    }

    private static boolean checkPackageInstalled(String packageName) {
        Path pythonExe = getPythonExecutable();
        if (pythonExe == null || !Files.exists(pythonExe)) return false;
        try {
            ProcessBuilder pb = new ProcessBuilder(
                    pythonExe.toString(), "-c", "import " + packageName
            ).redirectErrorStream(true);
            Process p = pb.start();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                while (reader.readLine() != null) { /* drain */ }
            }
            int exit = p.waitFor();
            return exit == 0;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Get the site-packages directory for the embedded Python.
     */
    public static Path getSitePackages() {
        if (sitePackagesPath != null) return sitePackagesPath;
        return computeSitePackages();
    }

    private static Path computeSitePackages() {
        if (pythonHome == null) return null;
        String ver = PYTHON_VERSION.substring(0, PYTHON_VERSION.lastIndexOf('.'));
        if (isWindows()) {
            return pythonHome.resolve("Lib").resolve("site-packages");
        }
        return pythonHome.resolve("lib").resolve("python" + ver).resolve("site-packages");
    }

    /**
     * Configure environment variables for Python/PyTorch compatibility.
     */
    private static void configureEnvironment() {
        String gilOverride = System.getProperty("jpy.python.gil");
        if ("0".equals(gilOverride) || "false".equals(gilOverride)) {
            log.info("PYTHON_GIL override disabled via jpy.python.gil={}", gilOverride);
        } else if (System.getenv("PYTHON_GIL") == null) {
            setEnv("PYTHON_GIL", "1");
        }

        String ompOverride = System.getProperty("jpy.omp.threads");
        if (System.getenv("OMP_NUM_THREADS") == null) {
            int threads;
            if (ompOverride != null) {
                threads = Integer.parseInt(ompOverride);
            } else {
                int cores = Runtime.getRuntime().availableProcessors();
                threads = Math.max(1, Math.min(cores, 4));
            }
            setEnv("OMP_NUM_THREADS", String.valueOf(threads));
        }
        if (System.getenv("MKL_NUM_THREADS") == null) {
            setEnv("MKL_NUM_THREADS", System.getenv().getOrDefault("OMP_NUM_THREADS", "4"));
        }
        if (System.getenv("OPENBLAS_NUM_THREADS") == null) {
            setEnv("OPENBLAS_NUM_THREADS", System.getenv().getOrDefault("OMP_NUM_THREADS", "4"));
        }

        if (System.getenv("TORCH_SHARED_MEMORY") == null) {
            setEnv("TORCH_SHARED_MEMORY", "1");
        }

        if (System.getenv("TOKENIZERS_PARALLELISM") == null) {
            setEnv("TOKENIZERS_PARALLELISM", "false");
        }
    }

    private static void setEnv(String key, String value) {
        try {
            var env = System.getenv();
            var field = env.getClass().getDeclaredField("m");
            field.setAccessible(true);
            @SuppressWarnings("unchecked")
            var writableEnv = (java.util.Map<String, String>) field.get(env);
            writableEnv.put(key, value);
        } catch (NoSuchFieldException e) {
            try {
                var env = System.getenv();
                for (var f : env.getClass().getSuperclass().getDeclaredFields()) {
                    if (java.util.Map.class.isAssignableFrom(f.getType())) {
                        f.setAccessible(true);
                        @SuppressWarnings("unchecked")
                        var writableEnv = (java.util.Map<String, String>) f.get(env);
                        writableEnv.put(key, value);
                        return;
                    }
                }
                log.debug("Could not set env var {} via reflection: no map field found", key);
            } catch (Exception e2) {
                log.debug("Could not set env var {} via reflection: {}", key, e2.getMessage());
            }
        } catch (Exception e) {
            log.debug("Could not set env var {}={} via reflection: {}", key, value, e.getMessage());
        }
    }

    private static void ensureInitialized() {
        if (!initialized.get()) {
            throw new PythonException("PythonRuntime not initialized. Call PythonRuntime.init() first.");
        }
    }

    private static Path resolvePipExecutable() {
        Path exe = getPythonExecutable();
        if (exe != null && Files.exists(exe)) return exe;
        // Fallback: try the venv's pip directly
        if (pythonHome != null) {
            if (isWindows()) {
                Path scripts = pythonHome.resolve("Scripts");
                Path pip = scripts.resolve("pip.exe");
                if (Files.exists(pip)) return pip;
                Path pip3 = scripts.resolve("pip3.exe");
                if (Files.exists(pip3)) return pip3;
            } else {
                Path pip = pythonHome.resolve("bin").resolve("pip");
                if (Files.exists(pip)) return pip;
                Path pip3 = pythonHome.resolve("bin").resolve("pip3");
                if (Files.exists(pip3)) return pip3;
            }
        }
        return null;
    }

    /**
     * Find the jep native library (.jnilib/.so/.dll) installed in site-packages.
     * Searches for exact names first, then falls back to pattern matching
     * for ABI-tagged names like libjep.cpython-312-x86_64-linux-gnu.so.
     */
    public static Path findJepNativeLibrary() {
        if (jepNativeLib != null) return jepNativeLib;

        Path sp = getSitePackages();
        if (sp == null) return null;

        Path jepDir = sp.resolve("jep");
        if (!Files.exists(jepDir)) return null;

        // Try exact known names
        String[] exactNames = isWindows()
                ? new String[]{"jep.dll"}
                : System.getProperty("os.name").toLowerCase().contains("mac")
                    ? new String[]{"libjep.jnilib"}
                    : new String[]{"libjep.so", "libjep.jnilib"};

        for (String name : exactNames) {
            Path direct = jepDir.resolve(name);
            if (Files.exists(direct)) return direct;
        }

        // Fallback: search for any native library matching the pattern
        // Handles ABI-tagged names: libjep.cpython-312-*.so, jep.dll, libjep.jnilib
        try (var stream = Files.walk(jepDir)) {
            return stream
                    .filter(Files::isRegularFile)
                    .filter(p -> {
                        String name = p.getFileName().toString();
                        return name.startsWith("libjep.") && (name.endsWith(".so") || name.endsWith(".jnilib"))
                                || name.equals("jep.dll");
                    })
                    .findFirst()
                    .orElse(null);
        } catch (IOException e) {
            return null;
        }
    }

    /**
     * Get the include path for JepConfig (site-packages).
     */
    public static String getJepIncludePath() {
        Path sp = getSitePackages();
        return sp != null ? sp.toString() : "";
    }

    // ==================== Download & Extract ====================

    private static void downloadAndExtractPython(String platformKey) throws IOException {
        String downloadUrl = buildDownloadUrl(platformKey);
        String archiveName = downloadUrl.substring(downloadUrl.lastIndexOf('/') + 1);
        Path archivePath = runtimeRoot.resolve("downloads").resolve(archiveName);

        Files.createDirectories(archivePath.getParent());

        // Download
        if (!Files.exists(archivePath)) {
            log.info("Downloading from {}", downloadUrl);
            downloadFile(downloadUrl, archivePath);
            log.info("Download complete: {}", archivePath);
        }

        // Extract
        log.info("Extracting...");
        Path extractTarget = runtimeRoot.resolve("python");
        Files.createDirectories(extractTarget);

        try {
            if (archiveName.endsWith(".tar.zst") || archiveName.endsWith(".tar.gz")) {
                extractTar(archivePath, extractTarget, platformKey);
            } else if (archiveName.endsWith(".zip")) {
                extractZip(archivePath, extractTarget, platformKey);
            }

            // Verify extraction
            if (!Files.exists(getPythonExecutable())) {
                throw new IOException("Python executable not found after extraction: " + getPythonExecutable());
            }
        } catch (Exception e) {
            Path pythonDir = pythonHome;
            if (pythonDir != null && Files.exists(pythonDir)) {
                deleteRecursively(pythonDir);
            }
            throw e;
        }

        // Install required packages from bundled requirements.txt
        log.info("Installing Python packages (first run may take a few minutes)...");
        installBundledRequirements();

        // Verify jep was installed
        sitePackagesPath = computeSitePackages();
        jepNativeLib = findJepNativeLibrary();
        if (jepNativeLib == null || !Files.exists(jepNativeLib)) {
            // Diagnostic: show what's actually in site-packages
            Path sp = computeSitePackages();
            System.out.println("[jpy-ml] jep native lib NOT found");
            System.out.println("[jpy-ml] site-packages: " + sp);
            System.out.println("[jpy-ml] site-packages exists: " + (sp != null && Files.exists(sp)));
            if (sp != null && Files.exists(sp)) {
                Path jepDir = sp.resolve("jep");
                System.out.println("[jpy-ml] jep dir exists: " + Files.exists(jepDir));
                if (Files.exists(jepDir)) {
                    System.out.println("[jpy-ml] jep dir contents:");
                    try (var stream = Files.walk(jepDir, 2)) {
                        stream.forEach(f -> System.out.println("  " + f.getFileName()));
                    }
                }
                System.out.println("[jpy-ml] site-packages top-level:");
                try (var stream = Files.list(sp)) {
                    stream.map(p -> p.getFileName().toString()).sorted().forEach(n -> System.out.println("  " + n));
                }
            }
            throw new IOException("Jep native library not found after installation. " +
                    "Expected at: " + (jepNativeLib != null ? jepNativeLib : "site-packages/jep/"));
        }
    }

    private static void installBundledRequirements() throws IOException {
        Path pythonExe = getPythonExecutable();
        System.out.println("[jpy-ml] Installing bundled requirements with: " + pythonExe);

        // Install directly by package name instead of temp file to avoid
        // classpath resource loading issues on different platforms
        try {
            ProcessBuilder pb = new ProcessBuilder(
                    pythonExe.toString(), "-m", "pip", "install", "jep>=4.3.1", "numpy>=1.26"
            ).redirectErrorStream(true);

            Process p = pb.start();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    System.out.println("[pip] " + line);
                }
            }

            int exit = p.waitFor();
            System.out.println("[jpy-ml] pip install exit code: " + exit);
            if (exit != 0) {
                throw new IOException("pip install failed (exit code " + exit + ")");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted during pip install", e);
        }
    }

    private static String buildDownloadUrl(String platformKey) {
        String baseUrl = "https://github.com/astral-sh/python-build-standalone/releases/download/";
        String releaseTag = PYTHON_RELEASE_TAG;
        String fileName = "cpython-" + PYTHON_VERSION + "+" + PYTHON_RELEASE_TAG;

        switch (platformKey) {
            case "macos-arm64":
                fileName += "-aarch64-apple-darwin-install_only.tar.gz";
                break;
            case "macos-x86_64":
                fileName += "-x86_64-apple-darwin-install_only.tar.gz";
                break;
            case "linux-x86_64":
                fileName += "-x86_64-unknown-linux-gnu-install_only.tar.gz";
                break;
            case "linux-aarch64":
                fileName += "-aarch64-unknown-linux-gnu-install_only.tar.gz";
                break;
            case "windows-x86_64":
                fileName += "-x86_64-pc-windows-msvc-install_only.tar.gz";
                break;
            default:
                throw new UnsupportedOperationException("Unsupported platform: " + platformKey);
        }

        return baseUrl + releaseTag + "/" + fileName;
    }

    private static void downloadFile(String url, Path target) throws IOException {
        var connection = (HttpURLConnection) new URL(url).openConnection();
        connection.setConnectTimeout(30_000);
        connection.setReadTimeout(300_000);
        int totalSize = connection.getContentLength();
        try (InputStream in = connection.getInputStream();
             OutputStream out = Files.newOutputStream(target)) {
            byte[] buf = new byte[65536];
            long downloaded = 0;
            int lastPercent = -1;
            int n;
            while ((n = in.read(buf)) != -1) {
                out.write(buf, 0, n);
                downloaded += n;
                if (totalSize > 0) {
                    int percent = (int) (downloaded * 100 / totalSize);
                    if (percent != lastPercent && percent % 10 == 0) {
                        log.info("Downloaded {}/{} MB ({}%%)",
                                downloaded / (1024 * 1024), totalSize / (1024 * 1024), percent);
                        lastPercent = percent;
                    }
                }
            }
        }
    }

    // ==================== Pure Java Archive Extraction ====================

    private static final int TAR_BLOCK = 512;

    private static void extractTar(Path archive, Path targetDir, String platformKey) throws IOException {
        Path tempExtract = runtimeRoot.resolve("downloads").resolve("extract-temp");
        Files.createDirectories(tempExtract);

        if (archive.toString().endsWith(".tar.zst")) {
            // zstd requires external tool — try tar command (available on most Linux/macOS)
            extractTarExternal(archive, tempExtract);
        } else {
            // .tar.gz — pure Java extraction, works on all platforms including Windows
            extractTarGzJava(archive, tempExtract);
        }

        // Find the "python" directory inside the extracted content
        Path pythonDir = findPythonDir(tempExtract);
        if (pythonDir == null) {
            throw new IOException("Could not find python directory in extracted archive");
        }

        // Move to final location
        Path finalDir = targetDir.resolve(platformKey);
        Files.createDirectories(finalDir);
        moveDirectoryContents(pythonDir, finalDir);

        // Cleanup temp
        deleteRecursively(tempExtract);
    }

    private static void extractZip(Path archive, Path targetDir, String platformKey) throws IOException {
        Path tempExtract = runtimeRoot.resolve("downloads").resolve("extract-temp");
        Files.createDirectories(tempExtract);

        extractZipJava(archive, tempExtract);

        Path pythonDir = findPythonDir(tempExtract);
        if (pythonDir == null) {
            throw new IOException("Could not find python directory in extracted archive");
        }

        Path finalDir = targetDir.resolve(platformKey);
        Files.createDirectories(finalDir);
        moveDirectoryContents(pythonDir, finalDir);
        deleteRecursively(tempExtract);
    }

    /**
     * Pure Java tar.gz extraction. No external tools required.
     */
    private static void extractTarGzJava(Path archive, Path targetDir) throws IOException {
        log.info("Extracting tar.gz (pure Java)...");
        try (InputStream fis = Files.newInputStream(archive);
             InputStream gis = new GZIPInputStream(fis)) {
            extractTarStream(gis, targetDir);
        }
    }

    private static void extractTarStream(InputStream is, Path targetDir) throws IOException {
        byte[] header = new byte[TAR_BLOCK];
        String longName = null;

        while (true) {
            // Read 512-byte header
            if (!readFully(is, header, TAR_BLOCK)) break;

            // Check for end of archive (two zero blocks)
            if (isZeroBlock(header)) {
                readFully(is, header, TAR_BLOCK);
                break;
            }

            // Parse header fields
            String name = readTarString(header, 0, 100).trim();
            long size = readTarOctal(header, 124, 12);
            byte typeFlag = header[156];
            String prefix = readTarString(header, 345, 155).trim();

            // UStar long path: combine prefix + name
            if (!prefix.isEmpty()) {
                name = prefix + "/" + name;
            }

            // GNU tar long name (@LongLink)
            if (typeFlag == 'L') {
                longName = readTarData(is, size);
                skipTarPadding(is, size);
                continue;
            }
            if (typeFlag == 'K') {
                // Long link name — skip
                skipTarData(is, size);
                continue;
            }

            // Apply GNU long name if set
            if (longName != null) {
                name = longName;
                longName = null;
            }

            // Skip PaxHeader entries
            if (name.contains("PaxHeader") || name.contains("@LongLink") || name.isEmpty()) {
                skipTarData(is, size);
                continue;
            }

            // Strip leading "./" or "/"
            while (name.startsWith("./")) name = name.substring(2);
            while (name.startsWith("/")) name = name.substring(1);

            if (name.isEmpty()) {
                skipTarData(is, size);
                continue;
            }

            Path entryPath = targetDir.resolve(name).normalize();
            // Security check: prevent path traversal
            if (!entryPath.startsWith(targetDir)) {
                skipTarData(is, size);
                continue;
            }

            long mode = readTarOctal(header, 100, 8);
            boolean isDir = (typeFlag == '5') || name.endsWith("/");
            boolean isExecutable = (mode & 0111) != 0;

            if (isDir) {
                Files.createDirectories(entryPath);
                skipTarPadding(is, size);
            } else {
                // Regular file
                Files.createDirectories(entryPath.getParent());
                try (OutputStream out = Files.newOutputStream(entryPath)) {
                    byte[] buf = new byte[8192];
                    long remaining = size;
                    while (remaining > 0) {
                        int toRead = (int) Math.min(buf.length, remaining);
                        int n = is.read(buf, 0, toRead);
                        if (n < 0) break;
                        out.write(buf, 0, n);
                        remaining -= n;
                    }
                }
                // Restore executable permission from tar header
                if (isExecutable && !isWindows()) {
                    try {
                        entryPath.toFile().setExecutable(true, false);
                    } catch (Exception e) {
                        log.debug("Could not set executable bit on {}: {}", entryPath, e.getMessage());
                    }
                }
                skipTarPadding(is, size);
            }
        }
    }

    /**
     * Pure Java zip extraction. No external tools required.
     */
    private static void extractZipJava(Path archive, Path targetDir) throws IOException {
        log.info("Extracting zip (pure Java)...");
        try (ZipInputStream zis = new ZipInputStream(Files.newInputStream(archive))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                String name = entry.getName();
                // Strip leading "./" or "/"
                while (name.startsWith("./")) name = name.substring(2);
                while (name.startsWith("/")) name = name.substring(1);

                if (name.isEmpty()) {
                    zis.closeEntry();
                    continue;
                }

                Path entryPath = targetDir.resolve(name).normalize();
                if (!entryPath.startsWith(targetDir)) {
                    zis.closeEntry();
                    continue;
                }

                if (entry.isDirectory()) {
                    Files.createDirectories(entryPath);
                } else {
                    Files.createDirectories(entryPath.getParent());
                    Files.copy(zis, entryPath, StandardCopyOption.REPLACE_EXISTING);
                }
                zis.closeEntry();
            }
        }
    }

    /**
     * External tar extraction (fallback for .tar.zst on Linux/macOS).
     */
    private static void extractTarExternal(Path archive, Path tempExtract) throws IOException {
        ProcessBuilder pb;
        if (archive.toString().endsWith(".tar.zst")) {
            pb = new ProcessBuilder("tar", "--zstd", "-xf", archive.toString(), "-C", tempExtract.toString());
        } else {
            pb = new ProcessBuilder("tar", "-xzf", archive.toString(), "-C", tempExtract.toString());
        }
        pb.redirectErrorStream(true);

        try {
            Process p = pb.start();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                while (reader.readLine() != null) { /* drain */ }
            }
            int exit = p.waitFor();
            if (exit != 0) {
                throw new IOException("tar extraction failed (exit code " + exit + ")");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted during extraction", e);
        }
    }

    // ==================== Tar Helpers ====================

    private static boolean readFully(InputStream is, byte[] buf, int len) throws IOException {
        int total = 0;
        while (total < len) {
            int n = is.read(buf, total, len - total);
            if (n < 0) return false;
            total += n;
        }
        return true;
    }

    private static boolean isZeroBlock(byte[] buf) {
        for (byte b : buf) {
            if (b != 0) return false;
        }
        return true;
    }

    private static String readTarString(byte[] buf, int offset, int length) {
        int end = offset;
        while (end < offset + length && buf[end] != 0) end++;
        return new String(buf, offset, end - offset);
    }

    private static long readTarOctal(byte[] buf, int offset, int length) {
        String s = readTarString(buf, offset, length).trim();
        if (s.isEmpty()) return 0;
        // Handle GNU tar base-256 encoding
        if ((buf[offset] & 0x80) != 0) {
            long val = 0;
            for (int i = offset; i < offset + length; i++) {
                val = (val << 8) | (buf[i] & 0xFF);
            }
            return val;
        }
        return Long.parseLong(s, 8);
    }

    private static String readTarData(InputStream is, long size) throws IOException {
        byte[] data = new byte[(int) Math.min(size, 65536)];
        int total = 0;
        while (total < data.length) {
            int n = is.read(data, total, data.length - total);
            if (n < 0) break;
            total += n;
        }
        return new String(data, 0, total).trim();
    }

    private static void skipTarData(InputStream is, long size) throws IOException {
        long remaining = size;
        while (remaining > 0) {
            long skipped = is.skip(remaining);
            if (skipped <= 0) {
                // skip() returned 0, try read()
                if (is.read() < 0) break;
                remaining--;
            } else {
                remaining -= skipped;
            }
        }
    }

    private static void skipTarPadding(InputStream is, long size) throws IOException {
        long padded = (size + TAR_BLOCK - 1) & ~(TAR_BLOCK - 1);
        long toSkip = padded - size;
        if (toSkip > 0) skipTarData(is, toSkip);
    }

    // ==================== Python Directory Detection ====================

    private static Path findPythonDir(Path searchRoot) throws IOException {
        // python-build-standalone extracts to a top-level dir containing a "python/" subdir:
        //   Unix:    cpython-{ver}+{release}-{platform}/python/
        //            Contains: bin/, lib/
        //   Windows: cpython-{ver}+{release}-{platform}/python/
        //            Contains: python.exe, Lib/, Scripts/
        try (var stream = Files.walk(searchRoot, 3)) {
            return stream
                    .filter(p -> p.getFileName().toString().equals("python") && Files.isDirectory(p))
                    .filter(p -> {
                        // Unix layout
                        if (Files.exists(p.resolve("bin")) || Files.exists(p.resolve("lib"))) return true;
                        // Windows layout
                        if (Files.exists(p.resolve("python.exe"))) return true;
                        if (Files.exists(p.resolve("Lib"))) return true;
                        if (Files.exists(p.resolve("Scripts"))) return true;
                        return false;
                    })
                    .findFirst()
                    .orElse(null);
        }
    }

    // ==================== File Utilities ====================

    private static void moveDirectoryContents(Path source, Path target) throws IOException {
        try (var stream = Files.list(source)) {
            for (Path entry : stream.collect(java.util.stream.Collectors.toList())) {
                Path dest = target.resolve(entry.getFileName().toString());
                Files.move(entry, dest, StandardCopyOption.REPLACE_EXISTING);
            }
        }
    }

    private static void deleteRecursively(Path dir) throws IOException {
        if (!Files.exists(dir)) return;
        try (var stream = Files.walk(dir)) {
            stream.sorted(Comparator.reverseOrder())
                    .map(Path::toFile)
                    .forEach(File::delete);
        }
    }

    // ==================== Platform Detection ====================

    public static String getPlatformKey() {
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();

        String platform;
        if (os.contains("mac") || os.contains("darwin")) {
            platform = "macos";
        } else if (os.contains("linux")) {
            platform = "linux";
        } else if (os.contains("win")) {
            platform = "windows";
        } else {
            throw new UnsupportedOperationException("Unsupported OS: " + os);
        }

        String architecture;
        if (arch.contains("aarch64") || arch.contains("arm")) {
            architecture = "arm64";
        } else {
            architecture = "x86_64";
        }

        return platform + "-" + architecture;
    }

    private static boolean isWindows() {
        return System.getProperty("os.name").toLowerCase().contains("win");
    }
}
