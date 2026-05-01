package io.github.javpower.jpyml.core;

import io.github.javpower.jpyml.exception.PythonException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.URL;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

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
    private static final String PYTHON_VERSION = "3.13.3";
    private static final String PYTHON_RELEASE_TAG = "20250317";
    private static final AtomicBoolean initialized = new AtomicBoolean(false);
    private static Path runtimeRoot;
    private static Path pythonHome;
    private static Path jepNativeLib;
    private static Path sitePackagesPath;

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
     * @param pythonPath       path to the Python executable (e.g. /path/to/venv/bin/python3)
     * @param jepLibraryPath   path to libjep.jnilib / libjep.so / jep.dll
     */
    public static synchronized void init(Path pythonPath, Path jepLibraryPath) throws IOException {
        if (initialized.get()) return;

        if (!Files.exists(pythonPath)) {
            throw new IOException("Python executable not found: " + pythonPath);
        }
        if (!Files.exists(jepLibraryPath)) {
            throw new IOException("Jep native library not found: " + jepLibraryPath);
        }

        // Derive pythonHome from the python executable
        // venv: pythonPath = venv/bin/python3  ->  pythonHome = venv
        // standalone: pythonPath = python/bin/python3  ->  pythonHome = python
        pythonHome = pythonPath.getParent().getParent();
        jepNativeLib = jepLibraryPath;

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

        sitePackagesPath = computeSitePackages();
        jepNativeLib = findJepNativeLibrary();

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

        ProcessBuilder pb = new ProcessBuilder(cmd)
                .redirectErrorStream(true)
                .directory(runtimeRoot.toFile());

        Process p = pb.start();
        // Stream output
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                log.debug("[pip] {}", line);
            }
        }
        return p.waitFor();
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
     * Check if the runtime is ready (Python + jep + all required packages installed).
     */
    public static boolean isReady() {
        if (!initialized.get()) return false;
        if (jepNativeLib == null || !Files.exists(jepNativeLib)) return false;
        return true;
    }

    /**
     * Ensure Python packages are installed. Checks for import availability and installs missing packages.
     */
    public static int ensurePackages(String... packages) throws IOException, InterruptedException {
        ensureInitialized();
        // Build a list of packages that need installation
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
            p.getOutputStream().close();
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
            Path pip = pythonHome.resolve("bin").resolve("pip");
            if (Files.exists(pip)) return pip;
            Path pip3 = pythonHome.resolve("bin").resolve("pip3");
            if (Files.exists(pip3)) return pip3;
        }
        return null;
    }

    /**
     * Find the jep native library (.jnilib/.so/.dll) installed in site-packages.
     */
    public static Path findJepNativeLibrary() {
        if (jepNativeLib != null) return jepNativeLib;

        Path sp = getSitePackages();
        if (sp == null) return null;

        Path jepDir = sp.resolve("jep");
        if (!Files.exists(jepDir)) return null;

        String libName = isWindows() ? "jep.dll" :
                System.getProperty("os.name").toLowerCase().contains("mac") ?
                        "libjep.jnilib" : "libjep.so";

        Path direct = jepDir.resolve(libName);
        if (Files.exists(direct)) return direct;

        try (var stream = Files.walk(jepDir)) {
            return stream
                    .filter(p -> p.getFileName().toString().equals(libName))
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
            // Clean up partial extraction so next run retries
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
            throw new IOException("Jep native library not found after installation. " +
                    "Expected at: " + (jepNativeLib != null ? jepNativeLib : "site-packages/jep/"));
        }
    }

    /**
     * Install packages from the bundled requirements.txt resource.
     */
    private static void installBundledRequirements() throws IOException {
        Path pythonExe = getPythonExecutable();

        // Extract requirements.txt from classpath to a temp file
        Path tempReq = Files.createTempFile("jpy-ml-requirements", ".txt");
        try (var is = PythonRuntime.class.getResourceAsStream("/requirements.txt")) {
            if (is == null) {
                throw new IOException("requirements.txt not found in classpath");
            }
            Files.copy(is, tempReq, StandardCopyOption.REPLACE_EXISTING);
        }

        try {
            ProcessBuilder pb = new ProcessBuilder(
                    pythonExe.toString(), "-m", "pip", "install", "-r", tempReq.toString()
            ).redirectErrorStream(true);

            Process p = pb.start();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    log.debug("[pip] {}", line);
                }
            }

            int exit = p.waitFor();
            if (exit != 0) {
                throw new IOException("pip install failed (exit code " + exit + ")");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted during pip install", e);
        } finally {
            Files.deleteIfExists(tempReq);
        }
    }

    private static String buildDownloadUrl(String platformKey) {
        String baseUrl = "https://github.com/indygreg/python-build-standalone/releases/download/";
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
        var connection = new URL(url).openConnection();
        int totalSize = connection.getContentLength();
        try (InputStream in = connection.getInputStream();
             OutputStream out = Files.newOutputStream(target)) {
            byte[] buf = new byte[8192];
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

    private static void extractTar(Path archive, Path targetDir, String platformKey) throws IOException {
        // The tar.gz from python-build-standalone extracts to a directory like
        // "python/install" or "cpython-.../python/install"
        // We extract to a temp dir, then move the python/ directory to the right place
        Path tempExtract = runtimeRoot.resolve("downloads").resolve("extract-temp");
        Files.createDirectories(tempExtract);

        ProcessBuilder pb;
        if (archive.toString().endsWith(".tar.zst")) {
            pb = new ProcessBuilder("tar", "--zstd", "-xf", archive.toString(), "-C", tempExtract.toString());
        } else {
            pb = new ProcessBuilder("tar", "-xzf", archive.toString(), "-C", tempExtract.toString());
        }
        pb.redirectErrorStream(true);

        try {
            Process p = pb.start();
            int exit = p.waitFor();
            if (exit != 0) {
                throw new IOException("tar extraction failed (exit code " + exit + ")");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted during extraction", e);
        }

        // Find the "python" directory inside the extracted content
        // python-build-standalone extracts to: cpython-{ver}+{release}-{platform}/python/
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

        ProcessBuilder pb = new ProcessBuilder("unzip", "-q", archive.toString(), "-d", tempExtract.toString());
        pb.redirectErrorStream(true);

        try {
            Process p = pb.start();
            int exit = p.waitFor();
            if (exit != 0) {
                throw new IOException("unzip failed (exit code " + exit + ")");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted during extraction", e);
        }

        Path pythonDir = findPythonDir(tempExtract);
        if (pythonDir == null) {
            throw new IOException("Could not find python directory in extracted archive");
        }

        Path finalDir = targetDir.resolve(platformKey);
        Files.createDirectories(finalDir);
        moveDirectoryContents(pythonDir, finalDir);
        deleteRecursively(tempExtract);
    }

    private static Path findPythonDir(Path searchRoot) throws IOException {
        // python-build-standalone extracts to a top-level dir containing a "python/" subdir
        try (var stream = Files.walk(searchRoot, 3)) {
            return stream
                    .filter(p -> p.getFileName().toString().equals("python") && Files.isDirectory(p))
                    .filter(p -> {
                        Path bin = p.resolve("bin");
                        Path lib = p.resolve("lib");
                        return Files.exists(bin) || Files.exists(lib);
                    })
                    .findFirst()
                    .orElse(null);
        }
    }

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
