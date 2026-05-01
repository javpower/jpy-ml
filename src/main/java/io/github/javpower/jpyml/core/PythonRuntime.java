package io.github.javpower.jpyml.core;

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

    private static final String RUNTIME_DIR_NAME = ".jpy-ml";
    private static final String PYTHON_VERSION = "3.12.6";
    private static final String PYTHON_RELEASE_TAG = "20240909";
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

        System.out.println("[jpy-ml] Using existing Python: " + pythonPath);
        System.out.println("[jpy-ml] Jep native lib: " + jepNativeLib);
        System.out.println("[jpy-ml] Site-packages: " + sitePackagesPath);

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
            System.out.println("[jpy-ml] First-time setup: downloading Python " + PYTHON_VERSION + " for " + platformKey + "...");
            downloadAndExtractPython(platformKey);
            System.out.println("[jpy-ml] Python runtime ready at " + pythonHome);
        } else {
            System.out.println("[jpy-ml] Using cached Python runtime at " + pythonHome);
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
                System.out.println("[pip] " + line);
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
                System.out.println("[pip] " + line);
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
            throw new IllegalStateException("PythonRuntime not initialized. Call PythonRuntime.init() first.");
        }
    }

    private static Path resolvePipExecutable() {
        Path exe = getPythonExecutable();
        if (exe != null && Files.exists(exe)) return exe;
        // Fallback: try the venv's pip directly
        if (pythonHome != null) {
            Path pip = pythonHome.resolve("bin").resolve("pip");
            if (Files.exists(pip)) return exe;
            Path pip3 = pythonHome.resolve("bin").resolve("pip3");
            if (Files.exists(pip3)) return exe;
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

    private static void configureJepPaths() {
        jepNativeLib = findJepNativeLibrary();
        if (jepNativeLib != null) {
            System.setProperty("jep.library.path", jepNativeLib.toString());
        }

        Path pythonLibDir = pythonHome.resolve("lib");
        Path pythonBinDir = getPythonExecutable().getParent();
        String extraPaths = pythonBinDir + File.pathSeparator + pythonLibDir;
        System.setProperty("java.library.path",
                System.getProperty("java.library.path", "") + File.pathSeparator + extraPaths);
    }

    private static void downloadAndExtractPython(String platformKey) throws IOException {
        String downloadUrl = buildDownloadUrl(platformKey);
        String archiveName = downloadUrl.substring(downloadUrl.lastIndexOf('/') + 1);
        Path archivePath = runtimeRoot.resolve("downloads").resolve(archiveName);

        Files.createDirectories(archivePath.getParent());

        // Download
        if (!Files.exists(archivePath)) {
            System.out.println("[jpy-ml] Downloading from " + downloadUrl);
            downloadFile(downloadUrl, archivePath);
            System.out.println("[jpy-ml] Download complete: " + archivePath);
        }

        // Extract
        System.out.println("[jpy-ml] Extracting...");
        Path extractTarget = runtimeRoot.resolve("python");
        Files.createDirectories(extractTarget);

        if (archiveName.endsWith(".tar.zst") || archiveName.endsWith(".tar.gz")) {
            extractTar(archivePath, extractTarget, platformKey);
        } else if (archiveName.endsWith(".zip")) {
            extractZip(archivePath, extractTarget, platformKey);
        }

        // Install jep into the embedded Python
        System.out.println("[jpy-ml] Installing jep bridge package...");
        Path pythonExe = getPythonExecutable();
        ProcessBuilder pb = new ProcessBuilder(
                pythonExe.toString(), "-m", "pip", "install", "jep"
        ).redirectErrorStream(true);
        Process p = pb.start();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println("[pip] " + line);
            }
        }
        try {
            int exit = p.waitFor();
            if (exit != 0) {
                throw new IOException("Failed to install jep package (exit code " + exit + ")");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while installing jep", e);
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
        try (InputStream in = new URL(url).openStream()) {
            Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
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
