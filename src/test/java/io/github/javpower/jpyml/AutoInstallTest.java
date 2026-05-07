package io.github.javpower.jpyml;

import io.github.javpower.jpyml.core.PythonEngine;
import io.github.javpower.jpyml.core.PythonRuntime;
import io.github.javpower.jpyml.ml.model.Device;
import io.github.javpower.jpyml.ml.model.Model;
import io.github.javpower.jpyml.ml.training.EpochMetric;
import io.github.javpower.jpyml.ml.training.TrainingConfig;
import io.github.javpower.jpyml.ml.training.TrainingResult;
import org.junit.jupiter.api.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end test for PythonRuntime auto-download mode.
 * Verifies: platform detection, Python download/extraction, pip install, engine init.
 * <p>
 * Used by CI (GitHub Actions) to validate Windows + Linux + macOS.
 */
@TestMethodOrder(MethodOrderer.MethodName.class)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class AutoInstallTest {

    private static final boolean IS_WINDOWS =
            System.getProperty("os.name").toLowerCase().contains("win");

    @Test
    void test01_PlatformDetection() {
        String key = PythonRuntime.getPlatformKey();
        System.out.println("[CI] Platform key: " + key);
        System.out.println("[CI] OS: " + System.getProperty("os.name"));
        System.out.println("[CI] Arch: " + System.getProperty("os.arch"));
        System.out.println("[CI] Java: " + System.getProperty("java.version"));

        assertNotNull(key);
        assertFalse(key.isEmpty());

        if (IS_WINDOWS) {
            assertTrue(key.startsWith("windows-"), "Expected windows-* but got: " + key);
        } else {
            String os = System.getProperty("os.name").toLowerCase();
            if (os.contains("mac") || os.contains("darwin")) {
                assertTrue(key.startsWith("macos-"), "Expected macos-* but got: " + key);
            } else {
                assertTrue(key.startsWith("linux-"), "Expected linux-* but got: " + key);
            }
        }
    }

    @Test
    void test02_RuntimeRoot() {
        Path root = PythonRuntime.getRuntimeRoot();
        assertNotNull(root);
        System.out.println("[CI] Runtime root: " + root);
        assertTrue(root.toString().endsWith(".jpy-ml"),
                "Runtime root should end with .jpy-ml: " + root);
    }

    @Test
    void test03_InitAutoDownload() throws Exception {
        System.out.println("[CI] Starting PythonRuntime.init() (auto-download mode)...");

        PythonRuntime.init();
        assertTrue(PythonRuntime.isInitialized(), "Runtime should be initialized");

        Path exe = PythonRuntime.getPythonExecutable();
        System.out.println("[CI] Python executable: " + exe);
        assertNotNull(exe, "Python executable path should not be null");
        assertTrue(Files.exists(exe),
                "Python executable should exist at: " + exe);

        if (IS_WINDOWS) {
            assertTrue(exe.toString().endsWith("python.exe"),
                    "Windows python should end with python.exe: " + exe);
        } else {
            assertTrue(exe.getFileName().toString().equals("python3"),
                    "Unix python should be named python3: " + exe);
        }
    }

    @Test
    void test04_SitePackages() {
        Path sp = PythonRuntime.getSitePackages();
        System.out.println("[CI] Site-packages: " + sp);
        assertNotNull(sp, "Site-packages path should not be null");
        assertTrue(Files.exists(sp),
                "Site-packages directory should exist: " + sp);

        if (IS_WINDOWS) {
            // Windows: pythonHome/Lib/site-packages
            assertTrue(sp.toString().contains("Lib"),
                    "Windows site-packages should be under Lib/: " + sp);
        } else {
            // Unix: pythonHome/lib/python3.x/site-packages
            assertTrue(sp.toString().contains("lib") && sp.toString().contains("python"),
                    "Unix site-packages should be under lib/pythonX.Y/: " + sp);
        }
    }

    @Test
    void test05_JepNativeLibrary() {
        Path jepLib = PythonRuntime.findJepNativeLibrary();
        System.out.println("[CI] Jep native lib: " + jepLib);
        assertNotNull(jepLib, "Jep native library should be found");
        assertTrue(Files.exists(jepLib),
                "Jep native library should exist at: " + jepLib);

        String name = jepLib.getFileName().toString().toLowerCase();
        if (IS_WINDOWS) {
            assertTrue(name.equals("jep.dll"),
                    "Windows jep should be jep.dll: " + name);
        } else {
            String os = System.getProperty("os.name").toLowerCase();
            if (os.contains("mac")) {
                assertTrue(name.equals("libjep.jnilib"),
                        "macOS jep should be libjep.jnilib: " + name);
            } else {
                assertTrue(name.equals("libjep.so"),
                        "Linux jep should be libjep.so: " + name);
            }
        }
    }

    @Test
    void test06_EngineEval() throws Exception {
        PythonEngine engine = PythonEngine.getInstance();
        long result = engine.eval("1 + 2");
        assertEquals(3L, result);
        System.out.println("[CI] Engine eval: 1 + 2 = " + result);
    }

    @Test
    void test07_ExecAndGet() throws Exception {
        PythonEngine engine = PythonEngine.getInstance();
        engine.put("_ci_a", 10);
        engine.put("_ci_b", 20);
        engine.exec("_ci_c = _ci_a + _ci_b");
        long c = engine.get("_ci_c");
        assertEquals(30L, c);
        System.out.println("[CI] Put/exec/get: 10 + 20 = " + c);
    }

    @Test
    void test08_NumpyImport() throws Exception {
        PythonEngine engine = PythonEngine.getInstance();
        engine.exec("import numpy as np");
        engine.exec("_ci_arr = np.array([1.0, 2.0, 3.0])");
        double sum = engine.eval("float(_ci_arr.sum())");
        assertEquals(6.0, sum, 0.001);
        System.out.println("[CI] NumPy: array sum = " + sum);
    }

    @Test
    void test09_PythonVersion() throws Exception {
        PythonEngine engine = PythonEngine.getInstance();
        String version = engine.eval("import sys; sys.version.split()[0]", String.class);
        System.out.println("[CI] Python version: " + version);
        assertTrue(version.startsWith("3.12"),
                "Expected Python 3.12.x but got: " + version);
    }

    @Test
    void test10_InstalledPackages() throws Exception {
        PythonEngine engine = PythonEngine.getInstance();
        engine.exec("import jep");
        engine.exec("import numpy");

        String jepVer = engine.eval("jep.__version__", String.class);
        String npVer = engine.eval("numpy.__version__", String.class);
        System.out.println("[CI] jep version: " + jepVer);
        System.out.println("[CI] numpy version: " + npVer);

        assertNotNull(jepVer);
        assertNotNull(npVer);
    }

    @Test
    void test11_PipInstall() throws Exception {
        // Test pip install works by installing a tiny package
        int exit = PythonRuntime.pipInstall("certifi");
        assertEquals(0, exit, "pip install certifi should succeed");
        System.out.println("[CI] pip install certifi: exit=" + exit);
    }

    @Test
    void test12_RuntimeStructure() throws Exception {
        Path root = PythonRuntime.getRuntimeRoot();
        Path pythonDir = root.resolve("python");

        System.out.println("[CI] === Runtime structure ===");
        System.out.println("[CI] Root: " + root);
        System.out.println("[CI] Python dir exists: " + Files.exists(pythonDir));

        // List top-level content of python dir for debugging
        if (Files.exists(pythonDir)) {
            String platformKey = PythonRuntime.getPlatformKey();
            Path platformDir = pythonDir.resolve(platformKey);
            if (Files.exists(platformDir)) {
                String listing = Files.list(platformDir)
                        .limit(20)
                        .map(p -> p.getFileName().toString())
                        .sorted()
                        .collect(Collectors.joining(", "));
                System.out.println("[CI] " + platformKey + "/ contents: " + listing);
            }
        }

        assertTrue(Files.exists(pythonDir),
                "python/ directory should exist under runtime root");
    }

    @Test
    void test13_CvDependencyLazyInstall() throws Exception {
        System.out.println("[CI] Testing CV dependency lazy install (ultralytics, opencv, etc.)...");
        // Creating a Model triggers DependencyManager.ensure("cv") which installs
        // ultralytics, opencv, onnx, onnxruntime, timm lazily.
        // yolov8n.pt will be auto-downloaded by ultralytics on first use.
        try (Model model = new Model("yolov8n.pt")) {
            assertNotNull(model.getTaskType());
            System.out.println("[CI] Model loaded, task type: " + model.getTaskType());
            System.out.println("[CI] CV dependencies installed successfully");
        }
    }

    @Test
    void test14_TrainWithCallback() throws Exception {
        System.out.println("[CI] Starting YOLO training test (2 epochs, CPU)...");

        try (Model model = new Model("yolov8n.pt")) {
            TrainingConfig config = new TrainingConfig()
                    .dataConfig("coco128.yaml")
                    .epochs(2)
                    .imageSize(320)
                    .batchSize(16)
                    .device(Device.cpu())
                    .project("runs/ci_test")
                    .name("auto_install_train")
                    .val(false)
                    .plots(false);

            AtomicInteger callbackCount = new AtomicInteger(0);

            TrainingResult result = model.train(config, (epoch, log) -> {
                callbackCount.incrementAndGet();
                System.out.println("  [callback] epoch " + epoch + ": " + log);
            });

            assertNotNull(result.getBestModelPath());
            assertFalse(result.getBestModelPath().isEmpty());
            assertTrue(result.getCompletedEpochs() > 0);
            assertFalse(result.getEpochMetrics().isEmpty(), "Should have epoch metrics");

            System.out.println("[CI] Training complete: " + result);
            System.out.println("[CI] Callbacks received: " + callbackCount.get());
            System.out.println("[CI] Best model: " + result.getBestModelPath());
            System.out.println("[CI] Epochs completed: " + result.getCompletedEpochs());
            for (EpochMetric m : result.getEpochMetrics()) {
                System.out.println("[CI]   " + m);
            }
        }
    }

    @AfterAll
    static void cleanup() {
        System.out.println("[CI] Shutting down PythonRuntime");
        PythonRuntime.shutdown();
    }
}
