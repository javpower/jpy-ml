package io.github.javpower.jpyml;

import io.github.javpower.jpyml.core.PythonEngine;
import io.github.javpower.jpyml.core.PythonRuntime;
import org.junit.jupiter.api.*;

import java.nio.file.Files;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests PythonRuntime.init() auto-download mode (no .venv).
 * Used by CI to verify environment setup works on Linux/Windows.
 */
@TestMethodOrder(MethodOrderer.MethodName.class)
public class AutoInstallTest {

    @Test
    void test01_Init() throws Exception {
        PythonRuntime.init();
        assertTrue(PythonRuntime.isInitialized());
        assertTrue(Files.exists(PythonRuntime.getPythonExecutable()));
        assertNotNull(PythonRuntime.findJepNativeLibrary());
    }

    @Test
    void test02_EngineEval() throws Exception {
        PythonEngine engine = PythonEngine.getInstance();
        long result = engine.eval("1 + 2");
        assertEquals(3L, result);
    }

    @Test
    void test03_ExecAndGet() throws Exception {
        PythonEngine engine = PythonEngine.getInstance();
        engine.put("_ci_a", 10);
        engine.put("_ci_b", 20);
        engine.exec("_ci_c = _ci_a + _ci_b");
        long c = engine.get("_ci_c");
        assertEquals(30L, c);
    }

    @Test
    void test04_NumpyImport() throws Exception {
        PythonEngine engine = PythonEngine.getInstance();
        engine.exec("import numpy as np");
        engine.exec("_ci_arr = np.array([1.0, 2.0, 3.0])");
        double sum = engine.eval("float(_ci_arr.sum())");
        assertEquals(6.0, sum, 0.001);
    }

    @AfterAll
    static void cleanup() {
        PythonRuntime.shutdown();
    }
}
