package io.github.javpower.jpyml;

import io.github.javpower.jpyml.core.PythonEngine;
import io.github.javpower.jpyml.core.PythonRuntime;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Quick verification using the project's .venv Python.
 * Uses singleton engine — tests share the same Python namespace.
 */
@TestMethodOrder(MethodOrderer.MethodName.class)
public class QuickVerifyTest {

    static PythonEngine engine;

    @BeforeAll
    static void setup() throws Exception {
        Path pythonExe = Paths.get(".venv/bin/python3");
        Path jepLib = Paths.get(".venv/lib/python3.13/site-packages/jep/libjep.jnilib");

        Path projectRoot = Paths.get(System.getProperty("user.dir"));
        if (!pythonExe.isAbsolute()) {
            pythonExe = projectRoot.resolve(pythonExe);
            jepLib = projectRoot.resolve(jepLib);
        }

        PythonRuntime.init(pythonExe, jepLib);
        engine = PythonEngine.getInstance();
    }

    @Test
    void test01_BasicEval() throws Exception {
        long result = engine.eval("1 + 2");
        assertEquals(3L, result);
    }

    @Test
    void test02_PutAndGet() throws Exception {
        engine.put("a", 10);
        engine.put("b", 20);
        engine.exec("c = a + b");
        long c = engine.get("c");
        assertEquals(30L, c);
    }

    @Test
    void test03_StringHandling() throws Exception {
        engine.put("name", "jpy-ml");
        String result = engine.eval("name.upper()");
        assertEquals("JPY-ML", result);
    }

    @Test
    void test04_FloatOperations() throws Exception {
        engine.put("pi", 3.14159);
        double rounded = engine.eval("round(pi, 2)");
        assertEquals(3.14, rounded, 0.001);
    }

    @Test
    void test05_ListHandling() throws Exception {
        engine.exec("_jpy_items = [1, 2, 3, 4, 5]");
        List<?> items = engine.get("_jpy_items");
        assertEquals(5, items.size());
    }

    @Test
    void test06_DictHandling() throws Exception {
        engine.exec("_jpy_data = {'name': 'test', 'value': 42}");
        Map<?, ?> data = engine.get("_jpy_data");
        assertEquals("test", data.get("name"));
    }

    @Test
    void test07_MultiLineExec() throws Exception {
        engine.exec(
                "def _jpy_fibonacci(n):\n" +
                "    a, b = 0, 1\n" +
                "    for _ in range(n):\n" +
                "        a, b = b, a + b\n" +
                "    return a\n"
        );
        long result = engine.eval("_jpy_fibonacci(10)");
        assertEquals(55L, result);
    }

    @Test
    void test08_NumpyArray() throws Exception {
        engine.exec("import numpy as np");
        engine.exec("_jpy_arr = np.array([1.0, 2.0, 3.0])");
        List<?> arr = engine.eval("_jpy_arr.tolist()");
        assertEquals(3, arr.size());
        assertEquals(2.0, ((Number) arr.get(1)).doubleValue(), 0.001);
    }

    @Test
    void test09_HasModule() throws Exception {
        assertTrue(engine.hasModule("json"));
        // numpy was already imported in test08
        assertTrue(engine.hasModule("numpy"));
        assertFalse(engine.hasModule("nonexistent_xyz"));
    }

    @Test
    void test10_Callback() throws Exception {
        StringBuilder output = new StringBuilder();
        engine.exec("print('hello from python')", line -> output.append(line), null);
        assertTrue(output.toString().contains("hello from python"));
    }

    @AfterAll
    static void cleanup() {
        if (engine != null) {
            engine.close();
            engine = null;
        }
    }
}
