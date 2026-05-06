//package io.github.javpower.jpyml;
//
//import io.github.javpower.jpyml.core.PythonEngine;
//import io.github.javpower.jpyml.core.PythonRuntime;
//import org.junit.jupiter.api.BeforeAll;
//import org.junit.jupiter.api.Test;
//
//import java.util.List;
//import java.util.Map;
//
//import static org.junit.jupiter.api.Assertions.*;
//
//public class PythonEngineTest {
//
//    @BeforeAll
//    static void setup() throws Exception {
//        PythonRuntime.init();
//    }
//
//    @Test
//    void testBasicEval() throws Exception {
//        try (PythonEngine engine = PythonEngine.create()) {
//            long result = engine.eval("1 + 2");
//            assertEquals(3L, result);
//        }
//    }
//
//    @Test
//    void testPutAndGet() throws Exception {
//        try (PythonEngine engine = PythonEngine.create()) {
//            engine.put("a", 10);
//            engine.put("b", 20);
//            engine.exec("c = a + b");
//            long c = engine.get("c");
//            assertEquals(30L, c);
//        }
//    }
//
//    @Test
//    void testStringHandling() throws Exception {
//        try (PythonEngine engine = PythonEngine.create()) {
//            engine.put("name", "jpy-ml");
//            String result = engine.eval("name.upper()");
//            assertEquals("JPY-ML", result);
//        }
//    }
//
//    @Test
//    void testFloatOperations() throws Exception {
//        try (PythonEngine engine = PythonEngine.create()) {
//            engine.put("pi", 3.14159);
//            double rounded = engine.eval("round(pi, 2)");
//            assertEquals(3.14, rounded, 0.001);
//        }
//    }
//
//    @Test
//    void testListHandling() throws Exception {
//        try (PythonEngine engine = PythonEngine.create()) {
//            engine.exec("items = [1, 2, 3, 4, 5]");
//            @SuppressWarnings("unchecked")
//            List<Number> items = engine.get("items");
//            assertEquals(5, items.size());
//            assertEquals(3L, items.get(2).longValue());
//        }
//    }
//
//    @Test
//    void testDictHandling() throws Exception {
//        try (PythonEngine engine = PythonEngine.create()) {
//            engine.exec("data = {'name': 'test', 'value': 42}");
//            @SuppressWarnings("unchecked")
//            Map<String, Object> data = engine.get("data");
//            assertEquals("test", data.get("name"));
//            assertEquals(42L, data.get("value"));
//        }
//    }
//
//    @Test
//    void testMultiLineExec() throws Exception {
//        try (PythonEngine engine = PythonEngine.create()) {
//            engine.exec(
//                    "def fibonacci(n):\n" +
//                    "    a, b = 0, 1\n" +
//                    "    for _ in range(n):\n" +
//                    "        a, b = b, a + b\n" +
//                    "    return a\n"
//            );
//            long result = engine.eval("fibonacci(10)");
//            assertEquals(55L, result);
//        }
//    }
//
//    @Test
//    void testNumpyArray() throws Exception {
//        try (PythonEngine engine = PythonEngine.create()) {
//            engine.exec("import numpy as np");
//            engine.exec("arr = np.array([1.0, 2.0, 3.0])");
//            @SuppressWarnings("unchecked")
//            List<Number> arr = engine.eval("arr.tolist()");
//            assertEquals(3, arr.size());
//            assertEquals(2.0, arr.get(1).doubleValue(), 0.001);
//        }
//    }
//
//    @Test
//    void testCallback() throws Exception {
//        try (PythonEngine engine = PythonEngine.create()) {
//            StringBuilder output = new StringBuilder();
//            engine.exec("print('hello from python')", line -> output.append(line), null);
//            assertTrue(output.toString().contains("hello from python"));
//        }
//    }
//
//    @Test
//    void testHasModule() throws Exception {
//        try (PythonEngine engine = PythonEngine.create()) {
//            assertTrue(engine.hasModule("json"));
//            assertTrue(engine.hasModule("os"));
//            assertFalse(engine.hasModule("nonexistent_module_xyz"));
//        }
//    }
//
//    @Test
//    void testMultipleEngines() throws Exception {
//        // With SharedInterpreter singleton, create() returns the same underlying interpreter.
//        // Variables from one engine instance are visible to getInstance() and vice versa.
//        PythonEngine e1 = PythonEngine.getInstance();
//        e1.put("_jpy_shared_val", "visible_everywhere");
//
//        PythonEngine e2 = PythonEngine.getInstance();
//        String result = e2.eval("_jpy_shared_val");
//        assertEquals("visible_everywhere", result);
//    }
//}
