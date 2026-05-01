package io.github.javpower.jpyml;

import io.github.javpower.jpyml.core.PythonRuntime;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class PythonRuntimeTest {

    @Test
    void testPlatformDetection() {
        String platform = PythonRuntime.getPlatformKey();
        assertNotNull(platform);
        assertTrue(platform.equals("macos-arm64") || platform.equals("macos-x86_64") ||
                   platform.equals("linux-x86_64") || platform.equals("linux-aarch64") ||
                   platform.equals("windows-x86_64"),
                "Unexpected platform: " + platform);
    }

    @Test
    void testRuntimeRootDefault() {
        String expected = System.getProperty("user.home") + "/.jpy-ml";
        assertEquals(expected, PythonRuntime.getRuntimeRoot().toString());
    }

    @Test
    void testPythonExecutableNotNull() {
        // If init was already called (by other tests), pythonHome should be set
        if (PythonRuntime.isInitialized()) {
            assertNotNull(PythonRuntime.getPythonExecutable());
            assertNotNull(PythonRuntime.getPythonHome());
        }
    }
}
