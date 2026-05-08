package io.github.javpower.jpyml.core;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Lazy dependency installer for optional ML features.
 * Packages are installed on first use via pip, only for the requested group.
 * <p>
 * Each dependency is a {module, pipSpec} pair:
 *   - module: Python import name (used to check if already installed)
 *   - pipSpec: pip install specifier with version constraint (e.g., "ultralytics>=8.4.45")
 */
public class DependencyManager {

    private static final Logger log = LoggerFactory.getLogger(DependencyManager.class);

    private static final Map<String, String[][]> GROUPS = new LinkedHashMap<>();

    static {
        GROUPS.put("llm", new String[][]{
                {"transformers", "transformers"},
                {"peft", "peft"},
                {"trl", "trl"},
                {"accelerate", "accelerate"},
                {"datasets", "datasets"},
                {"safetensors", "safetensors"},
                {"huggingface_hub", "huggingface_hub"}
        });
        GROUPS.put("cv", new String[][]{
                {"ultralytics", "ultralytics>=8.4.45"},
                {"cv2", "opencv-python>=4.6.0"},
                {"onnx", "onnx>=1.16"},
                {"onnxruntime", "onnxruntime>=1.18"},
                {"timm", "timm>=1.0"},
                {"clip", "git+https://github.com/ultralytics/CLIP.git"}
        });
        GROUPS.put("mediapipe", new String[][]{
                {"mediapipe", "mediapipe>=0.10.0"}
        });
        GROUPS.put("flux", new String[][]{
                {"diffusers", "diffusers>=0.30.0"},
                {"transformers", "transformers>=4.40.0"},
                {"accelerate", "accelerate>=0.30.0"},
                {"sentencepiece", "sentencepiece>=0.2.0"},
                {"protobuf", "protobuf>=4.25.0"},
                {"safetensors", "safetensors>=0.4.0"}
        });
    }

    private static final Set<String> verified = ConcurrentHashMap.newKeySet();

    /**
     * Ensure all Python packages for the given group are installed.
     * Installs missing packages via pip on first call; subsequent calls are no-ops.
     *
     * @param group dependency group name (e.g., "llm", "cv", "mediapipe")
     */
    public static void ensure(String group) {
        if (verified.contains(group)) return;
        String[][] deps = GROUPS.get(group);
        if (deps == null) throw new IllegalArgumentException("Unknown dependency group: " + group);

        PythonEngine engine = PythonEngine.getInstance();
        List<String> failed = new ArrayList<>();
        for (String[] dep : deps) {
            String module = dep[0];
            String pipSpec = dep[1];
            boolean ok = engine.ensurePackage(module, pipSpec);
            if (!ok) {
                failed.add(pipSpec);
                log.warn("Failed to install package: {}", pipSpec);
            }
        }
        if (!failed.isEmpty()) {
            throw new RuntimeException("Failed to install packages for group '" + group + "': " + failed);
        }
        verified.add(group);
        log.info("Dependency group '{}' verified", group);
    }
}
