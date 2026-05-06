package io.github.javpower.jpyml.llm;

import io.github.javpower.jpyml.core.PythonEngine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Lazy dependency installer for optional ML features.
 * Packages are installed on first use via pip, only for the requested group.
 */
public class DependencyManager {

    private static final Logger log = LoggerFactory.getLogger(DependencyManager.class);

    private static final Map<String, String[]> GROUPS = Map.of(
            "llm", new String[]{
                    "transformers", "peft", "trl", "accelerate",
                    "datasets", "safetensors", "huggingface_hub"
            }
    );

    private static final Set<String> verified = ConcurrentHashMap.newKeySet();

    /**
     * Ensure all Python packages for the given group are installed.
     * Installs missing packages via pip on first call; subsequent calls are no-ops.
     *
     * @param group dependency group name (e.g., "llm")
     */
    public static void ensure(String group) {
        if (verified.contains(group)) return;
        String[] deps = GROUPS.get(group);
        if (deps == null) throw new IllegalArgumentException("Unknown dependency group: " + group);

        PythonEngine engine = PythonEngine.getInstance();
        List<String> failed = new ArrayList<>();
        for (String dep : deps) {
            boolean ok = engine.ensurePackage(dep, dep);
            if (!ok) {
                failed.add(dep);
                log.warn("Failed to install package: {}", dep);
            }
        }
        if (!failed.isEmpty()) {
            throw new RuntimeException("Failed to install packages for group '" + group + "': " + failed);
        }
        verified.add(group);
        log.info("Dependency group '{}' verified", group);
    }
}
