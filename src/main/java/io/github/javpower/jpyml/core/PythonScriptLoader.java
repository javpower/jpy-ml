package io.github.javpower.jpyml.core;
import java.util.Collections;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import jep.JepException;
public class PythonScriptLoader {
    private static final Set<String> loaded = Collections.newSetFromMap(new ConcurrentHashMap<>());
    public static void ensureLoaded(PythonEngine engine, String scriptName) throws JepException {
        if (!loaded.contains(scriptName)) {
            engine.runResourceScript("python/" + scriptName);
            loaded.add(scriptName);
        }
    }
    public static void reset() { loaded.clear(); }
}
