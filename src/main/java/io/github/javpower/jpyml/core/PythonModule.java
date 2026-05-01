package io.github.javpower.jpyml.core;

/**
 * Isolated Python module context.
 * <p>
 * Creates a separate namespace so variables don't pollute __main__.
 * Each module is backed by the thread's Jep interpreter but uses
 * a dedicated dict as the namespace.
 *
 * @deprecated This class is not currently used in the framework. May be removed in future versions.
 */
@Deprecated
public class PythonModule implements AutoCloseable {

    private final PythonEngine engine;
    private final String moduleName;

    PythonModule(PythonEngine engine, String moduleName) {
        this.engine = engine;
        this.moduleName = moduleName;
    }

    /**
     * Get the module name.
     */
    public String getName() {
        return moduleName;
    }

    /**
     * Execute code in this module's namespace.
     */
    public void exec(String code) throws Exception {
        engine.put("_jpy_mod_code", code);
        engine.exec(String.format(
                "exec(_jpy_mod_code, %s.__dict__)", moduleName
        ));
    }

    /**
     * Set a variable in this module.
     */
    public void set(String name, Object value) throws Exception {
        engine.put("_jpy_mod_val", value);
        engine.exec(String.format("setattr(%s, '%s', _jpy_mod_val)", moduleName, name));
    }

    /**
     * Get a variable from this module.
     */
    public <T> T get(String name) throws Exception {
        return engine.eval(String.format("%s.%s", moduleName, name));
    }

    @Override
    public void close() {
        // Modules don't hold separate resources
    }
}
