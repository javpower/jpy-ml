package io.github.javpower.jpyml.core;

import io.github.javpower.jpyml.exception.PythonException;
import jep.*;

import java.io.Closeable;
import java.nio.file.Path;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Thread-safe Python engine backed by Jep.
 * <p>
 * Uses a singleton SharedInterpreter with ReentrantLock for mutual exclusion.
 * Jep's SharedInterpreter is thread-bound, so all operations are serialized.
 * <p>
 * Usage:
 * <pre>
 *   PythonRuntime.init();
 *   PythonEngine engine = PythonEngine.getInstance();
 *   engine.put("a", 10);
 *   engine.exec("b = a + 20");
 *   long b = engine.get("b");
 * </pre>
 */
public class PythonEngine implements Closeable {

    private static final Logger log = LoggerFactory.getLogger(PythonEngine.class);
    private static final ReentrantLock lock = new ReentrantLock();
    private static PythonEngine instance;
    private static boolean jepConfigured = false;
    private static boolean sharedInterpCreated = false;

    private final Jep interpreter;
    private volatile boolean closed = false;

    private PythonEngine(Jep interpreter) {
        this.interpreter = interpreter;
    }

    /**
     * Get the singleton PythonEngine instance.
     * Creates it on first call, requires PythonRuntime to be initialized.
     */
    public static synchronized PythonEngine getInstance() throws JepException {
        if (instance == null || instance.closed) {
            log.info("Creating PythonEngine instance");
            instance = createInternal(new JepConfig());
        }
        return instance;
    }

    /**
     * Shutdown the singleton engine. Called by PythonRuntime.shutdown().
     */
    public static synchronized void shutdown() {
        if (instance != null) {
            instance.close();
            instance = null;
        }
    }

    /**
     * Create a new engine (closes the previous one if any).
     * Use this if you need a fresh Python namespace.
     */
    public static synchronized PythonEngine create() throws JepException {
        log.info("Creating new PythonEngine (replacing previous)");
        return create(new JepConfig());
    }

    /**
     * Create with custom config.
     */
    public static synchronized PythonEngine create(JepConfig config) throws JepException {
        if (instance != null && !instance.closed) {
            instance.close();
        }
        PythonScriptLoader.reset();
        instance = createInternal(config);
        return instance;
    }

    private static PythonEngine createInternal(JepConfig config) throws JepException {
        if (!PythonRuntime.isInitialized()) {
            throw new PythonException(
                    "PythonRuntime not initialized. Call PythonRuntime.init() first.");
        }

        // Configure MainInterpreter to find jep native library (once only)
        if (!jepConfigured) {
            Path jepLib = PythonRuntime.findJepNativeLibrary();
            if (jepLib != null) {
                try {
                    MainInterpreter.setJepLibraryPath(jepLib.toString());
                } catch (IllegalStateException e) {
                    log.debug("Jep library path already set: {}", e.getMessage());
                }
            }
            jepConfigured = true;
        }

        // Add site-packages to include path
        String includePath = PythonRuntime.getJepIncludePath();
        if (includePath != null && !includePath.isEmpty()) {
            config.addIncludePaths(includePath);
        }

        config.redirectStdout(System.out);
        config.redirectStdErr(System.err);

        // SharedInterpreter: set config before first instantiation only
        if (!sharedInterpCreated) {
            SharedInterpreter.setConfig(config);
        }
        Jep interp = new SharedInterpreter();
        sharedInterpCreated = true;
        log.info("SharedInterpreter created successfully");

        // Python 3.13+ / PyTorch GIL safety: ensure GIL is held and check version
        try {
            interp.exec("import sys");
            int major = interp.getValue("sys.version_info.major", Integer.class);
            int minor = interp.getValue("sys.version_info.minor", Integer.class);
            if (major == 3 && minor >= 13) {
                String pyVersion = interp.getValue("sys.version", String.class);
                log.warn("Python {} detected. PyTorch/JEP may have GIL compatibility issues with 3.13+. " +
                        "Consider using Python 3.11 or 3.12 for maximum stability.", pyVersion.substring(0, 5));
                interp.exec("import _thread; _thread.allocate_lock()");
            }
        } catch (Exception e) {
            log.debug("Python version check skipped: {}", e.getMessage());
        }

        // Inject venv site-packages into sys.path at high priority
        // and filter out system Homebrew site-packages that may conflict with venv packages
        String sitePackages = PythonRuntime.getJepIncludePath();
        if (sitePackages != null && !sitePackages.isEmpty()) {
            interp.exec("import sys");
            interp.set("_jpy_site_pkg", sitePackages);
            interp.exec(
                "sys.path = [p for p in sys.path if 'site-packages' not in p or p == _jpy_site_pkg]"
            );
            interp.exec("if _jpy_site_pkg not in sys.path: sys.path.insert(0, _jpy_site_pkg)");
            interp.exec("del _jpy_site_pkg");
        }

        return new PythonEngine(interp);
    }

    /**
     * Evaluate a Python expression and return the result.
     */
    @SuppressWarnings("unchecked")
    public <T> T eval(String expression) throws JepException {
        ensureOpen();
        lock.lock();
        try {
            return (T) interpreter.getValue(expression);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Evaluate with explicit type.
     */
    public <T> T eval(String expression, Class<T> type) throws JepException {
        ensureOpen();
        lock.lock();
        try {
            return interpreter.getValue(expression, type);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Execute a Python statement (can be multi-line).
     */
    public void exec(String code) throws JepException {
        ensureOpen();
        lock.lock();
        try {
            interpreter.exec(code);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Execute with output callbacks.
     */
    public void exec(String code, Consumer<String> stdout, Consumer<String> stderr) throws JepException {
        ensureOpen();
        lock.lock();
        boolean stdoutCaptured = false;
        boolean stderrCaptured = false;
        try {
            if (stdout != null) {
                interpreter.exec("import sys, io");
                interpreter.exec("_jpy_cap_stdout = sys.stdout");
                interpreter.exec("sys.stdout = io.StringIO()");
                stdoutCaptured = true;
            }
            if (stderr != null) {
                interpreter.exec("import sys, io");
                interpreter.exec("_jpy_cap_stderr = sys.stderr");
                interpreter.exec("sys.stderr = io.StringIO()");
                stderrCaptured = true;
            }

            interpreter.exec(code);

            if (stdout != null) {
                String out = interpreter.getValue("sys.stdout.getvalue()", String.class);
                if (out != null) {
                    for (String line : out.split("\n")) {
                        if (!line.isEmpty()) stdout.accept(line);
                    }
                }
            }
            if (stderr != null) {
                String err = interpreter.getValue("sys.stderr.getvalue()", String.class);
                if (err != null) {
                    for (String line : err.split("\n")) {
                        if (!line.isEmpty()) stderr.accept(line);
                    }
                }
            }
        } finally {
            // Always restore stdout/stderr even if an exception occurs
            try {
                if (stdoutCaptured) {
                    interpreter.exec("sys.stdout = _jpy_cap_stdout");
                }
            } catch (Exception e) {
                    log.debug("Failed to restore stdout: {}", e.getMessage());
                }
                try {
                    if (stderrCaptured) {
                        interpreter.exec("sys.stderr = _jpy_cap_stderr");
                    }
                } catch (Exception e) {
                    log.debug("Failed to restore stderr: {}", e.getMessage());
                }
            lock.unlock();
        }
    }

    /**
     * Set a Java object as a Python variable.
     */
    public void put(String name, Object value) throws JepException {
        ensureOpen();
        lock.lock();
        try {
            interpreter.set(name, value);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Get a Python variable as a Java object.
     */
    @SuppressWarnings("unchecked")
    public <T> T get(String name) throws JepException {
        ensureOpen();
        lock.lock();
        try {
            return (T) interpreter.getValue(name);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Get with explicit type.
     */
    public <T> T get(String name, Class<T> type) throws JepException {
        ensureOpen();
        lock.lock();
        try {
            return interpreter.getValue(name, type);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Import a Python module.
     */
    public void importModule(String module) throws JepException {
        if (!module.matches("[a-zA-Z_][a-zA-Z0-9_.]*")) {
            throw new IllegalArgumentException("Invalid module name: " + module);
        }
        exec("import " + module);
    }

    /**
     * Run a Python script from classpath resources.
     */
    public void runResourceScript(String resourcePath) throws JepException {
        ensureOpen();
        lock.lock();
        try {
            try (var is = getClass().getClassLoader().getResourceAsStream(resourcePath)) {
                if (is == null) {
                    throw new JepException("Resource not found: " + resourcePath);
                }
                String script = new String(is.readAllBytes());
                interpreter.exec(script);
            }
        } catch (java.io.IOException e) {
            throw new JepException("Failed to read resource: " + resourcePath, e);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Check if a Python module is importable.
     */
    public boolean hasModule(String module) {
        try {
            put("_jpy_check_mod", module);
            exec("__import__(_jpy_check_mod)");
            return true;
        } catch (Exception e) {
            return false;
        } finally {
            try {
                exec("del _jpy_check_mod");
            } catch (Exception e) {
                log.debug("Failed to cleanup _jpy_check_mod: {}", e.getMessage());
            }
        }
    }

    /**
     * Install a pip package if the module isn't available.
     */
    public boolean ensurePackage(String module, String pipName) {
        if (hasModule(module)) return true;
        try {
            return PythonRuntime.pipInstall(pipName) == 0;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Get the underlying Jep instance for advanced usage.
     */
    public Jep getInterpreter() {
        return interpreter;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("PythonEngine is closed");
        }
    }

    @Override
    public void close() {
        lock.lock();
        try {
            if (!closed && interpreter != null) {
                closed = true;
                log.info("Closing PythonEngine");
                try {
                    interpreter.close();
                } catch (Exception e) {
                    log.debug("Error closing interpreter: {}", e.getMessage());
                }
            }
        } finally {
            lock.unlock();
        }
    }
}
