package io.github.javpower.jpyml.core;

import jep.*;

import java.io.Closeable;
import java.nio.file.Path;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Consumer;

/**
 * Thread-safe Python engine backed by Jep.
 * <p>
 * Uses a singleton SharedInterpreter with ReadWriteLock for concurrency.
 * Read operations (eval, get) can run concurrently;
 * write operations (exec, put) are mutually exclusive.
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

    private static final ReadWriteLock lock = new ReentrantReadWriteLock();
    private static PythonEngine instance;
    private static boolean jepConfigured = false;
    private static boolean sharedInterpCreated = false;

    private final Jep interpreter;
    private boolean closed = false;

    private PythonEngine(Jep interpreter) {
        this.interpreter = interpreter;
    }

    /**
     * Get the singleton PythonEngine instance.
     * Creates it on first call, requires PythonRuntime to be initialized.
     */
    public static synchronized PythonEngine getInstance() throws JepException {
        if (instance == null || instance.closed) {
            instance = createInternal(new JepConfig());
        }
        return instance;
    }

    /**
     * Create a new engine (closes the previous one if any).
     * Use this if you need a fresh Python namespace.
     */
    public static synchronized PythonEngine create() throws JepException {
        return create(new JepConfig());
    }

    /**
     * Create with custom config.
     */
    public static synchronized PythonEngine create(JepConfig config) throws JepException {
        if (instance != null && !instance.closed) {
            instance.close();
        }
        instance = createInternal(config);
        return instance;
    }

    private static PythonEngine createInternal(JepConfig config) throws JepException {
        if (!PythonRuntime.isInitialized()) {
            throw new IllegalStateException(
                    "PythonRuntime not initialized. Call PythonRuntime.init() first.");
        }

        // Configure MainInterpreter to find jep native library (once only)
        if (!jepConfigured) {
            Path jepLib = PythonRuntime.findJepNativeLibrary();
            if (jepLib != null) {
                try {
                    MainInterpreter.setJepLibraryPath(jepLib.toString());
                } catch (IllegalStateException ignored) {
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

        // Inject venv site-packages into sys.path at high priority
        // and filter out system Homebrew site-packages that may conflict with venv packages
        String sitePackages = PythonRuntime.getJepIncludePath();
        if (sitePackages != null && !sitePackages.isEmpty()) {
            interp.exec("import sys");
            // Only remove Homebrew/system site-packages (not stdlib paths)
            interp.exec(
                "sys.path = [p for p in sys.path if 'site-packages' not in p or p == '" + sitePackages + "']"
            );
            interp.exec("if '" + sitePackages + "' not in sys.path: sys.path.insert(0, '" + sitePackages + "')");
        }

        return new PythonEngine(interp);
    }

    /**
     * Evaluate a Python expression and return the result.
     */
    @SuppressWarnings("unchecked")
    public <T> T eval(String expression) throws JepException {
        ensureOpen();
        lock.readLock().lock();
        try {
            return (T) interpreter.getValue(expression);
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Evaluate with explicit type.
     */
    public <T> T eval(String expression, Class<T> type) throws JepException {
        ensureOpen();
        lock.readLock().lock();
        try {
            return interpreter.getValue(expression, type);
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Execute a Python statement (can be multi-line).
     */
    public void exec(String code) throws JepException {
        ensureOpen();
        lock.writeLock().lock();
        try {
            interpreter.exec(code);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Execute with output callbacks.
     */
    public void exec(String code, Consumer<String> stdout, Consumer<String> stderr) throws JepException {
        ensureOpen();
        lock.writeLock().lock();
        try {
            if (stdout != null) {
                interpreter.exec("import sys, io");
                interpreter.exec("_jpy_cap_stdout = sys.stdout");
                interpreter.exec("sys.stdout = io.StringIO()");
            }
            if (stderr != null) {
                interpreter.exec("import sys, io");
                interpreter.exec("_jpy_cap_stderr = sys.stderr");
                interpreter.exec("sys.stderr = io.StringIO()");
            }

            interpreter.exec(code);

            if (stdout != null) {
                String out = interpreter.getValue("sys.stdout.getvalue()", String.class);
                if (out != null) {
                    for (String line : out.split("\n")) {
                        if (!line.isEmpty()) stdout.accept(line);
                    }
                }
                interpreter.exec("sys.stdout = _jpy_cap_stdout");
            }
            if (stderr != null) {
                String err = interpreter.getValue("sys.stderr.getvalue()", String.class);
                if (err != null) {
                    for (String line : err.split("\n")) {
                        if (!line.isEmpty()) stderr.accept(line);
                    }
                }
                interpreter.exec("sys.stderr = _jpy_cap_stderr");
            }
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Set a Java object as a Python variable.
     */
    public void put(String name, Object value) throws JepException {
        ensureOpen();
        lock.writeLock().lock();
        try {
            interpreter.set(name, value);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Get a Python variable as a Java object.
     */
    @SuppressWarnings("unchecked")
    public <T> T get(String name) throws JepException {
        ensureOpen();
        lock.readLock().lock();
        try {
            return (T) interpreter.getValue(name);
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Get with explicit type.
     */
    public <T> T get(String name, Class<T> type) throws JepException {
        ensureOpen();
        lock.readLock().lock();
        try {
            return interpreter.getValue(name, type);
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Import a Python module.
     */
    public void importModule(String module) throws JepException {
        exec("import " + module);
    }

    /**
     * Run a Python script from classpath resources.
     */
    public void runResourceScript(String resourcePath) throws JepException {
        ensureOpen();
        lock.writeLock().lock();
        try {
            var is = getClass().getClassLoader().getResourceAsStream(resourcePath);
            if (is == null) {
                throw new JepException("Resource not found: " + resourcePath);
            }
            String script = new String(is.readAllBytes());
            interpreter.exec(script);
        } catch (java.io.IOException e) {
            throw new JepException("Failed to read resource: " + resourcePath, e);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Check if a Python module is importable.
     */
    public boolean hasModule(String module) {
        try {
            exec("__import__('" + module + "')");
            return true;
        } catch (Exception e) {
            return false;
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
        if (!closed && interpreter != null) {
            closed = true;
            try {
                interpreter.close();
            } catch (Exception ignored) {
            }
        }
    }
}
