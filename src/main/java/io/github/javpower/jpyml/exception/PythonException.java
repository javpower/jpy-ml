package io.github.javpower.jpyml.exception;


/**
 * Exception thrown when Python execution fails.
 */
public class PythonException extends RuntimeException {

    public PythonException(String message) {
        super(message);
    }

    public PythonException(String message, Throwable cause) {
        super(message, cause);
    }

    public PythonException(Throwable cause) {
        super(cause);
    }
}
