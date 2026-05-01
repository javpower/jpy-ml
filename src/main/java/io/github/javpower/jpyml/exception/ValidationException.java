package io.github.javpower.jpyml.exception;

/**
 * Exception thrown when model validation fails.
 */
public class ValidationException extends JpyMlException {
    public ValidationException(String msg) { super(msg); }
    public ValidationException(String msg, Throwable cause) { super(msg, cause); }
}
