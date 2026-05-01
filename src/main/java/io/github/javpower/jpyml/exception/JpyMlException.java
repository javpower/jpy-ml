package io.github.javpower.jpyml.exception;

/**
 * Base exception for all jpy-ml exceptions.
 * Provides a common hierarchy for all custom exceptions in the framework.
 */
public class JpyMlException extends RuntimeException {

    public JpyMlException(String message) {
        super(message);
    }

    public JpyMlException(String message, Throwable cause) {
        super(message, cause);
    }
}
