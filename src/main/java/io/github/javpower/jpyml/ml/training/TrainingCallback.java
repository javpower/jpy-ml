package io.github.javpower.jpyml.ml.training;

/**
 * Callback interface for receiving real-time training progress events.
 * All methods are called on a background monitor thread — implementors must
 * synchronize access to any shared state.
 */
@FunctionalInterface
public interface TrainingCallback {

    /**
     * Called after each training epoch completes.
     *
     * @param epoch   1-based epoch number
     * @param logLine raw JSON line containing epoch metrics
     */
    void onEpoch(int epoch, String logLine);

    /**
     * Called when training completes or fails.
     *
     * @param error null/empty if successful, error message if training failed
     */
    default void onComplete(String error) {}

    /**
     * Called if the training was cancelled via stopTraining().
     */
    default void onCancelled() {}
}
