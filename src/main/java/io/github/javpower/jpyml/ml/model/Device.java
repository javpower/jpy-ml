package io.github.javpower.jpyml.ml.model;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Compute device for Ultralytics inference and training.
 * Supports CPU, MPS (Apple Silicon), CUDA GPU, and multi-GPU configurations.
 */
public class Device {
    private final String pythonValue;

    private Device(String pythonValue) {
        this.pythonValue = pythonValue;
    }

    public static Device cpu() { return new Device("cpu"); }
    public static Device mps() { return new Device("mps"); }

    /** Single GPU by index, e.g. gpu(0) → "0" */
    public static Device gpu(int index) { return new Device(String.valueOf(index)); }

    /** CUDA device, e.g. cuda(0) → "cuda:0" */
    public static Device cuda(int index) { return new Device("cuda:" + index); }

    /** Multi-GPU, e.g. multiGpu(0, 1) → "[0, 1]" */
    public static Device multiGpu(int... indices) {
        List<Integer> list = Arrays.stream(indices).boxed().collect(Collectors.toList());
        return new Device(list.toString());
    }

    /** Advanced: pass any Ultralytics-compatible device string */
    public static Device fromString(String value) { return new Device(value); }

    public String toPython() { return pythonValue; }

    @Override
    public String toString() { return "Device(" + pythonValue + ")"; }
}
