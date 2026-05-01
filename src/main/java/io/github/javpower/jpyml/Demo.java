package io.github.javpower.jpyml;

import io.github.javpower.jpyml.ml.model.Model;
import io.github.javpower.jpyml.ml.result.ClassPrediction;
import io.github.javpower.jpyml.ml.result.DetectionResult;
import io.github.javpower.jpyml.ml.result.InferenceResult;
import io.github.javpower.jpyml.core.PythonEngine;
import io.github.javpower.jpyml.core.PythonRuntime;

import java.util.List;

/**
 * Quick demo: run with {@code mvn compile exec:java -Dexec.mainClass=io.github.javpower.jpyml.Demo}
 */
public class Demo {

    public static void main(String[] args) throws Exception {
        System.out.println("=== jpy-ml Demo ===\n");

        // 1. Initialize Python runtime
        //    Production mode: auto-downloads python-build-standalone on first run
        //    Dev mode: PythonRuntime.init(pythonPath, jepLibPath) with local venv
        PythonRuntime.init();
        PythonRuntime.ensurePackages("ultralytics");

        // 2. Basic Python usage
        System.out.println("--- Basic Python ---");
        try (PythonEngine engine = PythonEngine.create()) {
            engine.put("name", "World");
            engine.exec("greeting = f'Hello, {name}!'");
            String greeting = engine.eval("greeting");
            System.out.println(greeting);

            // Numpy
            engine.exec("import numpy as np");
            engine.exec("data = np.arange(10).reshape(2, 5).tolist()");
            System.out.println("Numpy array: " + engine.eval("data"));
        }

        // 3. YOLO Detection (if model/image available)
        if (args.length > 0) {
            String imagePath = args[0];
            String model = args.length > 1 ? args[1] : "yolov8n.pt";

            System.out.println("\n--- YOLO Detection ---");
            try (Model yolo = new Model(model)) {
                System.out.println("Model: " + yolo.getModelInfo());
                System.out.println("Task: " + yolo.getTaskType());
                System.out.println("Classes: " + yolo.getClassNames());

                // Single image
                InferenceResult result = yolo.predict(imagePath);
                System.out.println("Source: " + result.getSourcePath());
                System.out.println("Speed: " + result.getSpeed());
                System.out.println("Found " + result.count() + " objects");

                if (result instanceof DetectionResult dr) {
                    for (ClassPrediction pred : dr.getBoxes()) {
                        System.out.println("  " + pred);
                    }
                }

                // Batch inference
                if (args.length > 2) {
                    System.out.println("\n--- Batch Detection ---");
                    List<InferenceResult> batch = yolo.predict(List.of(args).subList(0, Math.min(args.length, 5)));
                    System.out.println("Batch results: " + batch.size() + " images");
                    for (int i = 0; i < batch.size(); i++) {
                        System.out.println("  Image " + i + ": " + batch.get(i).count() + " objects");
                    }
                }
            }
        } else {
            System.out.println("\nTo test YOLO detection, run:");
            System.out.println("  mvn compile exec:java -Dexec.mainClass=io.github.javpower.jpyml.Demo -Dexec.args=\"image1.jpg image2.jpg yolov8n.pt\"");
        }

        System.out.println("\n=== Done ===");
    }
}
