package io.github.javpower.jpyml.mp;

import io.github.javpower.jpyml.core.PythonEngine;
import io.github.javpower.jpyml.core.PythonScriptLoader;
import io.github.javpower.jpyml.exception.InferenceException;
import jep.JepException;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * MediaPipe engine for hand tracking, face detection, and pose estimation.
 * <p>
 * Usage:
 * <pre>
 *   MediaPipeEngine mp = new MediaPipeEngine();
 *   HandResult hands = mp.detectHands("hand.jpg");
 *   FaceResult faces = mp.detectFace("face.jpg");
 *   PoseLandmarks pose = mp.detectPose("pose.jpg");
 * </pre>
 */
public class MediaPipeEngine implements AutoCloseable {

    private final PythonEngine engine;
    private boolean initialized = false;
    private boolean closed = false;

    public MediaPipeEngine() throws JepException {
        this.engine = PythonEngine.getInstance();
    }

    private void ensureInitialized() throws JepException {
        if (!initialized) {
            PythonScriptLoader.ensureLoaded(engine, "_jpy_mediapipe.py");
            engine.exec("_jpy_mp_modules = jpy_mp_init()");
            initialized = true;
        }
    }

    /**
     * Detect hands in an image.
     *
     * @param imagePath path to the image
     * @return hand detection result
     */
    public HandResult detectHands(String imagePath) throws InferenceException {
        ensureOpen();
        try {
            ensureInitialized();
            engine.put("_jpy_mp_img", imagePath);
            engine.exec("_jpy_mp_result = jpy_mp_detect_hands(_jpy_mp_modules, _jpy_mp_img)");
            @SuppressWarnings("unchecked")
            Map<String, Object> result = engine.eval("_jpy_mp_result");

            int count = ((Number) result.get("count")).intValue();
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> rawHands = (List<Map<String, Object>>) result.get("hands");

            List<HandResult.Hand> hands = new ArrayList<>();
            for (Map<String, Object> rh : rawHands) {
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> rawLandmarks = (List<Map<String, Object>>) rh.get("landmarks");
                List<HandResult.Landmark> landmarks = new ArrayList<>();
                for (Map<String, Object> rl : rawLandmarks) {
                    landmarks.add(new HandResult.Landmark(
                            ((Number) rl.get("x")).floatValue(),
                            ((Number) rl.get("y")).floatValue(),
                            ((Number) rl.get("z")).floatValue()
                    ));
                }
                hands.add(new HandResult.Hand(landmarks));
            }

            return new HandResult(imagePath, hands);

        } catch (Exception e) {
            throw new InferenceException("Hand detection failed on: " + imagePath, e);
        }
    }

    /**
     * Detect face mesh in an image.
     *
     * @param imagePath path to the image
     * @return face detection result
     */
    public FaceResult detectFace(String imagePath) throws InferenceException {
        ensureOpen();
        try {
            ensureInitialized();
            engine.put("_jpy_mp_img", imagePath);
            engine.exec("_jpy_mp_result = jpy_mp_detect_face(_jpy_mp_modules, _jpy_mp_img)");
            @SuppressWarnings("unchecked")
            Map<String, Object> result = engine.eval("_jpy_mp_result");

            int count = ((Number) result.get("count")).intValue();
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> rawFaces = (List<Map<String, Object>>) result.get("faces");

            List<FaceResult.Face> faces = new ArrayList<>();
            for (Map<String, Object> rf : rawFaces) {
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> rawLandmarks = (List<Map<String, Object>>) rf.get("landmarks");
                List<FaceResult.Landmark> landmarks = new ArrayList<>();
                for (Map<String, Object> rl : rawLandmarks) {
                    landmarks.add(new FaceResult.Landmark(
                            ((Number) rl.get("x")).floatValue(),
                            ((Number) rl.get("y")).floatValue(),
                            ((Number) rl.get("z")).floatValue()
                    ));
                }
                faces.add(new FaceResult.Face(landmarks));
            }

            return new FaceResult(imagePath, faces);

        } catch (Exception e) {
            throw new InferenceException("Face detection failed on: " + imagePath, e);
        }
    }

    /**
     * Detect pose in an image.
     *
     * @param imagePath path to the image
     * @return pose detection result
     */
    public PoseResult detectPose(String imagePath) throws InferenceException {
        ensureOpen();
        try {
            ensureInitialized();
            engine.put("_jpy_mp_img", imagePath);
            engine.exec("_jpy_mp_result = jpy_mp_detect_pose(_jpy_mp_modules, _jpy_mp_img)");
            @SuppressWarnings("unchecked")
            Map<String, Object> result = engine.eval("_jpy_mp_result");

            int count = ((Number) result.get("count")).intValue();
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> rawLandmarks = (List<Map<String, Object>>) result.get("landmarks");

            List<PoseResult.Landmark> landmarks = new ArrayList<>();
            for (Map<String, Object> rl : rawLandmarks) {
                landmarks.add(new PoseResult.Landmark(
                        ((Number) rl.get("x")).floatValue(),
                        ((Number) rl.get("y")).floatValue(),
                        ((Number) rl.get("z")).floatValue(),
                        ((Number) rl.get("visibility")).floatValue()
                ));
            }

            return new PoseResult(imagePath, landmarks);

        } catch (Exception e) {
            throw new InferenceException("Pose detection failed on: " + imagePath, e);
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("MediaPipeEngine is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            try {
                engine.exec("_jpy_mp_modules = None");
            } catch (Exception ignored) {
            }
        }
    }

    /**
     * Hand detection result.
     */
    public record HandResult(String sourcePath, List<Hand> hands) {
        public int count() { return hands.size(); }

        public record Hand(List<Landmark> landmarks) {}
        public record Landmark(float x, float y, float z) {}
    }

    /**
     * Face detection result.
     */
    public record FaceResult(String sourcePath, List<Face> faces) {
        public int count() { return faces.size(); }

        public record Face(List<Landmark> landmarks) {}
        public record Landmark(float x, float y, float z) {}
    }

    /**
     * Pose detection result.
     */
    public record PoseResult(String sourcePath, List<Landmark> landmarks) {
        public int count() { return landmarks.size(); }

        public record Landmark(float x, float y, float z, float visibility) {}
    }
}
