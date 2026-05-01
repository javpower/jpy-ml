package io.github.javpower.jpyml.ml.result;
import java.util.List;
import java.util.Collections;
public class KeypointCollection {
    // COCO 17: nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder,
    //          l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle
    private final List<Keypoint> keypoints;
    public KeypointCollection(List<Keypoint> keypoints) {
        this.keypoints = Collections.unmodifiableList(keypoints);
    }
    public Keypoint get(int index) { return keypoints.get(index); }
    public List<Keypoint> getAll() { return keypoints; }
    public int size() { return keypoints.size(); }
    public Keypoint getNose() { return keypoints.get(0); }
    public Keypoint getLeftEye() { return keypoints.get(1); }
    public Keypoint getRightEye() { return keypoints.get(2); }
    public float averageConfidence() {
        return (float) keypoints.stream().mapToDouble(Keypoint::confidence).average().orElse(0);
    }
}
