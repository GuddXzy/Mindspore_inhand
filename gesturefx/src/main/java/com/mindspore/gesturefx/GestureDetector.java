package com.mindspore.gesturefx;

/**
 * Gesture classification from 21 hand keypoints.
 *
 * HMS MLHandKeypoint 21-keypoint convention:
 *   0  = wrist
 *   1  = thumb CMC
 *   2  = thumb MCP
 *   3  = thumb IP
 *   4  = thumb TIP
 *   5  = index MCP
 *   6  = index PIP
 *   7  = index DIP
 *   8  = index TIP
 *   9  = middle MCP
 *   10 = middle PIP
 *   11 = middle DIP
 *   12 = middle TIP
 *   13 = ring MCP
 *   14 = ring PIP
 *   15 = ring DIP
 *   16 = ring TIP
 *   17 = little MCP
 *   18 = little PIP
 *   19 = little DIP
 *   20 = little TIP
 */
public class GestureDetector {

    public static final int GESTURE_DEFAULT = 0;
    public static final int GESTURE_OPEN_PALM = 1;
    public static final int GESTURE_FIST = 2;
    public static final int GESTURE_GUN = 3;

    // ---- Thresholds (tuned larger to reduce false triggers) ----
    private static final float PALM_OPEN_RATIO = 1.8f;
    private static final float FIST_CLOSE_RATIO = 1.1f;  // relaxed from 0.9
    private static final float GUN_EXTENSION_RATIO = 1.5f;
    private static final float GUN_CURL_RATIO = 1.0f;
    private static final float SCORE_THRESHOLD = 0.5f;

    // Debounce: require gesture to persist for N consecutive frames
    private static final int DEBOUNCE_FRAMES = 3;
    private int mStableCounter = 0;
    private int mLastGesture = GESTURE_DEFAULT;

    // MCP joint indices for palm radius calculation
    private static final int[] MCP_INDICES = {1, 5, 9, 13, 17};
    // Fingertip indices
    private static final int[] TIP_INDICES = {4, 8, 12, 16, 20};
    // MCP indices for individual finger extension checks
    private static final int INDEX_MCP = 5;
    private static final int INDEX_TIP = 8;
    private static final int MIDDLE_MCP = 9;
    private static final int MIDDLE_TIP = 12;
    private static final int RING_MCP = 13;
    private static final int RING_TIP = 16;
    private static final int LITTLE_MCP = 17;
    private static final int LITTLE_TIP = 20;
    private static final int THUMB_MCP = 1;
    private static final int THUMB_TIP = 4;
    private static final int WRIST = 0;

    /**
     * Detect gesture from 21 keypoint positions.
     * @param keypoints [21][3] array: [i][0]=x, [i][1]=y, [i][2]=score
     * @return gesture type constant
     */
    public int detect(float[][] keypoints) {
        if (keypoints == null || keypoints.length < 21) {
            return GESTURE_DEFAULT;
        }

        // Check overall hand confidence
        float avgScore = 0;
        for (int i = 0; i < 21; i++) {
            avgScore += keypoints[i][2];
        }
        avgScore /= 21f;
        if (avgScore < SCORE_THRESHOLD) {
            resetDebounce();
            return GESTURE_DEFAULT;
        }

        // Compute palm center: average of wrist + 5 MCPs
        float palmX = 0, palmY = 0;
        for (int idx : MCP_INDICES) {
            palmX += keypoints[idx][0];
            palmY += keypoints[idx][1];
        }
        palmX += keypoints[WRIST][0];
        palmY += keypoints[WRIST][1];
        palmX /= (MCP_INDICES.length + 1);
        palmY /= (MCP_INDICES.length + 1);

        // Compute palm radius: average distance of MCPs from palm center
        float palmRadius = 0;
        for (int idx : MCP_INDICES) {
            float dx = keypoints[idx][0] - palmX;
            float dy = keypoints[idx][1] - palmY;
            palmRadius += (float) Math.sqrt(dx * dx + dy * dy);
        }
        palmRadius /= MCP_INDICES.length;
        if (palmRadius < 1f) palmRadius = 1f; // avoid div by zero

        // Compute average fingertip distance from palm center
        float tipDist = 0;
        for (int idx : TIP_INDICES) {
            float dx = keypoints[idx][0] - palmX;
            float dy = keypoints[idx][1] - palmY;
            tipDist += (float) Math.sqrt(dx * dx + dy * dy);
        }
        tipDist /= TIP_INDICES.length;

        // Extension ratio: fingertip distance / palm radius
        float extensionRatio = tipDist / palmRadius;

        int detectedGesture;

        // ---- Check FIST: all fingertips curled close to palm ----
        if (extensionRatio < FIST_CLOSE_RATIO) {
            detectedGesture = GESTURE_FIST;
        }
        // ---- Check GUN FIRST (before open palm): specific finger pattern ----
        else if (isGunPose(keypoints, palmRadius)) {
            detectedGesture = GESTURE_GUN;
        }
        // ---- Check OPEN PALM: all fingertips spread far from palm ----
        else if (extensionRatio > PALM_OPEN_RATIO) {
            detectedGesture = GESTURE_OPEN_PALM;
        }
        // ---- DEFAULT ----
        else {
            detectedGesture = GESTURE_DEFAULT;
        }

        // Debounce: require gesture to persist for DEBOUNCE_FRAMES
        if (detectedGesture == mLastGesture) {
            mStableCounter++;
            if (mStableCounter >= DEBOUNCE_FRAMES) {
                return detectedGesture;
            }
            // Not yet stable, return default but maintain counter
            return (detectedGesture == GESTURE_DEFAULT) ? GESTURE_DEFAULT : mLastGesture;
        } else {
            mStableCounter = 0;
            mLastGesture = detectedGesture;
            // Return previous stable gesture until new one is confirmed
            return (mLastGesture == GESTURE_DEFAULT || detectedGesture == GESTURE_DEFAULT)
                ? detectedGesture : GESTURE_DEFAULT;
        }
    }

    /**
     * Check for gun pose: thumb and index extended, ring and little curled.
     */
    private boolean isGunPose(float[][] keypoints, float palmRadius) {
        // Index finger extension: tip distance from MCP vs palm radius
        float indexDx = keypoints[INDEX_TIP][0] - keypoints[INDEX_MCP][0];
        float indexDy = keypoints[INDEX_TIP][1] - keypoints[INDEX_MCP][1];
        float indexLen = (float) Math.sqrt(indexDx * indexDx + indexDy * indexDy);
        float indexRatio = indexLen / palmRadius;

        // Thumb extension
        float thumbDx = keypoints[THUMB_TIP][0] - keypoints[THUMB_MCP][0];
        float thumbDy = keypoints[THUMB_TIP][1] - keypoints[THUMB_MCP][1];
        float thumbLen = (float) Math.sqrt(thumbDx * thumbDx + thumbDy * thumbDy);
        float thumbRatio = thumbLen / palmRadius;

        // Ring finger curl: tip close to MCP
        float ringDx = keypoints[RING_TIP][0] - keypoints[RING_MCP][0];
        float ringDy = keypoints[RING_TIP][1] - keypoints[RING_MCP][1];
        float ringLen = (float) Math.sqrt(ringDx * ringDx + ringDy * ringDy);
        float ringRatio = ringLen / palmRadius;

        // Little finger curl
        float littleDx = keypoints[LITTLE_TIP][0] - keypoints[LITTLE_MCP][0];
        float littleDy = keypoints[LITTLE_TIP][1] - keypoints[LITTLE_MCP][1];
        float littleLen = (float) Math.sqrt(littleDx * littleDx + littleDy * littleDy);
        float littleRatio = littleLen / palmRadius;

        // Also check middle finger: should be somewhat curled in gun pose
        float middleDx = keypoints[MIDDLE_TIP][0] - keypoints[MIDDLE_MCP][0];
        float middleDy = keypoints[MIDDLE_TIP][1] - keypoints[MIDDLE_MCP][1];
        float middleLen = (float) Math.sqrt(middleDx * middleDx + middleDy * middleDy);
        float middleRatio = middleLen / palmRadius;

        // Thumb + index extended; ring + little + middle curled
        return thumbRatio > GUN_EXTENSION_RATIO
            && indexRatio > GUN_EXTENSION_RATIO
            && ringRatio < GUN_CURL_RATIO
            && littleRatio < GUN_CURL_RATIO
            && middleRatio < GUN_CURL_RATIO;
    }

    private void resetDebounce() {
        mStableCounter = 0;
        mLastGesture = GESTURE_DEFAULT;
    }

    /**
     * Get human-readable gesture name.
     */
    public static String getGestureName(int gesture) {
        switch (gesture) {
            case GESTURE_OPEN_PALM: return "Open Palm";
            case GESTURE_FIST: return "Fist";
            case GESTURE_GUN: return "Gun";
            default: return "Default";
        }
    }
}
