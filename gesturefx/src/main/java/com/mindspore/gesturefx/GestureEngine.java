package com.mindspore.gesturefx;

import android.content.Context;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.GLUtils;
import android.opengl.Matrix;

import com.mindspore.gesturefx.R;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * OpenGL ES 2.0 renderer for gesture-driven particle effects.
 *
 * Coordinate mapping replicates LensEnginePreview.onLayout() algorithm
 * for precise overlay of particles on the camera image.
 */
public class GestureEngine implements GLSurfaceView.Renderer {

    private static final int MAX_PARTICLES = 5000;
    private static final int FLOATS_PER_PARTICLE = 8; // x,y,z,w, r,g,b,a
    private static final int BYTES_PER_FLOAT = 4;
    private static final int PARTICLE_STRIDE = FLOATS_PER_PARTICLE * BYTES_PER_FLOAT;

    private final Context mContext;
    private final ParticleSystem mParticleSystem;
    private final GestureDetector mGestureDetector;

    // ---- OpenGL handles ----
    private int mProgram;
    private int muMVPMatrix;
    private int muPointSize;
    private int muTexture;
    private int maPosition;
    private int maColor;
    private int mGlowTextureId;

    // ---- Matrices ----
    private final float[] mProjMatrix = new float[16];
    private final float[] mViewMatrix = new float[16];
    private final float[] mMVPMatrix = new float[16];

    // ---- Vertex buffer ----
    private final float[] mVertexData;
    private FloatBuffer mVertexBuffer;

    // ---- Viewport ----
    private int mViewportWidth;
    private int mViewportHeight;

    // ---- Camera info (set from Activity) ----
    private int mCameraImageWidth = 1280;   // camera sensor width
    private int mCameraImageHeight = 720;   // camera sensor height
    private boolean mIsPortrait = true;
    private boolean mIsFrontCamera = false;

    // ---- Particle physics state (updated from analyzer thread) ----
    private final Object mStateLock = new Object();
    private final List<float[]> mHandLandmarks = new ArrayList<>();
    private boolean mHasNewData = false;

    // ---- Gesture state ----
    private int[] mCurrentGestures = new int[]{0, 0};  // per-hand gesture
    private String mCurrentModeLabel = "指尖轨迹";

    // ---- Timing ----
    private long mLastFrameTime = 0;

    public GestureEngine(Context context) {
        mContext = context;
        mParticleSystem = new ParticleSystem();
        mGestureDetector = new GestureDetector();
        mVertexData = new float[MAX_PARTICLES * FLOATS_PER_PARTICLE];
        mIsPortrait = context.getResources().getConfiguration().orientation
                == Configuration.ORIENTATION_PORTRAIT;
    }

    // ===== Public API =====

    /**
     * Update hand keypoints from the HMS MLHandKeypointAnalyzer callback.
     * Called from the HMS transactor thread.
     */
    public void updateHandKeypoints(List<com.huawei.hms.mlsdk.handkeypoint.MLHandKeypoints> hands,
                                    int previewWidth, int previewHeight, boolean isFront) {
        synchronized (mStateLock) {
            mCameraImageWidth = previewWidth;
            mCameraImageHeight = previewHeight;
            mIsFrontCamera = isFront;
            mHandLandmarks.clear();

            for (com.huawei.hms.mlsdk.handkeypoint.MLHandKeypoints hand : hands) {
                float[] kps = new float[21 * 3];
                List<com.huawei.hms.mlsdk.handkeypoint.MLHandKeypoint> keypoints = hand.getHandKeypoints();
                for (int i = 0; i < keypoints.size() && i < 21; i++) {
                    com.huawei.hms.mlsdk.handkeypoint.MLHandKeypoint kp = keypoints.get(i);
                    kps[i * 3]     = kp.getPointX();
                    kps[i * 3 + 1] = kp.getPointY();
                    kps[i * 3 + 2] = kp.getScore();
                }
                mHandLandmarks.add(kps);
            }
            mHasNewData = true;
        }
    }

    /** Update orientation (called from Activity on configuration change) */
    public void setOrientation(boolean isPortrait) {
        mIsPortrait = isPortrait;
    }

    public String getCurrentModeLabel() {
        return mCurrentModeLabel;
    }

    // ===== GLSurfaceView.Renderer =====

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        GLES20.glClearColor(0f, 0f, 0f, 0f);
        GLES20.glEnable(GLES20.GL_BLEND);
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE);
        GLES20.glDepthMask(false);

        String vertexSource = loadShaderSource(R.raw.vertex_shader);
        String fragmentSource = loadShaderSource(R.raw.fragment_shader);
        mProgram = createProgram(vertexSource, fragmentSource);
        GLES20.glUseProgram(mProgram);

        muMVPMatrix = GLES20.glGetUniformLocation(mProgram, "uMVPMatrix");
        muPointSize = GLES20.glGetUniformLocation(mProgram, "uPointSize");
        muTexture = GLES20.glGetUniformLocation(mProgram, "uTexture");
        maPosition = GLES20.glGetAttribLocation(mProgram, "aPosition");
        maColor = GLES20.glGetAttribLocation(mProgram, "aColor");

        mGlowTextureId = loadGlowTexture();

        ByteBuffer bb = ByteBuffer.allocateDirect(MAX_PARTICLES * FLOATS_PER_PARTICLE * BYTES_PER_FLOAT);
        bb.order(ByteOrder.nativeOrder());
        mVertexBuffer = bb.asFloatBuffer();

        mLastFrameTime = System.nanoTime();
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        GLES20.glViewport(0, 0, width, height);
        mViewportWidth = width;
        mViewportHeight = height;

        // Orthographic projection: pixel-space mapping
        Matrix.orthoM(mProjMatrix, 0, 0, width, height, 0, -1, 1);
        Matrix.setIdentityM(mViewMatrix, 0);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

        long now = System.nanoTime();
        float deltaSeconds = Math.min((now - mLastFrameTime) / 1_000_000_000f, 0.1f);
        mLastFrameTime = now;

        processHandInput(deltaSeconds);

        int count = mParticleSystem.packVertexData(mVertexData);
        if (count == 0) return;

        mVertexBuffer.position(0);
        mVertexBuffer.put(mVertexData, 0, count * FLOATS_PER_PARTICLE);
        mVertexBuffer.position(0);

        Matrix.multiplyMM(mMVPMatrix, 0, mProjMatrix, 0, mViewMatrix, 0);
        GLES20.glUniformMatrix4fv(muMVPMatrix, 1, false, mMVPMatrix, 0);
        GLES20.glUniform1f(muPointSize, 40f);  // larger point size (was 32)
        GLES20.glUniform1i(muTexture, 0);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, mGlowTextureId);

        mVertexBuffer.position(0);
        GLES20.glVertexAttribPointer(maPosition, 4, GLES20.GL_FLOAT, false, PARTICLE_STRIDE, mVertexBuffer);
        GLES20.glEnableVertexAttribArray(maPosition);

        mVertexBuffer.position(4);
        GLES20.glVertexAttribPointer(maColor, 4, GLES20.GL_FLOAT, false, PARTICLE_STRIDE, mVertexBuffer);
        GLES20.glEnableVertexAttribArray(maColor);

        GLES20.glDrawArrays(GLES20.GL_POINTS, 0, count);

        GLES20.glDisableVertexAttribArray(maPosition);
        GLES20.glDisableVertexAttribArray(maColor);
    }

    // ===== Coordinate Mapping (replicates LensEnginePreview.onLayout) =====

    /**
     * Convert camera image coordinates to GL viewport pixel coordinates.
     * Uses the same algorithm as LensEnginePreview.onLayout().
     */
    private float[] mapImageToView(float imageX, float imageY) {
        // Camera image dimensions (sensor frame)
        int imgW = mCameraImageWidth;   // 1280
        int imgH = mCameraImageHeight;  // 720

        // In portrait mode, the image is effectively rotated 90°
        int previewW, previewH;
        if (mIsPortrait) {
            previewW = imgH;  // 720
            previewH = imgW;  // 1280
        } else {
            previewW = imgW;
            previewH = imgH;
        }

        int viewW = mViewportWidth;
        int viewH = mViewportHeight;

        // Compute child layout dimensions (same as LensEnginePreview.onLayout)
        float widthRatio = (float) viewW / (float) previewW;
        float heightRatio = (float) viewH / (float) previewH;

        int childW, childH;
        int childXOff = 0, childYOff = 0;

        if (widthRatio > heightRatio) {
            childW = viewW;
            childH = (int) ((float) previewH * widthRatio);
            childYOff = (childH - viewH) / 2;
        } else {
            childW = (int) ((float) previewW * heightRatio);
            childH = viewH;
            childXOff = (childW - viewW) / 2;
        }

        // Map from image coordinate to child view coordinate
        float scaleX = (float) childW / (float) previewW;
        float scaleY = (float) childH / (float) previewH;

        // Portrait: camera image (imgW×imgH = 1280×720 landscape) is rotated 90°
        // After rotation: imageY (0-720) → childX axis (previewW=720)
        //                imageX (0-1280) → childY axis (previewH=1280)
        float childX, childY;
        if (mIsPortrait) {
            // Front camera: mirror the horizontal axis (imageY in portrait)
            float yImg = mIsFrontCamera ? (imgH - imageY) : imageY;
            childX = (yImg / (float) imgH) * (float) childW;
            childY = (imageX / (float) imgW) * (float) childH;
        } else {
            // Landscape: imageX→X, imageY→Y, front camera mirrors X
            float x = mIsFrontCamera ? (imgW - imageX) : imageX;
            childX = (x / (float) imgW) * (float) childW;
            childY = (imageY / (float) imgH) * (float) childH;
        }

        // Offset from child coordinate to view coordinate (crop)
        float viewX = childX - childXOff;
        float viewY = childY - childYOff;

        return new float[]{viewX, viewY};
    }

    // ===== Hand Input Processing =====

    private void processHandInput(float deltaSeconds) {
        List<float[]> landmarks;
        boolean isFront;

        synchronized (mStateLock) {
            if (!mHasNewData) return;
            mHasNewData = false;
            landmarks = new ArrayList<>(mHandLandmarks);
            isFront = mIsFrontCamera;
        }

        float[][] targets = new float[landmarks.size()][];
        int[] gestureTypes = new int[landmarks.size()];
        float[][] palmCenters = new float[landmarks.size()][];
        float[][] beamDirections = new float[landmarks.size()][];

        for (int hIdx = 0; hIdx < landmarks.size(); hIdx++) {
            float[] kps = landmarks.get(hIdx);

            // Convert all 21 keypoints to viewport coordinates
            float[][] viewKps = new float[21][3];
            for (int i = 0; i < 21; i++) {
                float ix = kps[i * 3];
                float iy = kps[i * 3 + 1];
                float[] mapped = mapImageToView(ix, iy);
                viewKps[i][0] = mapped[0];
                viewKps[i][1] = mapped[1];
                viewKps[i][2] = kps[i * 3 + 2];
            }

            // Detect gesture
            int gesture = mGestureDetector.detect(viewKps);
            gestureTypes[hIdx] = gesture;

            // Compute palm center: wrist + 5 MCPs
            float palmX = 0, palmY = 0;
            int[] palmIdx = {0, 1, 5, 9, 13, 17};
            for (int idx : palmIdx) {
                palmX += viewKps[idx][0];
                palmY += viewKps[idx][1];
            }
            palmX /= palmIdx.length;
            palmY /= palmIdx.length;
            palmCenters[hIdx] = new float[]{palmX, palmY};

            // Beam direction: index MCP(5) → index tip(8)
            float dx = viewKps[8][0] - viewKps[5][0];
            float dy = viewKps[8][1] - viewKps[5][1];
            float len = (float) Math.sqrt(dx * dx + dy * dy);
            beamDirections[hIdx] = len > 0.1f
                ? new float[]{dx / len, dy / len}
                : new float[]{1f, 0f};

            // Build targets array: [tipAvgX, tipAvgY, kp0X, kp0Y, kp1X, kp1Y, ...]
            // Include all major keypoints for full hand particle coverage
            int[] spawnKps = {4, 8, 12, 16, 20, 5, 9, 13, 17, 1, 6, 10, 14, 18};
            targets[hIdx] = new float[2 + spawnKps.length * 2];
            // Primary tip average (index + middle)
            targets[hIdx][0] = (viewKps[8][0] + viewKps[12][0]) / 2f;
            targets[hIdx][1] = (viewKps[8][1] + viewKps[12][1]) / 2f;
            // All spawn keypoints
            for (int k = 0; k < spawnKps.length; k++) {
                targets[hIdx][2 + k * 2]     = viewKps[spawnKps[k]][0];
                targets[hIdx][2 + k * 2 + 1] = viewKps[spawnKps[k]][1];
            }

            // Apply gesture transitions
            int prevGesture = (hIdx < mCurrentGestures.length) ? mCurrentGestures[hIdx] : 0;
            if (gesture != prevGesture && gesture != GestureDetector.GESTURE_DEFAULT) {
                mParticleSystem.applyGesture(hIdx, gesture, palmX, palmY,
                    beamDirections[hIdx][0], beamDirections[hIdx][1]);
            }
            if (hIdx < mCurrentGestures.length) {
                mCurrentGestures[hIdx] = gesture;
            }
        }

        // Chinese mode labels
        if (gestureTypes.length > 0) {
            switch (gestureTypes[0]) {
                case GestureDetector.GESTURE_OPEN_PALM:
                    mCurrentModeLabel = "掌心爆发"; break;
                case GestureDetector.GESTURE_FIST:
                    mCurrentModeLabel = "握拳能量球"; break;
                case GestureDetector.GESTURE_GUN:
                    mCurrentModeLabel = "枪形光束"; break;
                default:
                    mCurrentModeLabel = "指尖轨迹"; break;
            }
        }

        mParticleSystem.update(deltaSeconds, targets, gestureTypes, palmCenters, beamDirections);
    }

    // ===== Shader Utilities =====

    private String loadShaderSource(int resourceId) {
        try {
            InputStream is = mContext.getResources().openRawResource(resourceId);
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append('\n');
            }
            reader.close();
            return sb.toString();
        } catch (Exception e) {
            throw new RuntimeException("Failed to load shader: " + resourceId, e);
        }
    }

    private int createProgram(String vertexSource, String fragmentSource) {
        int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexSource);
        int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource);

        int program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vertexShader);
        GLES20.glAttachShader(program, fragmentShader);
        GLES20.glLinkProgram(program);

        int[] linkStatus = new int[1];
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linkStatus, 0);
        if (linkStatus[0] == 0) {
            String log = GLES20.glGetProgramInfoLog(program);
            GLES20.glDeleteProgram(program);
            throw new RuntimeException("Program link failed: " + log);
        }

        GLES20.glDeleteShader(vertexShader);
        GLES20.glDeleteShader(fragmentShader);
        return program;
    }

    private int loadShader(int type, String source) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, source);
        GLES20.glCompileShader(shader);

        int[] compiled = new int[1];
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) {
            String log = GLES20.glGetShaderInfoLog(shader);
            GLES20.glDeleteShader(shader);
            throw new RuntimeException("Shader compile failed: " + log);
        }
        return shader;
    }

    private int loadGlowTexture() {
        int[] textureId = new int[1];
        GLES20.glGenTextures(1, textureId, 0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId[0]);

        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        Bitmap bitmap = BitmapFactory.decodeResource(mContext.getResources(), R.drawable.particle_glow);
        if (bitmap != null) {
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0);
            bitmap.recycle();
        }
        return textureId[0];
    }
}
