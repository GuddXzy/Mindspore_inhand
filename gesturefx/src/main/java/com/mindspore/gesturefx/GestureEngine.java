package com.mindspore.gesturefx;

import android.content.Context;
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
 * Uses point sprites with additive blending for glow effects.
 *
 * Coordinate system: pixel-space orthographic projection.
 * All world coordinates match viewport pixel coordinates directly.
 */
public class GestureEngine implements GLSurfaceView.Renderer {

    private static final int MAX_PARTICLES = 3500;
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

    // ---- Particle physics state (updated from analyzer thread) ----
    private final Object mStateLock = new Object();
    private final List<float[]> mHandLandmarks = new ArrayList<>(); // per-hand keypoints
    private int mPreviewWidth = 1280;
    private int mPreviewHeight = 720;
    private boolean mIsFrontCamera = false;
    private boolean mHasNewData = false;

    // ---- Gesture state ----
    private int mCurrentGesture = GestureDetector.GESTURE_DEFAULT;
    private String mCurrentModeLabel = "Default";

    // ---- Timing ----
    private long mLastFrameTime = 0;

    public GestureEngine(Context context) {
        mContext = context;
        mParticleSystem = new ParticleSystem();
        mGestureDetector = new GestureDetector();
        mVertexData = new float[MAX_PARTICLES * FLOATS_PER_PARTICLE];
    }

    // ===== Public API (called from any thread) =====

    /**
     * Update hand keypoints from the HMS MLHandKeypointAnalyzer callback.
     * Called from the HMS transactor thread.
     */
    public void updateHandKeypoints(List<com.huawei.hms.mlsdk.handkeypoint.MLHandKeypoints> hands,
                                    int previewWidth, int previewHeight, boolean isFront) {
        synchronized (mStateLock) {
            mPreviewWidth = previewWidth;
            mPreviewHeight = previewHeight;
            mIsFrontCamera = isFront;
            mHandLandmarks.clear();

            for (com.huawei.hms.mlsdk.handkeypoint.MLHandKeypoints hand : hands) {
                float[] kps = new float[21 * 3]; // x, y, score per keypoint
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

    /**
     * Get current gesture mode label for UI display.
     */
    public String getCurrentModeLabel() {
        return mCurrentModeLabel;
    }

    // ===== GLSurfaceView.Renderer =====

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        // Transparent background for camera overlay
        GLES20.glClearColor(0f, 0f, 0f, 0f);

        // Enable additive blending for glow effect
        GLES20.glEnable(GLES20.GL_BLEND);
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE);

        // Don't write to depth buffer
        GLES20.glDepthMask(false);

        // Load shaders
        String vertexSource = loadShaderSource(R.raw.vertex_shader);
        String fragmentSource = loadShaderSource(R.raw.fragment_shader);
        mProgram = createProgram(vertexSource, fragmentSource);
        GLES20.glUseProgram(mProgram);

        // Get uniform/attribute locations
        muMVPMatrix = GLES20.glGetUniformLocation(mProgram, "uMVPMatrix");
        muPointSize = GLES20.glGetUniformLocation(mProgram, "uPointSize");
        muTexture = GLES20.glGetUniformLocation(mProgram, "uTexture");
        maPosition = GLES20.glGetAttribLocation(mProgram, "aPosition");
        maColor = GLES20.glGetAttribLocation(mProgram, "aColor");

        // Load glow texture
        mGlowTextureId = loadGlowTexture();

        // Allocate vertex buffer (direct buffer for native access)
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

        // Orthographic projection: pixel-coordinate world space
        // Left=0, Right=width, Bottom=height, Top=0 (Y-down matches screen)
        Matrix.orthoM(mProjMatrix, 0, 0, width, height, 0, -1, 1);
        Matrix.setIdentityM(mViewMatrix, 0);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

        // Compute delta time
        long now = System.nanoTime();
        float deltaSeconds = Math.min((now - mLastFrameTime) / 1_000_000_000f, 0.1f);
        mLastFrameTime = now;

        // Process new hand keypoints and update particle physics
        // (processHandInput handles all particle updates internally)
        processHandInput(deltaSeconds);

        // Pack vertex data
        int count = mParticleSystem.packVertexData(mVertexData);
        if (count == 0) return;

        // Upload to GPU and draw
        mVertexBuffer.position(0);
        mVertexBuffer.put(mVertexData, 0, count * FLOATS_PER_PARTICLE);
        mVertexBuffer.position(0);

        // Set MVP matrix
        Matrix.multiplyMM(mMVPMatrix, 0, mProjMatrix, 0, mViewMatrix, 0);
        GLES20.glUniformMatrix4fv(muMVPMatrix, 1, false, mMVPMatrix, 0);
        GLES20.glUniform1f(muPointSize, 18f);
        GLES20.glUniform1i(muTexture, 0);

        // Bind texture
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, mGlowTextureId);

        // Set vertex attributes (interleaved: pos(4 floats) + color(4 floats))
        mVertexBuffer.position(0);
        GLES20.glVertexAttribPointer(maPosition, 4, GLES20.GL_FLOAT, false, PARTICLE_STRIDE, mVertexBuffer);
        GLES20.glEnableVertexAttribArray(maPosition);

        mVertexBuffer.position(4);
        GLES20.glVertexAttribPointer(maColor, 4, GLES20.GL_FLOAT, false, PARTICLE_STRIDE, mVertexBuffer);
        GLES20.glEnableVertexAttribArray(maColor);

        // Single draw call for all particles
        GLES20.glDrawArrays(GLES20.GL_POINTS, 0, count);

        GLES20.glDisableVertexAttribArray(maPosition);
        GLES20.glDisableVertexAttribArray(maColor);
    }

    // ===== Internal Methods =====

    private void processHandInput(float deltaSeconds) {
        List<float[]> landmarks;
        int prevWidth, prevHeight;
        boolean isFront;

        synchronized (mStateLock) {
            if (!mHasNewData) return;
            mHasNewData = false;
            landmarks = new ArrayList<>(mHandLandmarks);
            prevWidth = mPreviewWidth;
            prevHeight = mPreviewHeight;
            isFront = mIsFrontCamera;
        }

        // Scale factors from camera image coords to viewport pixel coords
        float scaleX = (float) mViewportWidth / (float) prevWidth;
        float scaleY = (float) mViewportHeight / (float) prevHeight;

        // Process each hand
        float[][] targets = new float[landmarks.size()][];
        int[] gestureTypes = new int[landmarks.size()];
        float[][] palmCenters = new float[landmarks.size()][];
        float[][] beamDirections = new float[landmarks.size()][];

        for (int hIdx = 0; hIdx < landmarks.size(); hIdx++) {
            float[] kps = landmarks.get(hIdx);

            // Convert all keypoints to viewport coordinates
            float[][] screenKps = new float[21][3];
            for (int i = 0; i < 21; i++) {
                float x = kps[i * 3];
                float y = kps[i * 3 + 1];
                if (isFront) x = prevWidth - x; // mirror for front camera
                screenKps[i][0] = x * scaleX;
                screenKps[i][1] = y * scaleY;
                screenKps[i][2] = kps[i * 3 + 2]; // score
            }

            // Detect gesture
            int gesture = mGestureDetector.detect(screenKps);
            gestureTypes[hIdx] = gesture;

            // Compute palm center: average of wrist + 5 MCP joints
            float palmX = 0, palmY = 0;
            int[] palmIndices = {0, 1, 5, 9, 13, 17};
            for (int idx : palmIndices) {
                palmX += screenKps[idx][0];
                palmY += screenKps[idx][1];
            }
            palmX /= palmIndices.length;
            palmY /= palmIndices.length;
            palmCenters[hIdx] = new float[]{palmX, palmY};

            // Trail target: index fingertip (8) and middle fingertip (12) average
            float tipX = (screenKps[8][0] + screenKps[12][0]) / 2f;
            float tipY = (screenKps[8][1] + screenKps[12][1]) / 2f;
            targets[hIdx] = new float[]{tipX, tipY};

            // Beam direction: from index MCP (5) to index tip (8)
            float dx = screenKps[8][0] - screenKps[5][0];
            float dy = screenKps[8][1] - screenKps[5][1];
            float len = (float) Math.sqrt(dx * dx + dy * dy);
            if (len > 0.1f) {
                beamDirections[hIdx] = new float[]{dx / len, dy / len};
            } else {
                beamDirections[hIdx] = new float[]{1f, 0f};
            }

            // Apply gesture effects to particles
            int prevGesture = mCurrentGesture;
            if (gesture != GestureDetector.GESTURE_DEFAULT
                && gesture != prevGesture
                && gesture != GestureDetector.GESTURE_DEFAULT) {
                mParticleSystem.applyGesture(hIdx, gesture, palmX, palmY,
                    beamDirections[hIdx][0], beamDirections[hIdx][1]);
            }
            mCurrentGesture = gesture;

            // Update mode label
            switch (gesture) {
                case GestureDetector.GESTURE_OPEN_PALM:
                    mCurrentModeLabel = "Palm Explosion";
                    break;
                case GestureDetector.GESTURE_FIST:
                    mCurrentModeLabel = "Fist Energy Orb";
                    break;
                case GestureDetector.GESTURE_GUN:
                    mCurrentModeLabel = "Gun Beam";
                    break;
                default:
                    mCurrentModeLabel = "Finger Trail";
                    break;
            }
        }

        // Update particle system with processed targets for trail mode
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

        // Shaders can be deleted after linking
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

        // Texture filtering
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        // Load from drawable resource
        Bitmap bitmap = BitmapFactory.decodeResource(mContext.getResources(), R.drawable.particle_glow);
        if (bitmap != null) {
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0);
            bitmap.recycle();
        }

        return textureId[0];
    }
}
