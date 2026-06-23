package com.mindspore.gesturefx;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.PixelFormat;
import android.net.Uri;
import android.opengl.GLSurfaceView;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;
import android.util.SparseArray;
import android.view.View;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.app.ActivityCompat;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.huawei.hms.mlsdk.common.LensEngine;
import com.huawei.hms.mlsdk.common.MLAnalyzer;
import com.huawei.hms.mlsdk.handkeypoint.MLHandKeypointAnalyzer;
import com.huawei.hms.mlsdk.handkeypoint.MLHandKeypointAnalyzerFactory;
import com.huawei.hms.mlsdk.handkeypoint.MLHandKeypointAnalyzerSetting;
import com.huawei.hms.mlsdk.handkeypoint.MLHandKeypoints;
import com.mindspore.hms.camera.LensEnginePreview;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Gesture Interaction Effects Main Activity
 *
 * Real-time hand keypoint detection with 3D particle effects
 * overlaid on the camera preview. Uses HMS MLHandKeypointAnalyzer
 * for hand tracking and OpenGL ES 2.0 for particle rendering.
 */
@Route(path = "/gesturefx/GestureFxMainActivity")
public class GestureFxMainActivity extends AppCompatActivity implements View.OnClickListener {

    private static final String TAG = "GestureFxMainActivity";

    // ---- Camera ----
    private LensEnginePreview mPreview;
    private LensEngine mLensEngine;
    private MLHandKeypointAnalyzer mHmsAnalyzer;
    private int mLensType = LensEngine.BACK_LENS;
    private boolean isFront = false;

    // ---- OpenGL ----
    private GLSurfaceView mGlSurfaceView;
    private GestureEngine mGestureEngine;

    // ---- UI ----
    private TextView mModeLabel;

    // ---- Camera preview dimensions (720p, 16:9) ----
    private static final int PREVIEW_WIDTH = 1280;
    private static final int PREVIEW_HEIGHT = 720;
    private static final float PREVIEW_FPS = 30.0f;

    // ---- Permissions ----
    private boolean isPermissionRequested;
    private static final String[] ALL_PERMISSION = new String[]{
        Manifest.permission.CAMERA,
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gesturefx_main);
        if (savedInstanceState != null) {
            mLensType = savedInstanceState.getInt("lensType");
        }
        init();
    }

    private void init() {
        // Toolbar
        Toolbar toolbar = findViewById(R.id.gesturefx_toolbar);
        setSupportActionBar(toolbar);
        toolbar.setNavigationOnClickListener(view -> finish());

        // Camera preview
        mPreview = findViewById(R.id.gesturefx_preview);

        // GLSurfaceView setup (must be before setRenderer)
        mGlSurfaceView = findViewById(R.id.gesturefx_gl_surface);
        mGlSurfaceView.setEGLContextClientVersion(2);
        mGlSurfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0);
        mGlSurfaceView.getHolder().setFormat(PixelFormat.TRANSLUCENT);
        mGlSurfaceView.setZOrderMediaOverlay(true);

        // Gesture engine (OpenGL renderer)
        mGestureEngine = new GestureEngine(this);
        mGlSurfaceView.setRenderer(mGestureEngine);
        mGlSurfaceView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);

        // Mode label
        mModeLabel = findViewById(R.id.gesturefx_mode_label);

        // Create hand keypoint analyzer
        createHandKeypointAnalyzer();

        // Check camera permissions
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            createLensEngine();
        } else {
            checkPermission();
        }
    }

    private void createHandKeypointAnalyzer() {
        MLHandKeypointAnalyzerSetting setting =
                new MLHandKeypointAnalyzerSetting.Factory()
                        .setSceneType(MLHandKeypointAnalyzerSetting.TYPE_ALL)
                        .setMaxHandResults(2) // both hands
                        .create();
        mHmsAnalyzer = MLHandKeypointAnalyzerFactory.getInstance()
                .getHandKeypointAnalyzer(setting);
        mHmsAnalyzer.setTransactor(new HandKeypointTransactor());
    }

    // ===== Camera Permissions =====

    private void checkPermission() {
        if (Build.VERSION.SDK_INT >= 23 && !isPermissionRequested) {
            isPermissionRequested = true;
            ArrayList<String> permissionsList = new ArrayList<>();
            for (String perm : ALL_PERMISSION) {
                if (PackageManager.PERMISSION_GRANTED != this.checkSelfPermission(perm)) {
                    permissionsList.add(perm);
                }
            }
            if (!permissionsList.isEmpty()) {
                requestPermissions(permissionsList.toArray(new String[0]), 0);
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            createLensEngine();
            startLensEngine();
        } else if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_DENIED) {
            if (!ActivityCompat.shouldShowRequestPermissionRationale(this, permissions[0])) {
                showWarningDialog();
            } else {
                finish();
            }
        }
    }

    private void showWarningDialog() {
        AlertDialog.Builder dialog = new AlertDialog.Builder(this);
        dialog.setMessage(R.string.app_need_permission)
                .setPositiveButton(R.string.app_permission_by_hand, (d, which) -> {
                    Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
                    Uri uri = Uri.fromParts("package", getApplicationContext().getPackageName(), null);
                    intent.setData(uri);
                    startActivity(intent);
                })
                .setNegativeButton("Cancel", (d, which) -> finish())
                .setCancelable(false)
                .show();
    }

    // ===== LensEngine =====

    private void createLensEngine() {
        Context context = this.getApplicationContext();
        mLensEngine = new LensEngine.Creator(context, mHmsAnalyzer)
                .setLensType(this.mLensType)
                .applyDisplayDimension(PREVIEW_WIDTH, PREVIEW_HEIGHT)
                .applyFps(PREVIEW_FPS)
                .enableAutomaticFocus(true)
                .create();
    }

    private void startLensEngine() {
        if (this.mLensEngine != null) {
            try {
                this.mPreview.start(this.mLensEngine);
            } catch (IOException e) {
                Log.e(TAG, "Failed to start lens engine.", e);
                this.mLensEngine.release();
                this.mLensEngine = null;
            }
        }
    }

    private void switchCamera() {
        isFront = !isFront;
        mLensType = isFront ? LensEngine.FRONT_LENS : LensEngine.BACK_LENS;
        if (this.mLensEngine != null) {
            this.mLensEngine.close();
        }
        createLensEngine();
        startLensEngine();
    }

    @Override
    public void onClick(View v) {
        if (v.getId() == R.id.gesturefx_mode_label) {
            // Toggle camera on label click (convenient for testing)
            switchCamera();
        }
    }

    // ===== HMS Transactor =====

    private class HandKeypointTransactor implements MLAnalyzer.MLTransactor<MLHandKeypoints> {
        @Override
        public void transactResult(MLAnalyzer.Result<MLHandKeypoints> result) {
            SparseArray<MLHandKeypoints> handKeypointsSparseArray = result.getAnalyseList();
            List<MLHandKeypoints> hands = new ArrayList<>();
            for (int i = 0; i < handKeypointsSparseArray.size(); i++) {
                hands.add(handKeypointsSparseArray.valueAt(i));
            }

            // Pass to GL engine
            mGestureEngine.updateHandKeypoints(hands, PREVIEW_WIDTH, PREVIEW_HEIGHT, isFront);

            // Update UI on main thread
            final String modeLabel = mGestureEngine.getCurrentModeLabel();
            runOnUiThread(() -> mModeLabel.setText(modeLabel));
        }

        @Override
        public void destroy() {
            // Nothing to clean up
        }
    }

    // ===== Lifecycle =====

    @Override
    protected void onSaveInstanceState(@NonNull Bundle outState) {
        outState.putInt("lensType", this.mLensType);
        super.onSaveInstanceState(outState);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (mGlSurfaceView != null) {
            mGlSurfaceView.onResume();
        }
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            createLensEngine();
            startLensEngine();
        } else {
            checkPermission();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mGlSurfaceView != null) {
            mGlSurfaceView.onPause();
        }
        if (mPreview != null) {
            mPreview.stop();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (this.mLensEngine != null) {
            this.mLensEngine.release();
        }
        if (this.mHmsAnalyzer != null) {
            this.mHmsAnalyzer.stop();
        }
    }
}
