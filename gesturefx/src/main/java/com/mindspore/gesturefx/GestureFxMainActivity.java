package com.mindspore.gesturefx;

import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import com.alibaba.android.arouter.facade.annotation.Route;

/**
 * Gesture Interaction Effects Main Activity
 * Real-time hand gesture recognition with 3D particle effects overlay on camera preview.
 */
@Route(path = "/gesturefx/GestureFxMainActivity")
public class GestureFxMainActivity extends AppCompatActivity {

    private static final String TAG = "GestureFxMainActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Placeholder: layout will be set in Step 2
    }
}
