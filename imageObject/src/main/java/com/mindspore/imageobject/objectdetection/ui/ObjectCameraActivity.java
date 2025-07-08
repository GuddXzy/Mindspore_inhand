/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mindspore.imageobject.objectdetection.ui;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.mindspore.common.config.MSLinkUtils;
import com.mindspore.common.utils.Utils;
import com.mindspore.customview.dialog.NoticeDialog;
import com.mindspore.imageobject.R;
import com.mindspore.imageobject.camera.CameraPreview;
import com.mindspore.imageobject.objectdetection.bean.RecognitionObjectBean;
import com.mindspore.imageobject.objectdetection.help.ObjectTrackingMobile;

import java.io.FileNotFoundException;
import java.util.List;

import static com.mindspore.imageobject.objectdetection.bean.RecognitionObjectBean.getRecognitionList;


/**
 * main page of entrance
 * <p>
 * Pass in pictures to JNI, test mindspore model, load reasoning, etc
 */

@Route(path = "/imageobject/ObjectCameraActivity")
public class ObjectCameraActivity extends AppCompatActivity implements CameraPreview.RecognitionDataCallBack {

    private final String TAG = "ObjectCameraActivity";

    private CameraPreview cameraPreview;

    private ObjectTrackingMobile mTrackingMobile;

    private ObjectRectView mObjectRectView;

    private List<RecognitionObjectBean> recognitionObjectBeanList;

    private NoticeDialog noticeDialog;

    private Button btnChooseModel;

    private static final int REQUEST_CODE_CHOOSE_MODEL = 1010;

    private static Uri sCustomModelUri; // 存储用户选中的自定义模型 Uri（静态变量）

    private Uri customModelUri;         // 当前 Activity 用到的实例级 Uri


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        sCustomModelUri = null;
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_object_camera);

        cameraPreview = findViewById(R.id.camera_preview);
        mObjectRectView = findViewById(R.id.objRectView);

        // 如果静态变量里有保留的自定义模型，则记录到本地实例变量
        if (sCustomModelUri != null) {
            customModelUri = sCustomModelUri;
        }

        // 继续原有逻辑
        btnChooseModel = findViewById(R.id.btn_choose_model);
        btnChooseModel.setOnClickListener(v -> {
            openFileManagerToChooseModel();
        });

        init();
    }

    private void init() {
        try {
            mTrackingMobile = new ObjectTrackingMobile(this);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        // 如果已经有自定义 Uri，就直接加载它
        if (customModelUri != null) {
            boolean ret = mTrackingMobile.reloadModel(customModelUri);
            Log.d(TAG, "Reload custom model from Uri: " + customModelUri + " success? " + ret);
        } else {
            // 否则就走默认加载
            boolean ret = mTrackingMobile.loadModelFromBuf(getAssets());
            Log.d(TAG, "TrackingMobile loadModelFromBuf: " + ret);
        }

        cameraPreview.addImageRecognitionDataCallBack(this);

        Log.i(TAG, "init ObjectPhotoActivity info");
        Toolbar mToolbar = findViewById(R.id.object_camera_toolbar);
        setSupportActionBar(mToolbar);
        mToolbar.setNavigationOnClickListener(view -> finish());
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_setting_app, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int itemId = item.getItemId();
        if (itemId == R.id.item_help) {
            showHelpDialog();
        } else if (itemId == R.id.item_more) {
            Utils.openBrowser(this, MSLinkUtils.HELP_CAMERA_DETECTION);
        }
        return super.onOptionsItemSelected(item);
    }

    // **新增：打开系统文件管理器，选择自定义.ms模型**
    private void openFileManagerToChooseModel() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        startActivityForResult(Intent.createChooser(intent, "请选择一个 .ms 模型文件"), REQUEST_CODE_CHOOSE_MODEL);
    }

    // **重写onActivityResult，用于接收文件选择器结果**
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_CODE_CHOOSE_MODEL && resultCode == RESULT_OK) {
            if (data != null && data.getData() != null) {
                Uri uri = data.getData();

                // 先把 Uri 临时加载一下以确定是否能成功
                boolean success = mTrackingMobile.reloadModel(uri);
                if (success) {
                    // “重建 Activity”，在这里先存下来
                    sCustomModelUri = uri;
                    Log.i(TAG, "自定义模型加载成功: " + uri);

                    // **关键：调用 recreate() 重走 Activity 生命周期**
                    recreate();

                } else {
                    Log.e(TAG, "自定义模型加载失败: " + uri);
                }
            }
        }
    }


    private void showMsgDialog(String contentString) {
        noticeDialog = new NoticeDialog(this);
        noticeDialog.setTitleString("调试信息");
        noticeDialog.setContentString(contentString);
        noticeDialog.setYesOnclickListener(() -> {
            noticeDialog.dismiss();
        });
        noticeDialog.show();
    }

    private void showHelpDialog() {
        noticeDialog = new NoticeDialog(this);
        noticeDialog.setTitleString(getString(R.string.explain_title));
        noticeDialog.setContentString(getString(R.string.explain_camera_detection));
        noticeDialog.setYesOnclickListener(() -> {
            noticeDialog.dismiss();
        });
        noticeDialog.show();
    }

    @Override
    protected void onResume() {
        super.onResume();
        cameraPreview.onResume(this);
    }

    @Override
    protected void onPause() {
        super.onPause();
        cameraPreview.onPause();
    }

    public void onRecognitionDataCallBack(String result) {
        if (TextUtils.isEmpty(result)) {
            mObjectRectView.clearCanvas();
            return;
        }
        Log.d(TAG, result);
        recognitionObjectBeanList = getRecognitionList(result);
        mObjectRectView.setInfo(recognitionObjectBeanList);
    }

    @Override
    public void onRecognitionBitmapCallBack(Bitmap bitmap) {
        long startTime = System.currentTimeMillis();
        String result = mTrackingMobile.MindSpore_runnet(bitmap);
        long endTime = System.currentTimeMillis();
        Log.d(TAG,"TrackingMobile inferenceTime:"+(endTime - startTime) + "ms ");
        onRecognitionDataCallBack(result);
        if (!bitmap.isRecycled()) {
            bitmap.recycle();
        }
    }
}
