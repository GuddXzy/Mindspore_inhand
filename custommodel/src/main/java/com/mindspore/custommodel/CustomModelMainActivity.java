/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
package com.mindspore.custommodel;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.content.FileProvider;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.alibaba.android.arouter.launcher.ARouter;
import com.mindspore.common.utils.Utils;
import com.mindspore.customview.dialog.NoticeDialog;
import com.mindspore.custommodel.R;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * Custom Model Main Activity
 * This is a blank page for custom model functionality
 */
@Route(path = "/custommodel/CustomModelMainActivity")
public class CustomModelMainActivity extends AppCompatActivity {
    private static final String TAG = "CustomModelMainActivity";
    private static final int RC_CHOOSE_PHOTO = 1;
    private static final int RC_CHOOSE_CAMERA = 2;
    private static final int RC_PICK_MODEL = 3;
    
    private NoticeDialog noticeDialog;
    private FrameLayout imagePreviewContainer;
    private ImageView imgPreview;
    private TextView tvImagePlaceholder;
    private ProgressBar progressBar;
    private Button btnExecute;
    private TextView tvOutput;
    
    private Uri imageUri;
    private Bitmap selectedBitmap;
    private String selectedModelPath;
    private boolean isRunningModel = false;
    private Integer maxWidthOfImage;
    private Integer maxHeightOfImage;
    private CustomModelExecutor modelExecutor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //inject
        ARouter.getInstance().inject(this);
        setContentView(R.layout.activity_custom_model_main);
        init();
    }

    private void init() {
        Toolbar mToolbar = findViewById(R.id.custom_model_toolbar);
        mToolbar.setTitle(getString(R.string.custom_model_title));
        setSupportActionBar(mToolbar);
        mToolbar.setNavigationOnClickListener(view -> finish());
        
        // 初始化UI组件
        imagePreviewContainer = findViewById(R.id.image_preview_container);
        imgPreview = findViewById(R.id.img_preview);
        tvImagePlaceholder = findViewById(R.id.tv_image_placeholder);
        progressBar = findViewById(R.id.progress);
        btnExecute = findViewById(R.id.btn_execute);
        tvOutput = findViewById(R.id.tv_output);
        
        // 初始化模型执行器
        modelExecutor = new CustomModelExecutor(this);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "onCreateOptionsMenu info");
        getMenuInflater().inflate(R.menu.menu_setting_app, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int itemId = item.getItemId();
        if (itemId == R.id.item_help) {
            showHelpDialog();
        } else if (itemId == R.id.item_more) {
            Utils.openBrowser(this, "www.baidu.com");
        }
        return super.onOptionsItemSelected(item);
    }

    private void showHelpDialog() {
        noticeDialog = new NoticeDialog(this);
        noticeDialog.setTitleString(getString(R.string.explain_title));
        noticeDialog.setContentString(getString(R.string.explain_custom_model));
        noticeDialog.setYesOnclickListener(() -> {
            noticeDialog.dismiss();
        });
        noticeDialog.show();
    }
    
    // 选择模型按钮点击事件
    public void onClickSelectModel(View view) {
        openModelPicker();
    }
    
    // 选择图片按钮点击事件
    public void onClickSelectPhoto(View view) {
        openGallery();
    }
    
    // 拍照按钮点击事件
    public void onClickTakePhoto(View view) {
        openCamera();
    }
    
    // 执行按钮点击事件
    public void onClickExecute(View view) {
        executeModel();
    }
    
    // 打开模型选择器
    private void openModelPicker() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        startActivityForResult(intent, RC_PICK_MODEL);
    }
    
    // 打开相册
    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, null);
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intent, RC_CHOOSE_PHOTO);
    }
    
    // 打开相机
    private void openCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        File photoDir = getExternalFilesDir(android.os.Environment.DIRECTORY_PICTURES);
        if (photoDir != null && !photoDir.exists()) {
            photoDir.mkdirs();
        }
        String mTempPhotoPath = new File(photoDir, "photo.jpeg").getAbsolutePath();
        imageUri = FileProvider.getUriForFile(this, getApplicationContext().getPackageName() + ".fileprovider", new File(mTempPhotoPath));
        intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
        startActivityForResult(intent, RC_CHOOSE_CAMERA);
    }
    
    // 执行模型
    private void executeModel() {
        if (selectedBitmap == null) {
            Toast.makeText(this, "Please select an image first", Toast.LENGTH_SHORT).show();
            return;
        }
        
        if (!modelExecutor.isModelLoaded()) {
            Toast.makeText(this, "Please select a model first", Toast.LENGTH_SHORT).show();
            return;
        }
        
        if (isRunningModel) {
            Toast.makeText(this, "Model is already running", Toast.LENGTH_SHORT).show();
            return;
        }
        
        isRunningModel = true;
        progressBar.setVisibility(View.VISIBLE);
        tvOutput.setText("Model execution in progress...\n\nRunning inference on selected image...");
        
        // 在后台线程执行模型推理
        new Thread(() -> {
            try {
                CustomModelExecutor.ModelExecutionResult result = modelExecutor.execute(selectedBitmap);
                
                runOnUiThread(() -> {
                    isRunningModel = false;
                    progressBar.setVisibility(View.INVISIBLE);
                    
                    if (result != null) {
                        // 格式化输出结果
                        StringBuilder output = new StringBuilder();
                        output.append("Model execution completed!\n\n");
                        output.append("Execution time: ").append(result.getExecutionTime()).append("ms\n\n");
                        output.append("Output data (first 10 values):\n");
                        
                        float[] outputData = result.getOutputData();
                        if (outputData != null) {
                            int displayCount = Math.min(10, outputData.length);
                            for (int i = 0; i < displayCount; i++) {
                                output.append(String.format("Value[%d]: %.6f\n", i, outputData[i]));
                            }
                            if (outputData.length > 10) {
                                output.append("... (showing first 10 values)\n");
                            }
                        }
                        
                        tvOutput.setText(output.toString());
                    } else {
                        tvOutput.setText("Model execution failed!\n\nError: Failed to get inference result");
                    }
                });
            } catch (Exception e) {
                runOnUiThread(() -> {
                    isRunningModel = false;
                    progressBar.setVisibility(View.INVISIBLE);
                    tvOutput.setText("Model execution failed!\n\nError: " + e.getMessage());
                });
            }
        }).start();
    }
    
    // 显示选择的图片
    private void showSelectedImage(Bitmap bitmap) {
        selectedBitmap = bitmap;
        imgPreview.setImageBitmap(bitmap);
        tvImagePlaceholder.setVisibility(View.GONE);
        imagePreviewContainer.setVisibility(View.VISIBLE);
        // 执行按钮现在默认可见，不需要再设置
    }
    
    // 处理从相册选择的图片
    private void showOriginImage() {
        try {
            if (imageUri == null) {
                Toast.makeText(this, "Image URI is null", Toast.LENGTH_SHORT).show();
                return;
            }
            InputStream inputStream = getContentResolver().openInputStream(imageUri);
            Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
            if (bitmap != null) {
                showSelectedImage(bitmap);
            } else {
                Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show();
            }
        } catch (Exception e) {
            Log.e(TAG, "Error loading image: " + e.getMessage());
            Toast.makeText(this, "Error loading image: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }
    
    // 处理拍照的图片
    private void showOriginCamera() {
        try {
            if (imageUri == null) {
                Toast.makeText(this, "Image URI is null", Toast.LENGTH_SHORT).show();
                return;
            }
            InputStream inputStream = getContentResolver().openInputStream(imageUri);
            Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
            if (bitmap != null) {
                showSelectedImage(bitmap);
            } else {
                Toast.makeText(this, "Failed to load camera image", Toast.LENGTH_SHORT).show();
            }
        } catch (Exception e) {
            Log.e(TAG, "Error loading camera image: " + e.getMessage());
            Toast.makeText(this, "Error loading camera image: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }
    
    // 复制URI到文件
    @Nullable
    private File copyUriToFile(Uri uri, String fileName) {
        try {
            InputStream inputStream = getContentResolver().openInputStream(uri);
            if (inputStream == null) {
                return null;
            }
            
            File dst = new File(getCacheDir(), fileName);
            FileOutputStream outputStream = new FileOutputStream(dst);
            
            byte[] buffer = new byte[1024];
            int length;
            while ((length = inputStream.read(buffer)) > 0) {
                outputStream.write(buffer, 0, length);
            }
            
            inputStream.close();
            outputStream.close();
            
            return dst;
        } catch (IOException e) {
            Log.e(TAG, "Error copying file: " + e.getMessage());
            return null;
        }
    }
    
    // 获取目标尺寸
    private Pair<Integer, Integer> getTargetSize() {
        int maxWidth = getMaxWidthOfImage();
        int maxHeight = getMaxHeightOfImage();
        return new Pair<>(maxWidth, maxHeight);
    }
    
    // 获取最大宽度
    private Integer getMaxWidthOfImage() {
        if (this.maxWidthOfImage == null) {
            this.maxWidthOfImage = ((View) this.imgPreview.getParent()).getWidth();
        }
        return this.maxWidthOfImage;
    }
    
    // 获取最大高度
    private Integer getMaxHeightOfImage() {
        if (this.maxHeightOfImage == null) {
            this.maxHeightOfImage = ((View) this.imgPreview.getParent()).getHeight();
        }
        return this.maxHeightOfImage;
    }
    
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (RC_CHOOSE_PHOTO == requestCode) {
                if (data != null && data.getData() != null) {
                    this.imageUri = data.getData();
                    showOriginImage();
                }
            } else if (RC_CHOOSE_CAMERA == requestCode) {
                showOriginCamera();
            } else if (RC_PICK_MODEL == requestCode) {
                if (data != null && data.getData() != null) {
                    File dst = copyUriToFile(data.getData(), "custom_model.ms");
                    if (dst != null) {
                        selectedModelPath = dst.getAbsolutePath();
                        boolean success = modelExecutor.loadModel(selectedModelPath);
                        if (success) {
                            Toast.makeText(this, "Model loaded successfully: " + selectedModelPath, Toast.LENGTH_SHORT).show();
                        } else {
                            Toast.makeText(this, "Failed to load model", Toast.LENGTH_SHORT).show();
                            selectedModelPath = null;
                        }
                    } else {
                        Toast.makeText(this, "Failed to copy model file", Toast.LENGTH_SHORT).show();
                    }
                }
            }
        }
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (modelExecutor != null) {
            modelExecutor.release();
        }
        if (selectedBitmap != null && !selectedBitmap.isRecycled()) {
            selectedBitmap.recycle();
        }
    }
    
    // 简单的Pair类
    private static class Pair<F, S> {
        public final F first;
        public final S second;
        
        public Pair(F first, S second) {
            this.first = first;
            this.second = second;
        }
    }
} 
