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
package com.mindspore.imageobject.objectdetection.help;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;
import android.net.Uri;

import com.mindspore.customview.dialog.NoticeDialog;

import java.io.FileNotFoundException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.HashMap;

public class ObjectTrackingMobile  {
    private final static String TAG = "ObjectTrackingMobile";

    static {
        try {
            System.loadLibrary("mlkit-label-MS");
            Log.i(TAG, "load libiMindSpore.so successfully.");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "UnsatisfiedLinkError >>>>>>" + e.getMessage());
        }
    }

    public static HashMap<Integer, String> synset_words_map = new HashMap<>();

    public static float[] threshold = new float[494];

    private long netEnv = 0;

    private final Context mActivity;

    private NoticeDialog noticeDialog;

    public ObjectTrackingMobile(Context activity) throws FileNotFoundException {
        this.mActivity = activity;
    }

    /**
     * jni Load model
     *
     * @param assetManager assetManager
     * @param buffer       buffer
     * @param numThread    numThread
     * @return Load model data
     */
    public native long loadModel(AssetManager assetManager, ByteBuffer buffer, int numThread);

    /**
     * jni Run model
     *
     * @param netEnv Load model data
     * @param img    Current picture
     * @return Run model data
     */
    public native String runNet(long netEnv, Bitmap img);

    /**
     * Unbind model data
     *
     * @param netEnv model data
     * @return Unbound state
     */
    public native boolean unloadModel(long netEnv);

    /**
     * C++ encapsulated as a method of the msnetworks class
     *
     * @param assetManager Model file location
     * @return Loading model file status
     */
    public boolean loadModelFromBuf(AssetManager assetManager) {
        String ModelPath = "model/ssd.ms"; // 这里加载模型
        ByteBuffer buffer = loadModelFile(ModelPath);
        Log.i(TAG, "origin buffer" + buffer.toString());
        netEnv = loadModel(assetManager, buffer, 2);
        Log.i(TAG, "now netEnv " + netEnv);
        return true;
    }

    /**
     * Run Mindspore
     *
     * @param img Current image recognition
     * @return Recognized text information
     */
    public String MindSpore_runnet(Bitmap img) {
        String ret_str = runNet(netEnv, img);
        return ret_str;
    }

    /**
     * Unbound model
     *
     * @return true
     */
    public boolean unloadModel() {
        unloadModel(netEnv);
        return true;
    }

    /**
     * Load model file stream
     *
     * @param modelPath Model file path
     * @return Load model file stream
     */
    public ByteBuffer loadModelFile(String modelPath) {
        InputStream is = null;
        try {
            is = mActivity.getAssets().open(modelPath);
            byte[] bytes = new byte[is.available()];
            is.read(bytes);
            return ByteBuffer.allocateDirect(bytes.length).put(bytes);
        } catch (Exception e) {
            Log.d("loadModelFile", " Exception occur ");
            e.printStackTrace();
        }
        return null;
    }

    // ====================== 新增代码：动态更换模型 ===========================

    /**
     * 通过外部 Uri 指定新的 .ms 模型进行重载
     * @param modelUri 新模型文件的 Uri
     * @return 是否成功
     */
    public boolean reloadModel(Uri modelUri) {
        try {
            // 1. 如果已有模型环境，先卸载
            if (netEnv != 0) {
                unloadModel(netEnv);
                netEnv = 0;
            }

            // 2. 从Uri读取模型文件
            ByteBuffer buffer = loadModelFileFromUri(modelUri);
            if (buffer == null) {
                Log.e(TAG, "reloadModel: buffer is null, failed to read model file.");
                return false;
            }
            Log.i(TAG, buffer.toString());

            // 3. 加载新模型（此处仍用2线程，可自行调整）
            netEnv = loadModel(mActivity.getAssets(), buffer, 2);
            if (netEnv == 0) {
                Log.e(TAG, "reloadModel: loadModel returned 0, failed to load new model.");
                return false;
            }

            Log.i(TAG, "reloadModel: Model reloaded successfully from " + modelUri);
            Log.i(TAG, "netEnv " + netEnv);
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    /**
     * 从Uri读取模型文件到ByteBuffer
     * @param uri 外部文件 Uri
     * @return ByteBuffer
     */
    private ByteBuffer loadModelFileFromUri(Uri uri) {
        InputStream is = null;
        try {
            is = mActivity.getContentResolver().openInputStream(uri);
            if (is == null) {
                Log.i(TAG, "InputStream is null.");
                return null;
            }

            byte[] bytes = new byte[is.available()];
            int readLen = is.read(bytes);
            if (readLen <= 0) {
                Log.i(TAG, "Failed to read data from InputStream.");
                return null;
            }

            ByteBuffer buffer = ByteBuffer.allocateDirect(bytes.length);
            buffer.put(bytes);
            buffer.position(0);

            Log.i(TAG, "成功读取模型buffer. Size: " + bytes.length + " bytes.");
            return buffer;
        } catch (Exception e) {
            Log.i(TAG, "Exception while loading model: " + e.getMessage());
            e.printStackTrace();
        } finally {
            if (is != null) {
                try {
                    is.close();
                } catch (Exception e) {
                    Log.i(TAG, "Exception while closing InputStream: " + e.getMessage());
                    e.printStackTrace();
                }
            }
        }
        Log.i(TAG, "Model loading failed.");
        return null;
    }
}
