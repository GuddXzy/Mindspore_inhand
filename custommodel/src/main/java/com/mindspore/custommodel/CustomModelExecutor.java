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

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.Map;

public class CustomModelExecutor {

    private static final String TAG = "CustomModelExecutor";
    private static final int INPUT_IMAGE_SIZE = 224; // 可以根据模型调整

    private Context mContext;
    private boolean isModelLoaded = false;

    private long fullExecutionTime;
    private long preProcessTime;
    private long inferenceTime;
    private long postProcessTime;

    private final int NUM_THREADS = 4;

    public CustomModelExecutor(Context context) {
        mContext = context;
        init();
    }

    public void init() {
        // 纯UI版本，不进行实际的模型初始化
        Log.i(TAG, "CustomModelExecutor initialized (UI only mode)");
    }

    /**
     * Load custom model from file path
     * @param modelPath path to the model file
     * @return true if loaded successfully
     */
    public boolean loadModel(String modelPath) {
        try {
            // 纯UI版本，模拟加载成功
            isModelLoaded = true;
            Log.i(TAG, "Model loaded successfully (UI only): " + modelPath);
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Error loading model: " + e.getMessage());
            return false;
        }
    }

    /**
     * Execute model inference
     * @param inputBitmap input image
     * @return inference result
     */
    public ModelExecutionResult execute(Bitmap inputBitmap) {
        if (!isModelLoaded) {
            Log.e(TAG, "Model not loaded");
            return null;
        }

        Log.i(TAG, "Running custom model inference (UI only mode)");

        fullExecutionTime = SystemClock.uptimeMillis();
        preProcessTime = SystemClock.uptimeMillis();

        try {
            // 模拟预处理
            ByteBuffer inputBuffer = preprocessImage(inputBitmap);
            
            preProcessTime = SystemClock.uptimeMillis() - preProcessTime;
            inferenceTime = SystemClock.uptimeMillis();

            // 模拟推理过程
            Thread.sleep(100); // 模拟推理时间

            inferenceTime = SystemClock.uptimeMillis() - inferenceTime;
            postProcessTime = SystemClock.uptimeMillis();

            // 模拟输出数据
            float[] outputData = new float[1000]; // 模拟1000个类别的输出
            for (int i = 0; i < outputData.length; i++) {
                outputData[i] = (float) Math.random(); // 随机输出
            }
            
            postProcessTime = SystemClock.uptimeMillis() - postProcessTime;
            fullExecutionTime = SystemClock.uptimeMillis() - fullExecutionTime;

            Log.d(TAG, "Inference completed in " + fullExecutionTime + "ms");
            Log.d(TAG, "Preprocess: " + preProcessTime + "ms, Inference: " + inferenceTime + "ms, Postprocess: " + postProcessTime + "ms");

            // Create result
            ModelExecutionResult result = new ModelExecutionResult();
            result.setOutputData(outputData);
            result.setExecutionTime(fullExecutionTime);
            
            return result;

        } catch (Exception e) {
            Log.e(TAG, "Error during inference: " + e.getMessage());
            return null;
        }
    }

    /**
     * Preprocess image for model input
     * @param bitmap input image
     * @return preprocessed data
     */
    private ByteBuffer preprocessImage(Bitmap bitmap) {
        // 模拟图像预处理
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, true);
        
        // Convert to ByteBuffer
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(1 * INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE * 3 * 4);
        inputBuffer.order(ByteOrder.nativeOrder());
        inputBuffer.rewind();
        
        int[] intValues = new int[INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE];
        resizedBitmap.getPixels(intValues, 0, INPUT_IMAGE_SIZE, 0, 0, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE);
        
        int pixel = 0;
        for (int y = 0; y < INPUT_IMAGE_SIZE; y++) {
            for (int x = 0; x < INPUT_IMAGE_SIZE; x++) {
                int value = intValues[pixel++];
                
                // Normalize to [0, 1] range
                float r = ((float) (value >> 16 & 255)) / 255.0f;
                float g = ((float) (value >> 8 & 255)) / 255.0f;
                float b = ((float) (value & 255)) / 255.0f;
                
                inputBuffer.putFloat(r);
                inputBuffer.putFloat(g);
                inputBuffer.putFloat(b);
            }
        }
        
        inputBuffer.rewind();
        return inputBuffer;
    }

    /**
     * Convert float array to byte array
     */
    public static byte[] floatArrayToByteArray(float[] floats) {
        ByteBuffer buffer = ByteBuffer.allocate(4 * floats.length);
        buffer.order(ByteOrder.nativeOrder());
        FloatBuffer floatBuffer = buffer.asFloatBuffer();
        floatBuffer.put(floats);
        return buffer.array();
    }

    /**
     * Release resources
     */
    public void release() {
        isModelLoaded = false;
        Log.i(TAG, "CustomModelExecutor released");
    }

    /**
     * Check if model is loaded
     */
    public boolean isModelLoaded() {
        return isModelLoaded;
    }

    /**
     * Model execution result
     */
    public static class ModelExecutionResult {
        private float[] outputData;
        private long executionTime;
        private Map<String, float[]> allOutputs; // 支持多个输出

        public float[] getOutputData() {
            return outputData;
        }

        public void setOutputData(float[] outputData) {
            this.outputData = outputData;
        }

        public long getExecutionTime() {
            return executionTime;
        }

        public void setExecutionTime(long executionTime) {
            this.executionTime = executionTime;
        }

        public Map<String, float[]> getAllOutputs() {
            return allOutputs;
        }

        public void setAllOutputs(Map<String, float[]> allOutputs) {
            this.allOutputs = allOutputs;
        }
    }

    /**
     * 获取所有输出张量的方法
     * @return 包含所有输出张量的Map，key为张量名称，value为张量数据
     */
    public Map<String, float[]> getAllOutputs() {
        if (!isModelLoaded) {
            Log.e(TAG, "Model not loaded");
            return null;
        }

        try {
            // 模拟多个输出
            Map<String, float[]> allOutputs = new java.util.HashMap<>();
            allOutputs.put("output1", new float[1000]);
            allOutputs.put("output2", new float[500]);
            
            Log.d(TAG, "Simulated multiple outputs generated");
            return allOutputs;
        } catch (Exception e) {
            Log.e(TAG, "Error getting all outputs: " + e.getMessage());
            return null;
        }
    }
} 