package org.arm.education.executorch.objectcutout;

import android.content.Context;
import android.graphics.Color;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.List;

final class ModelSpec {
    private static final String EXPECTED_LAYOUT = "nchw";

    final long[] inputShape;
    final long[] outputShape;
    final int inputWidth;
    final int inputHeight;
    final int outputWidth;
    final int outputHeight;
    final int classCount;
    final int backgroundClass;
    final int padColor;
    final float[] mean;
    final float[] std;

    private ModelSpec(
            long[] inputShape,
            long[] outputShape,
            int inputWidth,
            int inputHeight,
            int outputWidth,
            int outputHeight,
            int classCount,
            int backgroundClass,
            int padColor,
            float[] mean,
            float[] std
    ) {
        this.inputShape = inputShape;
        this.outputShape = outputShape;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.outputWidth = outputWidth;
        this.outputHeight = outputHeight;
        this.classCount = classCount;
        this.backgroundClass = backgroundClass;
        this.padColor = padColor;
        this.mean = mean;
        this.std = std;
    }

    static ModelSpec load(Context context, String assetName, List<String> labels) throws Exception {
        JSONObject root = AssetUtils.readJsonObject(context, assetName);
        JSONObject input = root.getJSONObject("input");
        JSONObject output = root.getJSONObject("output");
        requireValue("input layout", EXPECTED_LAYOUT, input.getString("layout"));
        requireValue("output layout", EXPECTED_LAYOUT, output.getString("layout"));
        requireValue("input dtype", "float32", input.getString("dtype"));
        requireValue("output dtype", "float32", output.getString("dtype"));

        long[] inputShape = readShape(input.getJSONArray("shape"));
        long[] outputShape = readShape(output.getJSONArray("shape"));
        if (inputShape[0] != 1 || inputShape[1] != 3) {
            throw new IllegalArgumentException("Expected input shape [1, 3, H, W]");
        }
        if (outputShape[0] != 1 || outputShape[1] != labels.size()) {
            throw new IllegalArgumentException("Output class count does not match voc_labels.json");
        }

        JSONObject preprocessing = input.getJSONObject("preprocessing");
        requireValue("resize mode", "letterbox", preprocessing.getString("resize_mode"));
        JSONArray resize = preprocessing.getJSONArray("resize");
        int inputWidth = Math.toIntExact(inputShape[3]);
        int inputHeight = Math.toIntExact(inputShape[2]);
        if (resize.getInt(0) != inputWidth || resize.getInt(1) != inputHeight) {
            throw new IllegalArgumentException("Preprocessing resize does not match the input shape");
        }

        JSONArray padRgb = preprocessing.getJSONArray("pad_rgb");
        int padColor = Color.rgb(padRgb.getInt(0), padRgb.getInt(1), padRgb.getInt(2));
        int backgroundClass = output.getJSONObject("postprocessing").getInt("background_class");
        int classCount = Math.toIntExact(outputShape[1]);
        if (backgroundClass < 0 || backgroundClass >= classCount) {
            throw new IllegalArgumentException("background_class is outside the output class range");
        }

        return new ModelSpec(
                inputShape,
                outputShape,
                inputWidth,
                inputHeight,
                Math.toIntExact(outputShape[3]),
                Math.toIntExact(outputShape[2]),
                classCount,
                backgroundClass,
                padColor,
                readFloatArray(preprocessing.getJSONArray("mean"), 3),
                readFloatArray(preprocessing.getJSONArray("std"), 3)
        );
    }

    private static long[] readShape(JSONArray values) throws Exception {
        if (values.length() != 4) {
            throw new IllegalArgumentException("Expected a four-dimensional tensor shape");
        }
        long[] shape = new long[4];
        for (int index = 0; index < shape.length; index++) {
            shape[index] = values.getLong(index);
        }
        return shape;
    }

    private static float[] readFloatArray(JSONArray values, int expectedLength) throws Exception {
        if (values.length() != expectedLength) {
            throw new IllegalArgumentException("Unexpected preprocessing array length");
        }
        float[] result = new float[expectedLength];
        for (int index = 0; index < expectedLength; index++) {
            result[index] = (float) values.getDouble(index);
        }
        return result;
    }

    private static void requireValue(String name, String expected, String actual) {
        if (!expected.equals(actual)) {
            throw new IllegalArgumentException("Expected " + name + " '" + expected + "', found '" + actual + "'");
        }
    }
}
