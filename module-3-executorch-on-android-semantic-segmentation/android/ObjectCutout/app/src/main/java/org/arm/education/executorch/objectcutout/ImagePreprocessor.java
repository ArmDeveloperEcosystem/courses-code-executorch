package org.arm.education.executorch.objectcutout;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

final class ImagePreprocessor {
    private ImagePreprocessor() {}

    static PreprocessedImage preprocess(Bitmap source, ModelSpec spec) {
        int sourceWidth = source.getWidth();
        int sourceHeight = source.getHeight();
        float scale = Math.min(
                spec.inputWidth / (float) sourceWidth,
                spec.inputHeight / (float) sourceHeight
        );
        int resizedWidth = Math.max(1, Math.min(spec.inputWidth, Math.round(sourceWidth * scale)));
        int resizedHeight = Math.max(1, Math.min(spec.inputHeight, Math.round(sourceHeight * scale)));
        int padLeft = (spec.inputWidth - resizedWidth) / 2;
        int padTop = (spec.inputHeight - resizedHeight) / 2;

        Bitmap scaled = Bitmap.createScaledBitmap(source, resizedWidth, resizedHeight, true);
        Bitmap letterboxed = Bitmap.createBitmap(spec.inputWidth, spec.inputHeight, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(letterboxed);
        canvas.drawColor(spec.padColor);
        canvas.drawBitmap(scaled, padLeft, padTop, new Paint(Paint.FILTER_BITMAP_FLAG));

        int pixelsPerChannel = spec.inputWidth * spec.inputHeight;
        int[] pixels = new int[pixelsPerChannel];
        letterboxed.getPixels(pixels, 0, spec.inputWidth, 0, 0, spec.inputWidth, spec.inputHeight);
        float[] tensor = new float[3 * pixelsPerChannel];
        for (int index = 0; index < pixels.length; index++) {
            int pixel = pixels[index];
            float red = Color.red(pixel) / 255.0f;
            float green = Color.green(pixel) / 255.0f;
            float blue = Color.blue(pixel) / 255.0f;
            tensor[index] = (red - spec.mean[0]) / spec.std[0];
            tensor[pixelsPerChannel + index] = (green - spec.mean[1]) / spec.std[1];
            tensor[2 * pixelsPerChannel + index] = (blue - spec.mean[2]) / spec.std[2];
        }

        return new PreprocessedImage(
                letterboxed,
                tensor,
                spec.inputWidth,
                spec.inputHeight,
                padLeft,
                padTop,
                resizedWidth,
                resizedHeight
        );
    }

    static final class PreprocessedImage {
        final Bitmap letterboxed;
        final float[] tensor;
        final int tensorWidth;
        final int tensorHeight;
        final int padLeft;
        final int padTop;
        final int contentWidth;
        final int contentHeight;

        PreprocessedImage(
                Bitmap letterboxed,
                float[] tensor,
                int tensorWidth,
                int tensorHeight,
                int padLeft,
                int padTop,
                int contentWidth,
                int contentHeight
        ) {
            this.letterboxed = letterboxed;
            this.tensor = tensor;
            this.tensorWidth = tensorWidth;
            this.tensorHeight = tensorHeight;
            this.padLeft = padLeft;
            this.padTop = padTop;
            this.contentWidth = contentWidth;
            this.contentHeight = contentHeight;
        }
    }
}
