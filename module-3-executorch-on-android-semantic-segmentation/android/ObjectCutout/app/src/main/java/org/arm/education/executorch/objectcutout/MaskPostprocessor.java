package org.arm.education.executorch.objectcutout;

import android.graphics.Bitmap;
import android.graphics.Color;

import java.util.List;

final class MaskPostprocessor {
    private MaskPostprocessor() {}

    static MaskResult makeCutout(
            float[] logits,
            ImagePreprocessor.PreprocessedImage preprocessed,
            List<String> labels,
            ModelSpec spec
    ) {
        int outputPixels = spec.outputWidth * spec.outputHeight;
        int expectedValues = outputPixels * spec.classCount;
        if (logits.length != expectedValues) {
            throw new IllegalArgumentException(
                    "Unexpected model output length: " + logits.length + ", expected " + expectedValues
            );
        }
        if (spec.outputWidth != preprocessed.tensorWidth || spec.outputHeight != preprocessed.tensorHeight) {
            throw new IllegalArgumentException("Output dimensions must match the preprocessed image dimensions");
        }

        Candidate candidate = buildCandidate(logits, outputPixels, spec.classCount, spec.backgroundClass);
        Bitmap content = Bitmap.createBitmap(
                preprocessed.letterboxed,
                preprocessed.padLeft,
                preprocessed.padTop,
                preprocessed.contentWidth,
                preprocessed.contentHeight
        );
        byte[] alphaMask = buildAlphaMask(
                candidate.mask,
                candidate.selectedClass,
                preprocessed.tensorWidth,
                preprocessed.padLeft,
                preprocessed.padTop,
                preprocessed.contentWidth,
                preprocessed.contentHeight
        );
        int selectedPixels = countSelected(alphaMask);
        int contentPixels = preprocessed.contentWidth * preprocessed.contentHeight;
        if (candidate.selectedClass == spec.backgroundClass || selectedPixels == 0) {
            return new MaskResult(content, content, alphaMask, "no supported object", 0, 0.0f);
        }

        return new MaskResult(
                makeCutout(content, alphaMask),
                content,
                alphaMask,
                labels.get(candidate.selectedClass),
                selectedPixels,
                selectedPixels / (float) contentPixels
        );
    }

    static Bitmap makeHighlight(Bitmap image, byte[] alphaMask, int highlightColor) {
        Bitmap highlight = image.copy(Bitmap.Config.ARGB_8888, true);
        int width = highlight.getWidth();
        int height = highlight.getHeight();
        int[] pixels = new int[width * height];
        highlight.getPixels(pixels, 0, width, 0, 0, width, height);

        float blend = Color.alpha(highlightColor) / 255.0f;
        int tintRed = Color.red(highlightColor);
        int tintGreen = Color.green(highlightColor);
        int tintBlue = Color.blue(highlightColor);
        for (int index = 0; index < pixels.length; index++) {
            if (alphaMask[index] == 0) {
                continue;
            }
            int pixel = pixels[index];
            int red = Math.round(Color.red(pixel) * (1.0f - blend) + tintRed * blend);
            int green = Math.round(Color.green(pixel) * (1.0f - blend) + tintGreen * blend);
            int blue = Math.round(Color.blue(pixel) * (1.0f - blend) + tintBlue * blend);
            pixels[index] = Color.argb(Color.alpha(pixel), red, green, blue);
        }
        highlight.setPixels(pixels, 0, width, 0, 0, width, height);
        return highlight;
    }

    private static Candidate buildCandidate(
            float[] logits,
            int pixels,
            int classes,
            int backgroundClass
    ) {
        int[] mask = new int[pixels];
        int[] counts = new int[classes];
        for (int pixel = 0; pixel < pixels; pixel++) {
            int bestClass = 0;
            float bestScore = logits[pixel];
            for (int classIndex = 1; classIndex < classes; classIndex++) {
                float score = logits[classIndex * pixels + pixel];
                if (score > bestScore) {
                    bestScore = score;
                    bestClass = classIndex;
                }
            }
            mask[pixel] = bestClass;
            counts[bestClass]++;
        }
        return new Candidate(mask, largestForegroundClass(counts, backgroundClass));
    }

    private static int largestForegroundClass(int[] counts, int backgroundClass) {
        int selected = backgroundClass;
        int bestCount = 0;
        for (int classIndex = 0; classIndex < counts.length; classIndex++) {
            if (classIndex != backgroundClass && counts[classIndex] > bestCount) {
                bestCount = counts[classIndex];
                selected = classIndex;
            }
        }
        return selected;
    }

    private static byte[] buildAlphaMask(
            int[] squareMask,
            int selectedClass,
            int tensorWidth,
            int padLeft,
            int padTop,
            int contentWidth,
            int contentHeight
    ) {
        byte[] alphaMask = new byte[contentWidth * contentHeight];
        for (int y = 0; y < contentHeight; y++) {
            for (int x = 0; x < contentWidth; x++) {
                int tensorIndex = (y + padTop) * tensorWidth + x + padLeft;
                int contentIndex = y * contentWidth + x;
                alphaMask[contentIndex] = squareMask[tensorIndex] == selectedClass ? (byte) 255 : 0;
            }
        }
        return alphaMask;
    }

    private static int countSelected(byte[] alphaMask) {
        int count = 0;
        for (byte value : alphaMask) {
            if (value != 0) {
                count++;
            }
        }
        return count;
    }

    private static Bitmap makeCutout(Bitmap image, byte[] alphaMask) {
        int width = image.getWidth();
        int height = image.getHeight();
        int[] pixels = new int[width * height];
        image.getPixels(pixels, 0, width, 0, 0, width, height);
        for (int index = 0; index < pixels.length; index++) {
            if (alphaMask[index] == 0) {
                pixels[index] = 0x00000000;
            }
        }
        Bitmap cutout = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        cutout.setPixels(pixels, 0, width, 0, 0, width, height);
        return cutout;
    }

    static final class MaskResult {
        final Bitmap cutout;
        final Bitmap content;
        final byte[] alphaMask;
        final String label;
        final int foregroundPixels;
        final float foregroundFraction;

        MaskResult(
                Bitmap cutout,
                Bitmap content,
                byte[] alphaMask,
                String label,
                int foregroundPixels,
                float foregroundFraction
        ) {
            this.cutout = cutout;
            this.content = content;
            this.alphaMask = alphaMask;
            this.label = label;
            this.foregroundPixels = foregroundPixels;
            this.foregroundFraction = foregroundFraction;
        }
    }

    private static final class Candidate {
        final int[] mask;
        final int selectedClass;

        Candidate(int[] mask, int selectedClass) {
            this.mask = mask;
            this.selectedClass = selectedClass;
        }
    }
}
