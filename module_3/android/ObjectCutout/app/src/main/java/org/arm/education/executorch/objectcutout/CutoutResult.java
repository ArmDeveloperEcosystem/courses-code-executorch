package org.arm.education.executorch.objectcutout;

import android.graphics.Bitmap;

final class CutoutResult {
    final Bitmap cutout;
    final Bitmap content;
    final byte[] alphaMask;
    final String label;
    final long preprocessMs;
    final long forwardMs;
    final long postprocessMs;
    final int foregroundPixels;
    final float foregroundFraction;

    CutoutResult(
            Bitmap cutout,
            Bitmap content,
            byte[] alphaMask,
            String label,
            long preprocessMs,
            long forwardMs,
            long postprocessMs,
            int foregroundPixels,
            float foregroundFraction
    ) {
        this.cutout = cutout;
        this.content = content;
        this.alphaMask = alphaMask;
        this.label = label;
        this.preprocessMs = preprocessMs;
        this.forwardMs = forwardMs;
        this.postprocessMs = postprocessMs;
        this.foregroundPixels = foregroundPixels;
        this.foregroundFraction = foregroundFraction;
    }

    long totalMs() {
        return preprocessMs + forwardMs + postprocessMs;
    }
}
