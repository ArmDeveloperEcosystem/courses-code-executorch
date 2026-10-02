package org.arm.education.executorch.objectcutout;

import android.content.Context;
import android.graphics.Bitmap;

import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

import java.io.File;
import java.util.Arrays;
import java.util.List;

final class ExecuTorchSegmenter {
    private static final String MODEL_ASSET = "lraspp_mobilenetv3.pte";
    private static final String LABELS_ASSET = "voc_labels.json";
    private static final String MODEL_SPEC_ASSET = "model_spec.json";

    private final Module module;
    private final List<String> labels;
    private final ModelSpec spec;

    ExecuTorchSegmenter(Context context) throws Exception {
        File modelFile = AssetUtils.copyAssetToFiles(context, MODEL_ASSET);
        labels = AssetUtils.readJsonStringArray(context, LABELS_ASSET);
        spec = ModelSpec.load(context, MODEL_SPEC_ASSET, labels);

        // ExecuTorch Android API: load the packaged .pte once for this segmenter.
        module = Module.load(modelFile.getAbsolutePath());
    }

    CutoutResult cutout(Bitmap source) {
        long preprocessStart = System.nanoTime();
        ImagePreprocessor.PreprocessedImage input = ImagePreprocessor.preprocess(source, spec);
        long preprocessMs = elapsedMs(preprocessStart);

        long forwardStart = System.nanoTime();
        // ExecuTorch Android API: wrap the app-owned NCHW float array as a tensor.
        Tensor inputTensor = Tensor.fromBlob(
                input.tensor,
                spec.inputShape
        );

        // ExecuTorch Android API: call the exported forward method and read its
        // single segmentation-logits tensor.
        EValue output = module.forward(EValue.from(inputTensor))[0];
        Tensor outputTensor = output.toTensor();
        if (!Arrays.equals(outputTensor.shape(), spec.outputShape)) {
            throw new IllegalArgumentException(
                    "Unexpected model output shape: " + Arrays.toString(outputTensor.shape())
                            + ", expected " + Arrays.toString(spec.outputShape)
            );
        }
        float[] logits = outputTensor.getDataAsFloatArray();
        long forwardMs = elapsedMs(forwardStart);

        long postStart = System.nanoTime();
        MaskPostprocessor.MaskResult mask = MaskPostprocessor.makeCutout(logits, input, labels, spec);
        long postprocessMs = elapsedMs(postStart);

        return new CutoutResult(
                mask.cutout,
                mask.content,
                mask.alphaMask,
                mask.label,
                preprocessMs,
                forwardMs,
                postprocessMs,
                mask.foregroundPixels,
                mask.foregroundFraction
        );
    }

    private static long elapsedMs(long startedNanos) {
        return Math.round((System.nanoTime() - startedNanos) / 1_000_000.0);
    }

    void close() {
        // Release the native ExecuTorch module with the Android component.
        module.close();
    }
}
