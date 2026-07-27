package org.arm.education.executorch.objectcutout;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.view.DisplayCutout;
import android.view.Gravity;
import android.view.View;
import android.view.WindowInsets;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final int PICK_IMAGE = 10;
    private static final int[] HIGHLIGHT_COLORS = {
            Color.argb(105, 238, 76, 124),
            Color.argb(110, 92, 200, 255),
            Color.argb(110, 255, 193, 94),
            Color.argb(110, 75, 207, 138),
            Color.argb(115, 155, 93, 229)
    };

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final List<View> colorSwatches = new ArrayList<>();
    private volatile boolean destroyed;
    private ExecuTorchSegmenter segmenter;
    private Bitmap selectedBitmap;
    private CutoutResult lastResult;
    private ImageView originalView;
    private ImageView cutoutView;
    private TextView status;
    private TextView timing;
    private Button runButton;
    private Button cutoutModeButton;
    private Button highlightModeButton;
    private DisplayMode displayMode = DisplayMode.CUTOUT;
    private int highlightColor = HIGHLIGHT_COLORS[0];

    private enum DisplayMode {
        CUTOUT,
        HIGHLIGHT
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(buildUi());
        loadModel();
    }

    private View buildUi() {
        int horizontalPad = dp(20);
        int topPad = dp(18);
        int bottomPad = dp(18);
        int pageBackground = Color.rgb(255, 247, 242);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(horizontalPad, topPad, horizontalPad, bottomPad);
        root.setBackgroundColor(pageBackground);
        root.setClipToPadding(false);
        root.setOnApplyWindowInsetsListener((view, insets) -> {
            int safeTop = insets.getSystemWindowInsetTop();
            DisplayCutout displayCutout = insets.getDisplayCutout();
            if (displayCutout != null) {
                safeTop = Math.max(safeTop, displayCutout.getSafeInsetTop());
            }
            view.setPadding(
                    horizontalPad + insets.getSystemWindowInsetLeft(),
                    topPad + safeTop,
                    horizontalPad + insets.getSystemWindowInsetRight(),
                    bottomPad + insets.getSystemWindowInsetBottom()
            );
            return insets;
        });

        ScrollView scrollView = new ScrollView(this);
        scrollView.setClipToPadding(false);
        scrollView.setFillViewport(true);
        scrollView.setVerticalScrollBarEnabled(true);
        scrollView.setScrollbarFadingEnabled(false);
        scrollView.setOverScrollMode(View.OVER_SCROLL_ALWAYS);

        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(0, 0, 0, dp(24));
        content.setMinimumHeight(getResources().getDisplayMetrics().heightPixels + dp(240));

        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.VERTICAL);
        header.setPadding(dp(18), dp(16), dp(18), dp(16));
        header.setBackground(gradient(
                dp(22),
                Color.rgb(255, 111, 97),
                Color.rgb(255, 193, 94),
                Color.rgb(92, 200, 255)
        ));
        header.setElevation(dp(3));

        TextView chip = new TextView(this);
        chip.setText("On-device AI");
        chip.setTextSize(12);
        chip.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        chip.setTextColor(Color.rgb(101, 44, 28));
        chip.setPadding(dp(10), dp(5), dp(10), dp(5));
        chip.setBackground(rounded(Color.argb(210, 255, 255, 255), dp(14)));
        header.addView(chip);

        TextView title = new TextView(this);
        title.setText("Cutout Studio");
        title.setTextSize(30);
        title.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        title.setTextColor(Color.rgb(16, 24, 40));
        title.setGravity(Gravity.START);
        title.setIncludeFontPadding(false);
        title.setPadding(0, dp(12), 0, 0);
        header.addView(title);

        TextView subtitle = new TextView(this);
        subtitle.setText("Remove backgrounds for clean social posts");
        subtitle.setTextColor(Color.rgb(52, 64, 84));
        subtitle.setTextSize(14);
        subtitle.setLineSpacing(0, 1.05f);
        subtitle.setPadding(0, dp(6), 0, 0);
        header.addView(subtitle);

        LinearLayout.LayoutParams headerParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        headerParams.setMargins(0, 0, 0, dp(18));
        content.addView(header, headerParams);

        LinearLayout buttons = new LinearLayout(this);
        buttons.setOrientation(LinearLayout.HORIZONTAL);
        buttons.setGravity(Gravity.CENTER_VERTICAL);

        Button pickButton = new Button(this);
        pickButton.setText("Choose image");
        pickButton.setAllCaps(false);
        pickButton.setTextColor(Color.rgb(132, 45, 24));
        pickButton.setBackground(roundedStroke(
                Color.WHITE,
                dp(24),
                Color.rgb(255, 138, 101),
                dp(2)
        ));
        pickButton.setOnClickListener(v -> pickPhoto());
        buttons.addView(pickButton, new LinearLayout.LayoutParams(0, dp(48), 1));

        runButton = new Button(this);
        runButton.setText("Remove background");
        runButton.setAllCaps(false);
        setRunButtonEnabled(false);
        runButton.setOnClickListener(v -> runCutout());
        LinearLayout.LayoutParams runParams = new LinearLayout.LayoutParams(0, dp(48), 1);
        runParams.setMargins(dp(10), 0, 0, 0);
        buttons.addView(runButton, runParams);
        content.addView(buttons);

        LinearLayout outputControls = outputControls();
        LinearLayout.LayoutParams outputParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        outputParams.setMargins(0, dp(14), 0, 0);
        content.addView(outputControls, outputParams);

        status = new TextView(this);
        status.setText("Loading model...");
        status.setTextColor(Color.rgb(155, 56, 34));
        status.setTextSize(14);
        status.setPadding(0, dp(18), 0, dp(8));
        content.addView(status);

        timing = new TextView(this);
        timing.setText("Preprocess: -   Forward: -   Postprocess: -   Total: -");
        timing.setTextColor(Color.rgb(91, 103, 121));
        timing.setTextSize(13);
        timing.setPadding(0, 0, 0, dp(12));
        content.addView(timing);

        LinearLayout images = new LinearLayout(this);
        images.setOrientation(LinearLayout.VERTICAL);

        originalView = imageView("Original");
        images.addView(imageCard(
                "Source image",
                "Pick a photo with one clear supported subject.",
                originalView
        ));

        cutoutView = imageView("Cutout");
        cutoutView.setBackground(new CheckerboardDrawable(dp(18)));
        View postPreviewCard = imageCard(
                "Post preview",
                "Checkerboard shows the transparent background.",
                cutoutView
        );
        LinearLayout.LayoutParams cutoutParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        cutoutParams.setMargins(0, dp(14), 0, 0);
        images.addView(postPreviewCard, cutoutParams);

        content.addView(images, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        ));
        scrollView.addView(content, new ScrollView.LayoutParams(
                ScrollView.LayoutParams.MATCH_PARENT,
                ScrollView.LayoutParams.WRAP_CONTENT
        ));
        root.addView(scrollView, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0,
                1
        ));

        return root;
    }

    private LinearLayout outputControls() {
        LinearLayout controls = new LinearLayout(this);
        controls.setOrientation(LinearLayout.VERTICAL);

        LinearLayout modeRow = new LinearLayout(this);
        modeRow.setOrientation(LinearLayout.HORIZONTAL);

        cutoutModeButton = modeButton("Cutout", DisplayMode.CUTOUT);
        highlightModeButton = modeButton("Highlight", DisplayMode.HIGHLIGHT);
        modeRow.addView(cutoutModeButton, new LinearLayout.LayoutParams(0, dp(42), 1));
        LinearLayout.LayoutParams highlightParams = new LinearLayout.LayoutParams(0, dp(42), 1);
        highlightParams.setMargins(dp(10), 0, 0, 0);
        modeRow.addView(highlightModeButton, highlightParams);
        controls.addView(modeRow);

        LinearLayout colorRow = new LinearLayout(this);
        colorRow.setOrientation(LinearLayout.HORIZONTAL);
        colorRow.setGravity(Gravity.CENTER_VERTICAL);
        LinearLayout.LayoutParams colorRowParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        colorRowParams.setMargins(0, dp(10), 0, 0);
        controls.addView(colorRow, colorRowParams);

        for (int color : HIGHLIGHT_COLORS) {
            View swatch = new View(this);
            swatch.setContentDescription("Highlight color");
            swatch.setOnClickListener(v -> {
                highlightColor = color;
                setDisplayMode(DisplayMode.HIGHLIGHT);
            });
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(dp(34), dp(34));
            params.setMargins(0, 0, dp(10), 0);
            colorRow.addView(swatch, params);
            colorSwatches.add(swatch);
        }
        updateOutputControls();
        return controls;
    }

    private Button modeButton(String text, DisplayMode mode) {
        Button button = new Button(this);
        button.setText(text);
        button.setAllCaps(false);
        button.setOnClickListener(v -> setDisplayMode(mode));
        return button;
    }

    private LinearLayout imageCard(String title, String caption, ImageView imageView) {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setPadding(dp(14), dp(12), dp(14), dp(14));
        card.setBackground(roundedStroke(
                Color.WHITE,
                dp(18),
                Color.rgb(255, 221, 205),
                dp(1)
        ));
        card.setElevation(dp(2));

        card.addView(label(title));
        card.addView(detail(caption));
        LinearLayout.LayoutParams imageParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(270)
        );
        imageParams.setMargins(0, dp(10), 0, 0);
        card.addView(imageView, imageParams);
        return card;
    }

    private TextView label(String text) {
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextColor(Color.rgb(52, 64, 84));
        view.setTextSize(13);
        view.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        view.setPadding(0, 0, 0, 0);
        return view;
    }

    private TextView detail(String text) {
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextColor(Color.rgb(102, 112, 133));
        view.setTextSize(12);
        view.setPadding(0, dp(4), 0, 0);
        return view;
    }

    private ImageView imageView(String description) {
        ImageView view = new ImageView(this);
        view.setContentDescription(description);
        view.setScaleType(ImageView.ScaleType.FIT_CENTER);
        view.setBackgroundColor(Color.WHITE);
        view.setPadding(dp(6), dp(6), dp(6), dp(6));
        return view;
    }

    private void setDisplayMode(DisplayMode mode) {
        displayMode = mode;
        updateOutputControls();
        displayLastResult();
    }

    private void updateOutputControls() {
        if (cutoutModeButton != null) {
            styleModeButton(cutoutModeButton, displayMode == DisplayMode.CUTOUT);
        }
        if (highlightModeButton != null) {
            styleModeButton(highlightModeButton, displayMode == DisplayMode.HIGHLIGHT);
        }
        for (int i = 0; i < colorSwatches.size(); i++) {
            int color = HIGHLIGHT_COLORS[i];
            boolean selected = displayMode == DisplayMode.HIGHLIGHT && color == highlightColor;
            colorSwatches.get(i).setBackground(swatchBackground(color, selected));
            colorSwatches.get(i).setAlpha(displayMode == DisplayMode.HIGHLIGHT ? 1.0f : 0.55f);
        }
    }

    private void styleModeButton(Button button, boolean selected) {
        button.setTextColor(selected ? Color.WHITE : Color.rgb(132, 45, 24));
        if (selected) {
            button.setBackground(gradient(
                    dp(22),
                    Color.rgb(238, 76, 124),
                    Color.rgb(255, 137, 76)
            ));
        } else {
            button.setBackground(roundedStroke(
                    Color.WHITE,
                    dp(22),
                    Color.rgb(255, 221, 205),
                    dp(1)
            ));
        }
    }

    private GradientDrawable swatchBackground(int color, boolean selected) {
        GradientDrawable drawable = rounded(color, dp(17));
        drawable.setStroke(
                selected ? dp(3) : dp(1),
                selected ? Color.rgb(16, 24, 40) : Color.rgb(255, 221, 205)
        );
        return drawable;
    }

    private GradientDrawable roundedStroke(int color, int radius, int strokeColor, int strokeWidth) {
        GradientDrawable drawable = rounded(color, radius);
        drawable.setStroke(strokeWidth, strokeColor);
        return drawable;
    }

    private GradientDrawable gradient(int radius, int... colors) {
        GradientDrawable drawable = new GradientDrawable(GradientDrawable.Orientation.TL_BR, colors);
        drawable.setCornerRadius(radius);
        return drawable;
    }

    private GradientDrawable rounded(int color, int radius) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(color);
        drawable.setCornerRadius(radius);
        return drawable;
    }

    private void loadModel() {
        executor.execute(() -> {
            try {
                ExecuTorchSegmenter loadedSegmenter = new ExecuTorchSegmenter(this);
                if (destroyed) {
                    loadedSegmenter.close();
                    return;
                }
                segmenter = loadedSegmenter;
                runOnUiThread(() -> {
                    status.setText("Model ready. Choose an image with a clear supported subject.");
                    setRunButtonEnabled(selectedBitmap != null);
                });
            } catch (Throwable exc) {
                runOnUiThread(() -> status.setText("Could not load model assets: " + describe(exc)));
            }
        });
    }

    private void pickPhoto() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        startActivityForResult(intent, PICK_IMAGE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != PICK_IMAGE || resultCode != RESULT_OK || data == null) {
            return;
        }
        Uri uri = data.getData();
        if (uri == null) {
            return;
        }
        try (InputStream stream = getContentResolver().openInputStream(uri)) {
            selectedBitmap = BitmapFactory.decodeStream(stream);
            lastResult = null;
            originalView.setImageBitmap(selectedBitmap);
            cutoutView.setImageDrawable(new CheckerboardDrawable(dp(18)));
            status.setText("Image ready. Remove the background to create a post asset.");
            setRunButtonEnabled(segmenter != null);
        } catch (Throwable exc) {
            status.setText("Could not open image: " + describe(exc));
        }
    }

    private void runCutout() {
        if (selectedBitmap == null || segmenter == null) {
            return;
        }
        setRunButtonEnabled(false);
        status.setText("Creating cutout with ExecuTorch...");
        executor.execute(() -> {
            try {
                CutoutResult result = segmenter.cutout(selectedBitmap);
                runOnUiThread(() -> {
                    lastResult = result;
                    displayLastResult();
                    if (result.foregroundPixels == 0) {
                        status.setText("No supported foreground object found. Try a clearer person, cat, dog, car, or similar VOC class.");
                    } else {
                        status.setText(String.format(
                                Locale.US,
                                "Post asset ready. Detected: %s (%.1f%% of image). Use Cutout or Highlight.",
                                result.label,
                                result.foregroundFraction * 100.0f
                        ));
                    }
                    timing.setText(String.format(
                            Locale.US,
                            "Preprocess: %d ms   Forward: %d ms   Postprocess: %d ms   Total: %d ms",
                            result.preprocessMs,
                            result.forwardMs,
                            result.postprocessMs,
                            result.totalMs()
                    ));
                    setRunButtonEnabled(true);
                });
            } catch (Throwable exc) {
                runOnUiThread(() -> {
                    status.setText("Cutout failed: " + describe(exc));
                    setRunButtonEnabled(true);
                });
            }
        });
    }

    private String describe(Throwable exc) {
        String message = exc.getMessage();
        if (message == null || message.isEmpty()) {
            return exc.getClass().getSimpleName();
        }
        return message;
    }

    private void setRunButtonEnabled(boolean enabled) {
        runButton.setEnabled(enabled);
        runButton.setTextColor(enabled ? Color.WHITE : Color.rgb(152, 162, 179));
        if (enabled) {
            runButton.setBackground(gradient(
                    dp(24),
                    Color.rgb(238, 76, 124),
                    Color.rgb(255, 137, 76)
            ));
        } else {
            runButton.setBackground(rounded(Color.rgb(234, 236, 240), dp(24)));
        }
    }

    private void displayLastResult() {
        if (lastResult == null) {
            return;
        }
        if (displayMode == DisplayMode.HIGHLIGHT && lastResult.foregroundPixels > 0) {
            cutoutView.setImageBitmap(MaskPostprocessor.makeHighlight(
                    lastResult.content,
                    lastResult.alphaMask,
                    highlightColor
            ));
        } else {
            cutoutView.setImageBitmap(lastResult.cutout);
        }
    }

    @Override
    protected void onDestroy() {
        destroyed = true;
        ExecuTorchSegmenter loadedSegmenter = segmenter;
        if (loadedSegmenter != null) {
            executor.execute(loadedSegmenter::close);
        }
        executor.shutdown();
        super.onDestroy();
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
