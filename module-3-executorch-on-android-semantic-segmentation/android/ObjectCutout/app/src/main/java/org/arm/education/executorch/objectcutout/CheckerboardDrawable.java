package org.arm.education.executorch.objectcutout;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.drawable.Drawable;

final class CheckerboardDrawable extends Drawable {
    private final Paint paint = new Paint();
    private final int cellSize;

    CheckerboardDrawable(int cellSize) {
        this.cellSize = cellSize;
    }

    @Override
    public void draw(Canvas canvas) {
        int width = getBounds().width();
        int height = getBounds().height();
        for (int y = 0; y < height; y += cellSize) {
            for (int x = 0; x < width; x += cellSize) {
                boolean dark = ((x / cellSize) + (y / cellSize)) % 2 == 0;
                paint.setColor(dark ? Color.rgb(224, 230, 238) : Color.rgb(247, 250, 252));
                canvas.drawRect(x, y, x + cellSize, y + cellSize, paint);
            }
        }
    }

    @Override
    public void setAlpha(int alpha) {
        paint.setAlpha(alpha);
    }

    @Override
    public void setColorFilter(android.graphics.ColorFilter colorFilter) {
        paint.setColorFilter(colorFilter);
    }

    @Override
    public int getOpacity() {
        return android.graphics.PixelFormat.OPAQUE;
    }
}
