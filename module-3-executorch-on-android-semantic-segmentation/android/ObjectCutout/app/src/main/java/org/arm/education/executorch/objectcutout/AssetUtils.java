package org.arm.education.executorch.objectcutout;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

final class AssetUtils {
    private AssetUtils() {}

    static File copyAssetToFiles(Context context, String assetName) throws Exception {
        File out = new File(context.getFilesDir(), assetName);
        try (InputStream input = context.getAssets().open(assetName);
             FileOutputStream output = new FileOutputStream(out)) {
            byte[] buffer = new byte[64 * 1024];
            int read;
            while ((read = input.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
        }
        return out;
    }

    static List<String> readJsonStringArray(Context context, String assetName) throws Exception {
        JSONArray array = new JSONArray(readText(context, assetName));
        List<String> values = new ArrayList<>();
        for (int i = 0; i < array.length(); i++) {
            values.add(array.getString(i));
        }
        return values;
    }

    static JSONObject readJsonObject(Context context, String assetName) throws Exception {
        return new JSONObject(readText(context, assetName));
    }

    private static String readText(Context context, String assetName) throws Exception {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        try (InputStream input = context.getAssets().open(assetName)) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = input.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
        }
        return output.toString(StandardCharsets.UTF_8.name());
    }
}
