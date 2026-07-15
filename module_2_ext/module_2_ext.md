# Module 2: YOLOv8s Object Detection with an AI Portal Model

> Draft status: keep this section unpublished until the AI Portal model page
> and download path are public. When released, integrate it into Module 2. The
> current workflow assumes access to `Arm/yolov8s-int8-xnnpack-executorch`.

## Introduction

In this section, you will use a pre-optimized YOLOv8s object detection model
packaged for ExecuTorch with XNNPACK. Instead of exporting and quantizing the
model yourself, you will download the prepared `.pte` artifact and focus on
application integration.

Earlier in this module, you used the full AOT path for MobileNetV2: prepare a
PyTorch model, lower it to ExecuTorch with XNNPACK, quantize it with PT2E, copy
the `.pte` artifact to the Raspberry Pi, and measure latency and
classification output on the target.

This section uses a different workflow. Instead of doing the AOT work
yourself, you start from a model that has already been optimized for a
specific target. The AI Portal provides a model garden where you can find
models by task, runtime, hardware target, delegate, and quantization
format. For this workflow, the model is:

- model: YOLOv8s INT8
- runtime: ExecuTorch
- delegate: XNNPACK
- target: Raspberry Pi 5, Edge Linux, `arm64`
- package: `Arm/yolov8s-int8-xnnpack-executorch`

In this section, you skip local export and quantization. The work is to
integrate a pre-optimized `.pte` artifact into an application and understand
the preprocessing, postprocessing, and runtime checks needed around it.

| Workflow step | What changes in this section |
|---|---|
| Model preparation | Download a pre-optimized AI Portal package instead of exporting from PyTorch |
| Quantization | Use the packaged INT8 model instead of running PTQ calibration locally |
| Runtime | Run ExecuTorch with XNNPACK on Raspberry Pi |
| Application work | Add camera input, YOLO preprocessing, postprocessing, and box drawing |

YOLO stands for "You Only Look Once." It is a family of object detection models
that predicts object locations and classes in one forward pass. YOLOv8s is the
small variant of YOLOv8, so it is a practical choice when you want object
detection on an edge CPU rather than only image classification.

Earlier in this module, we used MobileNetV2 for image classification. Object
detection is more complex because the model output is not a single class score
vector. A detector returns boxes and class scores, and the application must
apply confidence filtering, box conversion, and non-maximum suppression before
the output is useful.

The scripts follow the same layout used earlier in this module:

- `module_2_ext/scripts/pi`: Raspberry Pi download and application entry points
- `module_2_ext/scripts/pi/core`: YOLOv8s runtime, preprocessing, and postprocessing helpers
- `module_2_ext/scripts/_support/web`: web page rendering for the webcam demo

![AI Portal model garden with YOLOv8s INT8 ExecuTorch with XNNPACK visible.](images/ai-portal-model-garden-yolov8s.png)

## AI Portal Model Garden

The AI Portal Model Garden shows YOLOv8s INT8 optimized for
Raspberry Pi 5 with ExecuTorch and XNNPACK. It reports:

| Metric | AI Portal value |
|---|---:|
| mAP@0.50:0.95 | 42.04% |
| mAP@0.50 | 47.06% |
| mAP@0.75 | 37.02% |
| Frames per second | 1.78 fps |
| Latency | 562.754 ms |
| Delegation | 96.27% |
| Peak memory | 93.84 MB |
| Model size | about 17 MB |

![AI Portal detail page for YOLOv8s INT8 ExecuTorch with XNNPACK.](images/ai-portal-yolov8s-model-detail.png)

## Raspberry Pi Setup

Connect to the Raspberry Pi over SSH and forward the Object Detection app port:

```bash
ssh -L 8094:localhost:8094 <user>@<pi>
```

This keeps the YOLOv8s web app available at `http://localhost:8094` on your
development machine while the server runs on the Raspberry Pi.

Use the same Raspberry Pi environment as earlier in this module:

```bash
cd ~/executorch_on_arm
source .cpu_lab_venv/bin/activate
python -m pip install -r module_2_ext/requirements.txt
```

## Download and Prepare the Pre-Optimized Model on the Raspberry Pi

Download the AI Portal model package from Hugging Face directly on the Pi:

```bash
module_2_ext/scripts/pi/download_yolov8s_ai_portal_model.py \
  --repo-id Arm/yolov8s-int8-xnnpack-executorch \
  --out module_2_ext/artifacts/yolov8s_int8_xnnpack
```

The script does three jobs. First, it downloads the model package from Hugging
Face:

```python
snapshot_path = Path(
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        allow_patterns=["*.pte", "*.json", "*.txt", "*.md"],
    )
)
```

The `allow_patterns` list keeps the download focused on files the application
needs: ExecuTorch `.pte` artifacts, metadata, labels, and documentation. It
does not download training checkpoints or unrelated repository files.

Next, the script copies those files into the local artifact folder:

```python
for path in source.rglob("*"):
    if path.suffix.lower() not in {".pte", ".json", ".txt", ".md"}:
        continue
    shutil.copy2(path, out / path.name)
```

Finally, it makes sure the downloaded package has the files expected by the web
app:

```python
labels_path = out / "coco_labels.json"
pte_name = select_pte(out).name
spec_path = out / "model_spec.json"
```

The AI Portal package may use its own `.pte` filename, such as
`yolov8s-int8-executorch.pte`. The script selects the INT8 `.pte` artifact and
records that filename in `model_spec.json`, so the server command does not need
to hard-code it. It also writes `coco_labels.json` if the package does not
include one. Those labels are needed to turn YOLO class IDs into names such as
`person`, `chair`, or `laptop`.

The downloaded `model_spec.json` records the application contract:

```json
{
  "model_variant": "yolov8s",
  "task": "object_detection",
  "runtime": "executorch",
  "backend": "xnnpack",
  "precision": "int8",
  "inputs": [
    {
      "shape": [1, 3, 640, 640],
      "color": "rgb",
      "value_range": [0.0, 1.0]
    }
  ]
}
```

The model is already quantized, so this workflow does not run
calibration or PT2E conversion.

## Attach your USB Webcam to the Pi and Run the Webcam Detector

Attach a USB webcam to the Raspberry Pi. The app
uses OpenCV camera index `0`, which is usually the first connected camera. The
`--camera` argument can also accept a different OpenCV camera index, such as
`1`, or a video file path if you want to test without a live camera.

Start the app on the Raspberry Pi:

```bash
module_2_ext/scripts/pi/serve_yolov8s_webcam.py \
  --spec ~/executorch_on_arm/module_2_ext/artifacts/yolov8s_int8_xnnpack/model_spec.json \
  --camera 0 \
  --host 0.0.0.0 \
  --port 8094
```

Open `http://localhost:8094` on the development machine.

The app displays the camera frame, detected objects, inference latency, and
end-to-end pipeline timing:

![YOLOv8s webcam object detection demo running with ExecuTorch with XNNPACK.](images/yolov8s-webcam-detection-demo.png)

## Understand the Application Code

The app follows the same high-level pattern as the MobileNetV2 web demo:
create a runner, accept input from an application source, call the ExecuTorch
runtime, and convert the raw model output into something the user can read.

The server entry point creates a `YoloV8Runner` from the downloaded
`model_spec.json`:

```python
runner = YoloV8Runner(args.spec)
```

The runner uses the `.pte` artifact recorded in `model_spec.json`. This keeps
the server command simple and avoids hard-coding the AI Portal package's
downloaded filename:

```python
from executorch.extension.pybindings.portable_lib import _load_for_executorch

model_path = spec_path.parent / self.spec["pte"]
self.model = _load_for_executorch(str(model_path))
```

`_load_for_executorch()` loads the prepared `.pte` artifact through the
ExecuTorch Python bindings. The runtime call itself is small:

```python
with torch.inference_mode(), suppress_stderr():
    raw_output = self.model.forward((input_tensor,))[0]
```

The tuple around `input_tensor` is intentional. ExecuTorch programs receive
their inputs as a tuple, even when the model has only one input tensor. The
`[0]` selects the first output tensor from the returned output tuple.

The difference from MobileNetV2 is the data around the model call. MobileNetV2
converted one image into a normalized tensor and then used softmax plus top-k.
YOLOv8s converts each camera frame into a `1x3x640x640` tensor, runs the
detector, and turns raw detector output into boxes:

```python
input_tensor, scale, pad = self._preprocess(frame_bgr)
raw_output = self.model.forward((input_tensor,))[0]
detections = self._postprocess(raw_output, frame_bgr.shape[:2], scale, pad, 0.25, 0.45)
```

Preprocessing uses letterboxing, which resizes the image without changing
its aspect ratio and pads the remaining area:

```python
scale = min(self.input_size / height, self.input_size / width)
resized = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized

tensor = torch.from_numpy(canvas).permute(2, 0, 1).contiguous().float().unsqueeze(0) / 255.0
```

The final tensor shape is `[1, 3, 640, 640]`, matching the input shape recorded
in `model_spec.json`. The image is converted from OpenCV's BGR camera format to
RGB, arranged as NCHW, converted to floating point, and scaled to `[0.0, 1.0]`.

Postprocessing reverses the letterbox padding and scale so the boxes line up
with the original camera frame. This is where object detection differs most from
classification:

- confidence filtering removes low-confidence candidate boxes
- box conversion changes YOLO center-format boxes into corner-format
  boxes
- per-class non-maximum suppression removes overlapping boxes that refer
  to the same object
- drawing code overlays labels and boxes on the frame

The core box handling looks like this:

```python
boxes_xywh = predictions[:, :4]
class_scores = predictions[:, 4:84]

class_ids = class_scores.argmax(axis=1)
scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
keep = scores >= conf_thres

boxes = xywh_to_xyxy(boxes_xywh[keep])
selected = nms_by_class(boxes, scores[keep], class_ids[keep], iou_thres)
```

These steps are part of the application integration. They are not handled by
the `.pte` artifact unless the packaged model explicitly includes them. For
this AI Portal package, ExecuTorch runs the YOLOv8s model, and the Python
application handles camera capture, letterbox preprocessing, confidence
filtering, non-maximum suppression, label lookup, drawing, and web serving.

## Summary

This workflow uses the same target-side ExecuTorch pattern from earlier in this
module, but starts from a pre-optimized AI Portal package. The AOT export and
PTQ work are already done, so the main work is application integration: camera
input, letterbox preprocessing, ExecuTorch runtime execution, detector
postprocessing, box drawing, and checking latency on the Raspberry Pi.

## Draft Release Checklist

Before releasing this section:

- confirm that the AI Portal model page is public
- confirm that `Arm/yolov8s-int8-xnnpack-executorch` can be downloaded
  without private credentials
- refresh the two screenshots if the AI Portal UI or metrics changed
- verify that the downloaded package file names match the commands in
  this draft
