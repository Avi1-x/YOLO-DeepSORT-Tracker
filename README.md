# YOLOv8 Object Detection and DeepSORT Tracking

![Example GIF](videos/example.gif)

## Features
- **Object Detection**: Uses YOLOv8 to detect objects in video frames with a configurable confidence threshold.
- **Object Tracking**: Implements DeepSORT to track objects across frames and assign unique IDs to each detected object.

## Installation

1. Install dependencies:
    ```bash
    pip install opencv-python torch ultralytics deep_sort_realtime
    ```

2. **Download and export YOLOv8 model to ONNX**:
   - The first script ("convert_model.py") downloads the specified YOLO model from PyTorch and converts it to ONNX format, which can be used in the main tracking script.

   Run the script to export the model:
   ```bash
   python export_model.py
   ```
   
## Usage
1. Configure the `model_path` and `video_path`, variables in `yolo_detect.py` to point to your YOLO model and input video.
2. Run the script:
    ```bash
    python yolo_detect.py
    ```
3. The script will output a video with tracked objects, saved to the specified `output_path`.


## Example Video

Note that the  video used in this project is sourced from [Pexels](https://www.pexels.com), a platform that provides free stock videos for personal and commercial use. You can find the original video [here](https://www.pexels.com/video/group-of-boys-plays-soccer-in-a-soccer-field-2932301/). The video is free for use under the Pexels License.
