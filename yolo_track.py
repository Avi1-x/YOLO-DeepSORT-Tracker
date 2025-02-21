import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def initialize_model_and_tracker(model_path):
    """Initialize YOLO model and DeepSORT tracker."""
    model = YOLO(model_path, task="detect")
    tracker = DeepSort(
        max_age=5,
        n_init=5,
        nms_max_overlap=0.7
    )
    return model, tracker

def process_detections(results, confidence_threshold):
    """Process YOLO results into DeepSORT detection format."""
    detections = []
    detection_classes = []
    
    for result in results:
        boxes = result.boxes
        for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
            if conf > confidence_threshold:
                x1, y1, x2, y2 = map(int, box[:4])
                class_id = int(cls)
                # DeepSORT format: ([left, top, width, height], confidence, class_id)
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), class_id))
                detection_classes.append(class_id)
    
    return detections, detection_classes

def draw_tracks(frame, tracks, detection_classes, class_names):
    """Draw bounding boxes and labels for tracked objects."""
    for i, track in enumerate(tracks):
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        
        # Get class name if available            
        label = f"ID: {track_id}"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label with background for better visibility
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        y1_label = max(y1 - 10, label_size[1])
        cv2.rectangle(frame, (x1, y1_label - label_size[1] - baseline),
                     (x1 + label_size[0], y1_label + baseline),
                     (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, label, (x1, y1_label),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return frame

def main():
    # Configuration
    model_path = "./models/yolov8l.onnx"
    video_path = "./videos/football.mp4"
    output_path = "./videos/output_tracking.mp4"
    confidence_threshold = 0.5
    frame_skip = 2  # Skip every 2 frames
    
    # Initialize model and tracker
    model, tracker = initialize_model_and_tracker(model_path)
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))  # Keep original fps

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames based on frame_skip value
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue  # Skip this frame and go to the next one
                
            # Run YOLOv8 detection
            results = model(frame)
            
            # Process detections
            detections, detection_classes = process_detections(results, confidence_threshold)
            
            # Update tracker
            tracks = tracker.update_tracks(detections, frame=frame)
            
            # Draw results
            frame = draw_tracks(frame, tracks, detection_classes, model.names)
            
            # Write frame to output video
            out.write(frame)

            # Display output
            cv2.imshow("YOLOv8 Detection with DeepSORT Tracking", frame)
            
            # Check for ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    finally:
        # Clean up
        cap.release()
        out.release()  # Save the video
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
