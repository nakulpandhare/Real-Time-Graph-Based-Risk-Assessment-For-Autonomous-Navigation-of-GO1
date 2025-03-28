import cv2
import numpy as np
from static_graph import update_graph  # Graph updater

# GStreamer pipeline for Jetson-compatible H.264 RTP decoding
pipeline = (
    "udpsrc address=192.168.123.100 port=9201 caps=application/x-rtp,media=video,encoding-name=H264 ! "
    "rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! appsink"
)

# Load YOLOv4-tiny
net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Or DNN_TARGET_CUDA if Jetson supports it

# Load class names
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

allowed_classes = {"person", "chair", "bottle"}

# Get output layer names
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Open video stream
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Failed to open video stream.")
    exit()

print("✅ Video stream opened successfully!")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Empty frame received.")
        continue

    rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)
    height, width, _ = rotated_frame.shape

    # Run YOLOv4-tiny detection
    blob = cv2.dnn.blobFromImage(rotated_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    # Prepare detection data
    boxes = []
    confidences = []
    class_ids = []
    positions = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            label = class_names[class_id]
            if confidence > 0.4 and label in allowed_classes:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # Estimate position
                if center_x < width / 3:
                    positions.append("Left")
                elif center_x > 2 * width / 3:
                    positions.append("Right")
                else:
                    positions.append("Front")

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.4, nms_threshold=0.4)

    root_nodes = {}

    for i in indices:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        x, y, w, h = boxes[i]
        label = class_names[class_ids[i], position[i]]
        conf = confidences[i]
        position = positions[i]

        # Create unique ID for node
        object_id = f"{label}_{x}_{y}"
        root_nodes[object_id] = position

        # Draw bounding box and label
        cv2.rectangle(rotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(rotated_frame, f"{label} {conf:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Update graph with current detections
    if root_nodes:
        update_graph(root_nodes)

    # Display left half of the rotated frame
    half_frame = rotated_frame[:, :width // 2]
    cv2.imshow("Go1 Fisheye + YOLOv4-tiny", rotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
