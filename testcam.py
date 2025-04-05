import cv2
import numpy as np
from static_graph import update_graph  # Graph updater

# Load YOLOv4-tiny
net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

allowed_classes = {"person", "bottle", "chair"}

# Get output layer names
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Open two video streams
cap1 = cv2.VideoCapture(6, cv2.CAP_V4L2)  # Intel RealSense
cap2 = cv2.VideoCapture(0, cv2.CAP_V4L2)  # USB webcam

# Just before reading from camera, you can set:
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


if not cap1.isOpened() or not cap2.isOpened():
    print("❌ Failed to open video streams.")
    exit()

print("✅ Video streams opened successfully!")

while True:
    # Clear buffer to reduce latency
    for _ in range(4):
        cap1.grab()
        cap2.grab()

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("⚠️ Empty frame received.")
        continue

    # Resize second frame to match first
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    # Combine side-by-side
    master_frame = cv2.hconcat([frame1, frame2])
    height, width = master_frame.shape[:2]
    half_width = width // 2  # To split detection zones

    # YOLO detection
    blob = cv2.dnn.blobFromImage(master_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    centers = []

    for output in detections:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            label = class_names[class_id]

            if confidence > 0.3 and label in allowed_classes:
                center_x = int(det[0] * width)
                center_y = int(det[1] * height)
                w = int(det[2] * width)
                h = int(det[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                centers.append((center_x, center_y))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)

    instance_count = {cls: 0 for cls in allowed_classes}
    root_nodes = {}

    for idx in indices:
        i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        label = class_names[class_id]
        conf = confidences[i]
        center_x, center_y = centers[i]

        # Determine position based on camera
        if center_x < half_width:  # cap1 (left side of combined frame)
            rel_x = center_x  # relative to cap1
            if rel_x < half_width / 3:
                position = "FrontLeft"
            elif rel_x > 2 * half_width / 3:
                position = "FrontRight"
            else:
                position = "Front"
        else:  # cap2 (right side of combined frame)
            position = "Right"

        # Label instances
        instance_count[label] += 1
        label_id = f"{label}{instance_count[label]}"
        root_nodes[label_id] = position

        # Draw bounding box and label
        cv2.rectangle(master_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(master_frame, f"{label_id} {conf:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Update dynamic graph
    if root_nodes:
        update_graph(root_nodes)

    cv2.imshow("Camera View", master_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()