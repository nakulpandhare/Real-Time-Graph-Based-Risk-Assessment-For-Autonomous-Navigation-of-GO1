import cv2
import numpy as np

# Load YOLOv4-tiny
net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Prepare the input
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Forward pass
    outputs = net.forward(output_layers)

    # Draw detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.6:
                label = class_names[class_id]
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show the result
    cv2.imshow("YOLOv4-Tiny Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

# Load YOLOv4-tiny
net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Prepare the input
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Forward pass
    outputs = net.forward(output_layers)

    # Draw detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.6:
                label = class_names[class_id]
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show the result
    cv2.imshow("YOLOv4-Tiny Detection", frame) 
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()