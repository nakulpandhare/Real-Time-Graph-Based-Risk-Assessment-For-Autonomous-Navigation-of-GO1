# cam.py
import cv2
import numpy as np
from static_graph import update_graph  # Your graph update function
import robot_sim
from multiprocessing import Process, Value
import ctypes

# Load YOLOv4-tiny
net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

allowed_classes = {"person", "bottle", "chair", "handbag", "tvmonitor", "fire hydrant", "sports ball", "laptop"}

def camera_loop(object_queue=None, risk_flag=None):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    print("✅ Webcam opened successfully!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        third_width = width // 3

        # Apply brightness/contrast correction
        alpha = 1.2  # Contrast
        beta = -40   # Brightness
        corrected_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # Use corrected frame in blob
        blob = cv2.dnn.blobFromImage(corrected_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        #blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
        outputs = net.forward(output_layers)

        boxes, confidences, class_ids, centers = [], [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.6:
                    label = class_names[class_id]
                    if label not in allowed_classes:
                        continue

                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    centers.append((center_x, center_y))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.4)
        instance_count = {cls: 0 for cls in allowed_classes}
        root_nodes = {}
        detections_to_send = []

        for idx in indices:
            i = idx[0] if isinstance(idx, (tuple, list, np.ndarray)) else idx
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            label = class_names[class_id]
            center_x, center_y = centers[i]

            if center_x < third_width:
                position = "FrontLeft"
            elif center_x < 2 * third_width:
                position = "Front"
            else:
                position = "FrontRight"

            instance_count[label] += 1
            label_id = f"{label}{instance_count[label]}"
            root_nodes[label_id] = position

            detections_to_send.append((label, (center_x, center_y)))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_id} {confidences[i]:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        #if root_nodes:
        update_graph(root_nodes, risk_flag=risk_flag)

        front_count = sum(1 for pos in root_nodes.values() if pos in {"Front", "FrontLeft", "FrontRight"})

        if risk_flag is not None:
            risk_flag.value = 1 if front_count > 3 else 0

        #print("⚠️ Risk detected!" if front_count > 2 else "✅ No risk.")

        if object_queue is not None:
            object_queue.put(detections_to_send)

        if object_queue is None:
            resized_frame = cv2.resize(frame, (700, 600)) 
            cv2.imshow("YOLOv4-Tiny Detection", resized_frame)
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Shared flag
    emergency_stop_flag = Value(ctypes.c_int, 0)

    # Start robot simulation in a new process
    sim_process = Process(target=robot_sim.run_simulation, args=(emergency_stop_flag,))
    sim_process.start()

    # Start camera loop
    camera_loop(risk_flag=emergency_stop_flag)

    sim_process.join()
