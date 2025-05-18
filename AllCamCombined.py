import cv2
import numpy as np
from static_graph import update_graph
import robot_sim
from multiprocessing import Process, Value
import ctypes

# GStreamer pipelines
pipelines = {
    "front": "udpsrc address=192.168.123.100 port=9201 caps=application/x-rtp,media=video,encoding-name=H264 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink",
    "left": "udpsrc address=192.168.123.100 port=9203 caps=application/x-rtp,media=video,encoding-name=H264 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink",
    "right": "udpsrc address=192.168.123.100 port=9204 caps=application/x-rtp,media=video,encoding-name=H264 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink"
}

# Load YOLOv4-tiny
net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

allowed_classes = {"person", "bottle", "chair", "handbag", "tvmonitor", "fire hydrant", "sports ball", "laptop"}

def camera_loop(risk_flag=None):
    caps = {pos: cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER) for pos, pipeline in pipelines.items()}
    for pos, cap in caps.items():
        if not cap.isOpened():
            print(f"❌ Failed to open {pos} video stream.")
            return
        else:
            print(f"✅ {pos.capitalize()} video stream opened successfully!")

    while True:
        root_nodes = {}
        combined_frame = None

        for position, cap in caps.items():
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"⚠️ Empty frame received from {position} camera.")
                continue

            rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)
            height, width = rotated_frame.shape[:2]
            third_width = width // 3

            alpha = 1.2
            beta = -40
            corrected_frame = cv2.convertScaleAbs(rotated_frame, alpha=alpha, beta=beta)
            blob = cv2.dnn.blobFromImage(corrected_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

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

            for idx in indices:
                i = idx[0] if isinstance(idx, (tuple, list, np.ndarray)) else idx
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                label = class_names[class_id]
                center_x, center_y = centers[i]

                if position == "front":
                    if center_x < third_width:
                        pos_label = "FrontLeft"
                    elif center_x < 2 * third_width:
                        pos_label = "Front"
                    else:
                        pos_label = "FrontRight"
                elif position == "left":
                    pos_label = "Left"
                elif position == "right":
                    pos_label = "Right"

                instance_count[label] += 1
                label_id = f"{label}{instance_count[label]}"
                root_nodes[label_id] = pos_label

                cv2.rectangle(rotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(rotated_frame, f"{label_id} {confidences[i]:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            resized_frame = cv2.resize(rotated_frame, (600, 500))
            if combined_frame is None:
                combined_frame = resized_frame
            else:
                combined_frame = np.hstack((combined_frame, resized_frame))

        update_graph(root_nodes, risk_flag=risk_flag)

        front_count = sum(1 for pos in root_nodes.values() if pos in {"Front", "FrontLeft", "FrontRight"})
        if risk_flag is not None:
            risk_flag.value = 1 if front_count > 1 else 0

        if combined_frame is not None:
            cv2.imshow("Combined YOLOv4-Tiny Detection", combined_frame)
            if cv2.waitKey(1) == ord('q'):
                break

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    emergency_stop_flag = Value(ctypes.c_int, 0)
    sim_process = Process(target=robot_sim.run_simulation, args=(emergency_stop_flag,))
    sim_process.start()

    camera_loop(risk_flag=emergency_stop_flag)

    sim_process.join()
