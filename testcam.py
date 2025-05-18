import cv2

# GStreamer pipeline for Jetson-compatible H.264 RTP decoding
pipeline = (
    "udpsrc address=192.168.123.100 port=9201 caps=application/x-rtp,media=video,encoding-name=H264 ! "
    "rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! appsink"
)

# Open video stream
#cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
cap = cv2.VideoCapture(6)

if not cap.isOpened():
    print("❌ Failed to open video stream.")
    exit()

print("✅ Video stream opened successfully!")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Empty frame received.")
        continue

    # Rotate the frame if needed (e.g., upside-down camera)
    rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Display the frame
    cv2.imshow("Go1 Fisheye Camera Stream", rotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
