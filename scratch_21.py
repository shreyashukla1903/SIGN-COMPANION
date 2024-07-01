import cv2
import helper

model_path = "best.onnx"
yolov7_detector = helper.YOLOv8(model_path, conf_thres=0.3, iou_thres=0.7)

# define a video capture object
vid = cv2.VideoCapture(0)

while True:
    # Capture the video frame by frame
    ret, frame = vid.read()

    # Check if the frame is valid
    if not ret:
        print("Failed to capture frame from the video stream.")
        break

    # Pass the frame to the detector
    detections = yolov7_detector(frame)

    # Draw bounding boxes on the frame if detections exist
    if detections is not None:
        frame = yolov7_detector.draw_detections(frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Check for the quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
