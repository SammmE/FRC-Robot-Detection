import cv2
from ultralytics import YOLO

SOURCE_PATH = "./FRC 6423 Robot POV.mp4"
OUTPUT_PATH = "./output.mp4"
CONFIDENCE = 0.5
SHOW = False
LABELS = ["Note", "Robot"]

# Load the YOLOv11 model
model = YOLO(r".\runs\detect\train4\weights\last.pt")

# Open the video file
cap = cv2.VideoCapture(SOURCE_PATH)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# Process the video
for result in model(source=SOURCE_PATH, stream=True, conf=CONFIDENCE):
    frame = result.orig_img
    annotated_frame = result.plot()

    # Write the frame to the output video
    out.write(annotated_frame)

    if SHOW:
        cv2.imshow("YOLOv11 Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
