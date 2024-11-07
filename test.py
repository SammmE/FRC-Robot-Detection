import cv2
from ultralytics import YOLO

SOURCE_PATH = "./FRC 6423 Robot POV.mp4"
OUTPUT_PATH = "./output.mp4"
CONFIDENCE = 0.5
SHOW = True
FAST = True
LABELS = ["Note", "Robot"]

# Load the YOLOv11 model
model = YOLO(r".\runs\detect\train4\weights\last.pt")

# Open the video file
cap = cv2.VideoCapture(SOURCE_PATH)

# Loop through the video frames
for result in model(source=SOURCE_PATH, stream=True, conf=CONFIDENCE):
    # Get the original frame from the results
    frame = result.orig_img

    # Plot the detection results on the frame
    annotated_frame = result.plot()

    # Display the annotated frame
    cv2.imshow(SOURCE_PATH, annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
