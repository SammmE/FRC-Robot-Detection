import cv2
from ultralytics import YOLO
import tqdm

SOURCE_PATH = "./FRC 6423 Robot POV.mp4"
OUTPUT_PATH = "./output.mp4"
CONFIDENCE = 0.5
SHOW = True
FAST = True
LABELS = ["Note", "Robot"]
model = YOLO(r".\runs\detect\train4\weights\last.pt")

cap = cv2.VideoCapture(SOURCE_PATH)

if OUTPUT_PATH:
    out = cv2.VideoWriter(
        OUTPUT_PATH,
        int(cap.get(cv2.CAP_PROP_FOURCC)),
        cap.get(cv2.CAP_PROP_FPS),
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )


if SHOW:
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if FAST or ret:
            results = model(frame, conf=CONFIDENCE)

            annotated_frame = frame.copy()

            for result in results:
                boxes = (
                    result.boxes
                )  # Assuming 'boxes' is an attribute of the result object
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]  # Assuming xyxy format
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Add label (if available)
                    if hasattr(box, "cls"):
                        label = LABELS[int(box.cls.item())]
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                        )

            # Display the annotated frame
            cv2.imshow(SOURCE_PATH, annotated_frame)

            # if OUTPUT_PATH:
            #     out.write(annotated_frame)

            if (
                cv2.waitKey(25) & 0xFF == ord("q")
                or cv2.getWindowProperty(SOURCE_PATH, cv2.WND_PROP_VISIBLE) < 1
            ):
                break
        else:
            break
else:
    vid = []
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        vid.append((ret, frame))

    for ret, frame in tqdm.tqdm(vid):
        out.write(model(frame)[0].plot())


cap.release()
cv2.destroyAllWindows()
if OUTPUT_PATH:
    out.release()
