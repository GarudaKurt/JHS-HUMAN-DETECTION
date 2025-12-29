from ultralytics import YOLO
import cv2

def main():

    # Load YOLOv8 pretrained model
    model = YOLO("yolov8n.pt")  # small & fast model

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True)

        person_count = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])

                # class 0 = person
                if cls == 0:
                    person_count += 1

                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"COUNT: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("YOLO Human Detection & Counting", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
