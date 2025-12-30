from ultralytics import YOLO
import cv2
import serial
import time

def main():
    # ----------------- Serial Setup -----------------
    # Replace 'COM3' with your Arduino port on Windows, or '/dev/ttyACM0' on Linux
    ser = serial.Serial('COM7', 9600, timeout=1)
    time.sleep(2)  # wait for Arduino to reset

    # ----------------- Load YOLO -----------------
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
                if cls == 0:  # class 0 = person
                    person_count += 1

                    # Draw bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ----------------- Send count to Arduino -----------------
        ser.write(f"{person_count}\n".encode())  # newline required for Arduino parse
        # optional: small delay
        time.sleep(0.05)

        # ----------------- Display frame -----------------
        cv2.putText(frame, f"COUNT: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imshow("YOLO Human Detection & Counting", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    ser.close()

if __name__ == "__main__":
    main()
