import cv2
import numpy as np
import platform


def _play_alert_sound():
    system = platform.system()
    if system == "Windows":
        try:
            import winsound

            winsound.Beep(2500, 1000)
            return
        except Exception:
            pass
    # Fallback: terminal beep
    try:
        print("\a", end="", flush=True)
    except Exception:
        pass


def run_drowsiness_yawning_detection(camera_index: int = 0):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    if face_cascade.empty() or eye_cascade.empty() or mouth_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade classifiers from OpenCV data directory")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    eye_closed_frames = 0
    yawn_frames = 0

    EYE_CLOSED_THRESHOLD_FRAMES = 15
    YAWN_THRESHOLD_FRAMES = 10

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

            drowsy = False
            yawning = False

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                roi_gray = gray[y : y + h, x : x + w]
                roi_color = frame[y : y + h, x : x + w]

                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

                if len(eyes) == 0:
                    eye_closed_frames += 1
                else:
                    eye_closed_frames = 0

                if eye_closed_frames >= EYE_CLOSED_THRESHOLD_FRAMES:
                    drowsy = True
                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                mouths = mouth_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.5,
                    minNeighbors=15,
                    minSize=(40, 40),
                )

                if len(mouths) > 0:
                    yawn_frames += 1
                    for (mx, my, mw, mh) in mouths[:1]:
                        cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 255, 255), 2)
                else:
                    yawn_frames = 0

                if yawn_frames >= YAWN_THRESHOLD_FRAMES:
                    yawning = True
                    cv2.putText(
                        frame,
                        "YAWNING ALERT",
                        (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

            if drowsy or yawning:
                _play_alert_sound()

            status_text = []
            if drowsy:
                status_text.append("Drowsy")
            if yawning:
                status_text.append("Yawning")
            if not status_text:
                status_text.append("Normal")

            cv2.putText(
                frame,
                " | ".join(status_text),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if not (drowsy or yawning) else (0, 0, 255),
                2,
            )

            cv2.imshow("Driver Drowsiness & Yawning Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_drowsiness_yawning_detection(camera_index=0)
