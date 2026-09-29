from picamera2 import Picamera2
import cv2

# Haar cascade paths
FACE_CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_eye.xml"

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

if face_cascade.empty():
    raise RuntimeError("Face cascade failed to load")

if eye_cascade.empty():
    raise RuntimeError("Eye cascade failed to load")

# Camera
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (1280, 720),
        "format": "RGB888"
    }
)

picam2.configure(config)
picam2.start()

print("Eye detection test started.")
print("Press Q to quit.")

while True:

    frame = picam2.capture_array()

    # RGB -> BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve contrast slightly
    gray = cv2.equalizeHist(gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120)
    )

    for (x, y, w, h) in faces:

        # Face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Only search for eyes in the upper half of the face
        upper_face_gray = gray[
            y:y + int(h * 0.60),
            x:x + w
        ]

        upper_face_color = frame[
            y:y + int(h * 0.60),
            x:x + w
        ]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(
            upper_face_gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(25, 15)
        )

        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(
                upper_face_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (0, 255, 0),
                2
            )

        # Display number of detected eyes
        cv2.putText(
            frame,
            f"Eyes detected: {len(eyes)}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Driver Eye Detection Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()

print("Eye detection test stopped.")
