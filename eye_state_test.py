import cv2
import time
import math
import numpy as np
from picamera2 import Picamera2


# ============================================================
# CONFIGURATION
# ============================================================

FACE_CASCADE = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
LBF_MODEL = "models/lbfmodel.yaml"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Based on our actual Pi Camera measurements:
#
# Eyes OPEN   = approximately 0.370
# Eyes CLOSED = approximately 0.193
#
EAR_THRESHOLD = 0.25

CLOSED_TIME_REQUIRED = 2.0


# ============================================================
# EYE ASPECT RATIO
# ============================================================

def eye_aspect_ratio(eye_points):

    vertical_1 = math.dist(
        eye_points[1],
        eye_points[5]
    )

    vertical_2 = math.dist(
        eye_points[2],
        eye_points[4]
    )

    horizontal = math.dist(
        eye_points[0],
        eye_points[3]
    )

    if horizontal == 0:
        return 0.0

    return (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

face_detector = cv2.CascadeClassifier(
    FACE_CASCADE
)

if face_detector.empty():
    raise RuntimeError(
        "Could not load Haar face detector."
    )


# ============================================================
# LOAD LBF MODEL
# ============================================================

facemark = cv2.face.createFacemarkLBF()

facemark.loadModel(
    LBF_MODEL
)

print("LBF facial landmark model loaded.")


# ============================================================
# CAMERA
# ============================================================

picam2 = Picamera2()

camera_config = picam2.create_preview_configuration(
    main={
        "size": (
            CAMERA_WIDTH,
            CAMERA_HEIGHT
        ),
        "format": "RGB888"
    }
)

picam2.configure(camera_config)

picam2.start()

time.sleep(2)


# ============================================================
# STATE VARIABLES
# ============================================================

eyes_closed_start = None

fatigue_condition = False


print()
print("================================================")
print("          EYE STATE / 2 SECOND TEST")
print("================================================")
print()
print(f"EAR threshold       : {EAR_THRESHOLD}")
print(f"Required closed time: {CLOSED_TIME_REQUIRED} seconds")
print()
print("Test:")
print("1. Keep eyes OPEN.")
print("2. Close both eyes for less than 2 seconds.")
print("3. Open eyes again.")
print("4. Close both eyes for more than 2 seconds.")
print()
print("NO LED / BUZZER / MP3 are connected to this test.")
print()
print("Press Q to quit.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    frame = picam2.capture_array()

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(faces) > 0:

        # Largest face
        face = max(
            faces,
            key=lambda rect:
            rect[2] * rect[3]
        )

        x, y, w, h = face

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 255, 0),
            2
        )

        # IMPORTANT:
        # Facemark requires NumPy array.
        face_array = np.array(
            [face],
            dtype=np.int32
        )

        success, landmarks = facemark.fit(
            frame,
            face_array
        )

        if success and len(landmarks) > 0:

            points = landmarks[0][0]

            # ------------------------------------------------
            # 68-POINT EYE INDEXES
            # ------------------------------------------------

            left_eye = [
                (
                    int(points[i][0]),
                    int(points[i][1])
                )
                for i in range(36, 42)
            ]

            right_eye = [
                (
                    int(points[i][0]),
                    int(points[i][1])
                )
                for i in range(42, 48)
            ]

            # ------------------------------------------------
            # EAR
            # ------------------------------------------------

            left_ear = eye_aspect_ratio(
                left_eye
            )

            right_ear = eye_aspect_ratio(
                right_eye
            )

            average_ear = (
                left_ear + right_ear
            ) / 2.0

            # ------------------------------------------------
            # EYE STATE
            # ------------------------------------------------

            if average_ear <= EAR_THRESHOLD:

                # Eyes are closed

                if eyes_closed_start is None:

                    eyes_closed_start = time.monotonic()

                closed_duration = (
                    time.monotonic()
                    - eyes_closed_start
                )

                # Trigger after 2 seconds
                if closed_duration >= CLOSED_TIME_REQUIRED:

                    fatigue_condition = True

                else:

                    fatigue_condition = False

            else:

                # Eyes are open
                eyes_closed_start = None
                fatigue_condition = False

                closed_duration = 0.0

            # ------------------------------------------------
            # DRAW EYE LANDMARKS
            # ------------------------------------------------

            for i in range(36, 48):

                px = int(points[i][0])
                py = int(points[i][1])

                cv2.circle(
                    frame,
                    (px, py),
                    2,
                    (0, 255, 0),
                    -1
                )

            # ------------------------------------------------
            # DRAW EAR
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Average EAR: {average_ear:.3f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # ------------------------------------------------
            # DISPLAY EYE STATE
            # ------------------------------------------------

            if average_ear <= EAR_THRESHOLD:

                cv2.putText(
                    frame,
                    "EYES CLOSED",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Closed: {closed_duration:.1f} s",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "EYES OPEN",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Closed: 0.0 s",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            # ------------------------------------------------
            # FATIGUE CONDITION
            # ------------------------------------------------

            if fatigue_condition:

                cv2.putText(
                    frame,
                    "FATIGUE CONDITION DETECTED",
                    (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "2+ SECONDS EYES CLOSED",
                    (30, 205),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )

    else:

        # No face detected

        eyes_closed_start = None
        fatigue_condition = False

        cv2.putText(
            frame,
            "FACE NOT DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Driver Fatigue - Eye State Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

picam2.stop()

cv2.destroyAllWindows()

print("Camera stopped.")
print("Test finished.")
