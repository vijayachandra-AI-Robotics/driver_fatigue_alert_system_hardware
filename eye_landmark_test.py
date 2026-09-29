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


# ============================================================
# EYE ASPECT RATIO
# ============================================================

def eye_aspect_ratio(eye_points):

    vertical_1 = math.dist(eye_points[1], eye_points[5])
    vertical_2 = math.dist(eye_points[2], eye_points[4])

    horizontal = math.dist(eye_points[0], eye_points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

face_detector = cv2.CascadeClassifier(FACE_CASCADE)

if face_detector.empty():
    raise RuntimeError("Could not load Haar face detector.")


# ============================================================
# LOAD LBF FACIAL LANDMARK MODEL
# ============================================================

facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(LBF_MODEL)

print("LBF facial landmark model loaded successfully.")


# ============================================================
# CAMERA SETUP
# ============================================================

picam2 = Picamera2()

camera_config = picam2.create_preview_configuration(
    main={
        "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
        "format": "RGB888"
    }
)

picam2.configure(camera_config)
picam2.start()

time.sleep(2)


print()
print("================================================")
print("       EYE LANDMARK TEST")
print("================================================")
print()
print("Look directly at the camera.")
print()
print("1. Keep your eyes OPEN for about 5 seconds.")
print("2. Close BOTH eyes for about 5 seconds.")
print("3. Open your eyes again.")
print()
print("Watch the EAR values.")
print("Press Q to quit.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # Capture image
    frame = picam2.capture_array()

    # Picamera2 gives RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Grayscale for Haar detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(faces) > 0:

        # Select largest face
        face = max(
            faces,
            key=lambda rect: rect[2] * rect[3]
        )

        x, y, w, h = face

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 255, 0),
            2
        )

        # IMPORTANT:
        # FacemarkLBF.fit() requires a NumPy array
        face_array = np.array([face], dtype=np.int32)

        # Detect 68 facial landmarks
        success, landmarks = facemark.fit(
            frame,
            face_array
        )

        if success and len(landmarks) > 0:

            points = landmarks[0][0]

            # =================================================
            # STANDARD 68-POINT MODEL
            #
            # LEFT EYE  = 36 - 41
            # RIGHT EYE = 42 - 47
            # =================================================

            left_eye = [
                (int(points[i][0]), int(points[i][1]))
                for i in range(36, 42)
            ]

            right_eye = [
                (int(points[i][0]), int(points[i][1]))
                for i in range(42, 48)
            ]

            # Calculate EAR
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            average_ear = (
                left_ear + right_ear
            ) / 2.0

            # =================================================
            # DRAW ALL 68 LANDMARKS
            # =================================================

            for i, point in enumerate(points):

                px = int(point[0])
                py = int(point[1])

                cv2.circle(
                    frame,
                    (px, py),
                    2,
                    (0, 255, 0),
                    -1
                )

            # =================================================
            # DRAW EYE LANDMARK NUMBERS
            # =================================================

            for i in range(36, 48):

                px = int(points[i][0])
                py = int(points[i][1])

                cv2.putText(
                    frame,
                    str(i),
                    (px + 4, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 255),
                    1
                )

            # =================================================
            # DRAW EYE CONTOURS
            # =================================================

            for eye in (left_eye, right_eye):

                for i in range(6):

                    p1 = eye[i]
                    p2 = eye[(i + 1) % 6]

                    cv2.line(
                        frame,
                        p1,
                        p2,
                        (0, 255, 255),
                        2
                    )

            # =================================================
            # DISPLAY EAR VALUES
            # =================================================

            cv2.putText(
                frame,
                f"Left EAR   : {left_ear:.3f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Right EAR  : {right_ear:.3f}",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Average EAR: {average_ear:.3f}",
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "EYES DETECTED",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

    else:

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
        "Driver Fatigue - Eye Landmark Test",
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
