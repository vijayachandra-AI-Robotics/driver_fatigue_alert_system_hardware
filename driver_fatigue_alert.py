import cv2
import time
import math
import subprocess
import numpy as np

from picamera2 import Picamera2
from gpiozero import LED, Buzzer


# ============================================================
# DRIVER FATIGUE ALERT SYSTEM
# Raspberry Pi 4 + Pi Camera Module 3
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

FACE_CASCADE = (
    "/usr/share/opencv4/haarcascades/"
    "haarcascade_frontalface_default.xml"
)

LBF_MODEL = "models/lbfmodel.yaml"

ALERT_AUDIO = "data/driver_alert_ringtone.mp3"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Based on our real camera tests:
#
# Eyes open  ≈ 0.289 - 0.370
# Eyes closed ≈ 0.19 - 0.20
#
EAR_THRESHOLD = 0.25

# Eyes must remain continuously closed for this duration
CLOSED_TIME_REQUIRED = 2.0


# ============================================================
# GPIO CONFIGURATION
# ============================================================

GREEN_LED_PIN = 17
RED_LED_PIN = 22
BUZZER_PIN = 23


green_led = LED(GREEN_LED_PIN)
red_led = LED(RED_LED_PIN)
buzzer = Buzzer(BUZZER_PIN)


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
        "ERROR: Could not load Haar face detector."
    )


# ============================================================
# LOAD 68-POINT FACIAL LANDMARK MODEL
# ============================================================

print()
print("Loading LBF facial landmark model...")

facemark = cv2.face.createFacemarkLBF()

facemark.loadModel(
    LBF_MODEL
)

print("LBF model loaded successfully.")


# ============================================================
# CAMERA SETUP
# ============================================================

print("Starting Raspberry Pi Camera...")

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

print("Camera started successfully.")


# ============================================================
# INITIAL SYSTEM STATE
# ============================================================

eyes_closed_start = None

fatigue_condition = False

alert_audio_process = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def set_normal_state():

    global fatigue_condition

    fatigue_condition = False

    green_led.on()
    red_led.off()
    buzzer.off()


def start_alert():

    global fatigue_condition
    global alert_audio_process

    fatigue_condition = True

    # LED / buzzer
    green_led.off()
    red_led.on()
    buzzer.on()

    # --------------------------------------------------------
    # Start MP3 only once
    # --------------------------------------------------------

    if alert_audio_process is None:

        print("Starting driver alert voice...")

        try:

            alert_audio_process = subprocess.Popen(
                [
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    ALERT_AUDIO
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        except Exception as error:

            print(
                "WARNING: Could not start MP3:",
                error
            )


def stop_alert_audio():

    global alert_audio_process

    if alert_audio_process is not None:

        if alert_audio_process.poll() is None:

            try:
                alert_audio_process.terminate()

            except Exception:
                pass

        alert_audio_process = None


def cleanup():

    print()
    print("Shutting down Driver Fatigue Alert System...")

    stop_alert_audio()

    green_led.off()
    red_led.off()
    buzzer.off()

    picam2.stop()

    cv2.destroyAllWindows()

    print("System stopped safely.")


# ============================================================
# START IN NORMAL STATE
# ============================================================

set_normal_state()


# ============================================================
# STARTUP MESSAGE
# ============================================================

print()
print("================================================")
print("       DRIVER FATIGUE ALERT SYSTEM")
print("================================================")
print()
print("Camera              : Pi Camera Module 3")
print("Face landmarks      : LBF 68-point")
print(f"EAR threshold       : {EAR_THRESHOLD}")
print(
    f"Closed-eye duration : "
    f"{CLOSED_TIME_REQUIRED} seconds"
)
print("Green LED           : GPIO17")
print("Red LED             : GPIO22")
print("Buzzer              : GPIO23")
print("Alert audio         :", ALERT_AUDIO)
print()
print("SYSTEM STATUS: MONITORING")
print()
print("Keep your eyes open for normal operation.")
print("Close both eyes for 2+ seconds to test alarm.")
print()
print("Press Q to quit.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # Capture camera frame
        # ----------------------------------------------------

        frame = picam2.capture_array()

        # RGB → BGR
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        # ----------------------------------------------------
        # Face detection
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # FACE FOUND
        # ----------------------------------------------------

        if len(faces) > 0:

            # Select largest face
            face = max(
                faces,
                key=lambda rect:
                rect[2] * rect[3]
            )

            x, y, w, h = face

            # Face rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 255, 0),
                2
            )

            # Facemark requires NumPy array
            face_array = np.array(
                [face],
                dtype=np.int32
            )

            # ------------------------------------------------
            # Detect 68 facial landmarks
            # ------------------------------------------------

            success, landmarks = facemark.fit(
                frame,
                face_array
            )

            if success and len(landmarks) > 0:

                points = landmarks[0][0]

                # ------------------------------------------------
                # STANDARD 68-POINT EYE INDEXES
                #
                # Left eye  = 36-41
                # Right eye = 42-47
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
                # Calculate EAR
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

                    # ============================================
                    # EYES CLOSED
                    # ============================================

                    if eyes_closed_start is None:

                        eyes_closed_start = (
                            time.monotonic()
                        )

                    closed_duration = (
                        time.monotonic()
                        - eyes_closed_start
                    )

                    # --------------------------------------------
                    # FATIGUE TRIGGER
                    # --------------------------------------------

                    if (
                        closed_duration
                        >= CLOSED_TIME_REQUIRED
                    ):

                        if not fatigue_condition:

                            print()
                            print(
                                "!!! FATIGUE CONDITION "
                                "DETECTED !!!"
                            )
                            print(
                                "Eyes closed for "
                                f"{closed_duration:.1f} seconds"
                            )

                            start_alert()

                    else:

                        # Before 2 seconds
                        green_led.off()
                        red_led.off()
                        buzzer.off()

                else:

                    # ============================================
                    # EYES OPEN
                    # ============================================

                    eyes_closed_start = None

                    if fatigue_condition:

                        print(
                            "Eyes opened - "
                            "resetting alert."
                        )

                    fatigue_condition = False

                    green_led.on()
                    red_led.off()
                    buzzer.off()

                    # Stop MP3 if it is still playing
                    stop_alert_audio()

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
                # DISPLAY EAR
                # ------------------------------------------------

                cv2.putText(
                    frame,
                    f"EAR: {average_ear:.3f}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                # ------------------------------------------------
                # DISPLAY STATE
                # ------------------------------------------------

                if fatigue_condition:

                    # ============================================
                    # FATIGUE WARNING
                    # ============================================

                    cv2.putText(
                        frame,
                        "WARNING - DRIVER WAKE UP",
                        (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        frame,
                        "FATIGUE ALERT ACTIVE",
                        (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Eyes closed: "
                        f"{closed_duration:.1f} s",
                        (30, 165),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                elif average_ear <= EAR_THRESHOLD:

                    # ============================================
                    # EYES CURRENTLY CLOSED
                    # ============================================

                    cv2.putText(
                        frame,
                        "EYES CLOSED",
                        (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 165, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Closed: "
                        f"{closed_duration:.1f} s",
                        (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 165, 255),
                        2
                    )

                else:

                    # ============================================
                    # NORMAL
                    # ============================================

                    cv2.putText(
                        frame,
                        "SYSTEM: NORMAL",
                        (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "EYES OPEN",
                        (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2
                    )

        # ----------------------------------------------------
        # NO FACE FOUND
        # ----------------------------------------------------

        else:

            # Reset closed-eye timer
            eyes_closed_start = None

            # For this prototype, no face means monitoring
            # cannot determine eye state.
            #
            # We keep the alarm OFF rather than falsely
            # claiming fatigue.

            fatigue_condition = False

            green_led.off()
            red_led.off()
            buzzer.off()

            stop_alert_audio()

            cv2.putText(
                frame,
                "FACE NOT DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "POSITION FACE IN CAMERA",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2
            )

        # ----------------------------------------------------
        # DISPLAY CAMERA
        # ----------------------------------------------------

        cv2.imshow(
            "DRIVER FATIGUE ALERT SYSTEM",
            frame
        )

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


# ============================================================
# SAFE SHUTDOWN
# ============================================================

except KeyboardInterrupt:

    print()
    print("Program interrupted by user.")


finally:

    cleanup()
