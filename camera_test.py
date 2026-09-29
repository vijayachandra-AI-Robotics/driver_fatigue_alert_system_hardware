from picamera2 import Picamera2
import cv2

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)

picam2.configure(config)
picam2.start()

print("Camera running. Press Q to quit.")

while True:
    frame = picam2.capture_array()

    # Picamera2 gives RGB; OpenCV displays BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("Driver Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
print("Camera stopped.")
