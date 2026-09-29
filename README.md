# Driver Fatigue Alert System

A real-time driver fatigue detection and alert system built with a Raspberry Pi 4 Model B and Raspberry Pi Camera Module 3.

The system continuously monitors the driver's eyes using facial landmarks and Eye Aspect Ratio (EAR). If both eyes remain closed continuously for 2 seconds or longer, the system activates a visual, audible, and voice alert.

## Features

- Real-time driver face monitoring
- 68-point facial landmark detection
- Eye Aspect Ratio (EAR) based eye-state detection
- Continuous eye-closure timing
- 2-second fatigue detection threshold
- Green LED for normal/awake state
- Red LED for fatigue warning
- Piezo buzzer alert
- Voice alert through external speaker
- On-screen warning:
  `WARNING - DRIVER WAKE UP`
- Automatic alarm reset when eyes open again
- Raspberry Pi Camera Module 3 support
- Complete hardware and software test programs included

---

## System Overview

```text
                    Raspberry Pi Camera Module 3
                              |
                              v
                       Live Camera Frame
                              |
                              v
                     Face Detection
                              |
                              v
                  68-Point Facial Landmarks
                              |
                              v
                  Eye Landmark Extraction
                              |
                              v
                   Eye Aspect Ratio (EAR)
                              |
                    +---------+---------+
                    |                   |
                 Eyes Open          Eyes Closed
                    |                   |
                    v                   v
              Normal State       Start Timer
                    |                   |
                    |             >= 2 seconds
                    |                   |
                    |                   v
                    |          FATIGUE DETECTED
                    |                   |
                    |       +-----------+-----------+
                    |       |           |           |
                    |       v           v           v
                    |    Red LED     Buzzer      MP3 Voice
                    |       ON          ON          ON
                    |       |
                    |       v
                    |  WARNING - DRIVER
                    |       WAKE UP
                    |
                    v
               Green LED ON
