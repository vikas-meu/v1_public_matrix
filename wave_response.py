import cv2
import mediapipe as mp
import serial
import time

# Initialize serial communication
try:
    arduino = serial.Serial('COM12', 9600, timeout=1)
    time.sleep(2)  # Wait for the connection to establish
    print("Serial communication established!")
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    exit()

# Mediapipe setup for pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Flags and variables
wave_threshold = 100  # Threshold for hand movement to detect a wave
previous_y = None  # To track hand movement

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Pose estimation started. Wave your hand to trigger the 'warn' command.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip and process the frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Check for hand waving gesture
        if results.pose_landmarks:
            # Get the right hand's Y coordinate
            right_wrist_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y

            # Convert Y coordinate to pixel value
            h, w, _ = frame.shape
            current_y = int(right_wrist_y * h)

            if previous_y is not None:
                # Check if hand is moving up and down
                movement = abs(current_y - previous_y)
                if movement > wave_threshold:
                    print("Wave detected! Sending 'warn' command to Arduino.")
                    arduino.write(b"warn")  # Send command to Arduino
                    time.sleep(1)  # Avoid spamming the command

            previous_y = current_y  # Update for the next frame

        # Display the frame
        cv2.imshow("Pose Estimation - Wave Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    arduino.write(b"h")  # Send command normal to Arduino
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()
    print("Program terminated.")
