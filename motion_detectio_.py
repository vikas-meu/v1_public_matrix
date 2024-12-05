import cv2
import mediapipe as mp
import serial
import time
import numpy as np

# Initialize serial communication with Arduino
arduino = serial.Serial('COM12', 9600)
time.sleep(2)  # Wait for the connection to establish

# Initialize MediaPipe Hands and Face Detection
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Timer variables
last_wave_time = 0
last_smile_time = 0
last_person_time = 0
wave_cooldown = 10
smile_cooldown = 10
motion_timeout = 5  # Seconds of no motion before sending "bye"

# Flag to check if motion was detected
motion_detected = False
last_frame = None

# Function to send commands to Arduino
def send_command(command):
    print(f"Sending command: {command}")
    arduino.write(f"{command}\n".encode())

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    hand_results = hands.process(rgb_frame)

    # Process face detection
    face_results = face_detection.process(rgb_frame)

    current_time = time.time()

    # Check for waving gesture
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate vertical difference between wrist and middle finger tip
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            if abs(wrist.y - middle_finger_tip.y) > 0.3:
                if current_time - last_wave_time > wave_cooldown:
                    send_command("hi_hello")
                    last_wave_time = current_time
                    time.sleep(10)  # Ensure Arduino processes the "warn" command
                    send_command("normal")

    # Check for smile detection
    if face_results.detections:
        for detection in face_results.detections:
            bbox_c = detection.location_data.relative_bounding_box

            # Get confidence score for smile detection
            if detection.score[0] > 9:  # Assuming higher confidence correlates with a smile
                if current_time - last_smile_time > smile_cooldown:
                    send_command("happy_hello")
                    last_smile_time = current_time

    # Motion detection
    if last_frame is not None:
        # Calculate absolute difference between the current and previous frame
        frame_diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), last_frame)
        motion_level = np.sum(frame_diff)

        if motion_level > 10:  # Threshold for motion detection
            motion_detected = True
            last_person_time = current_time  # Reset the motion timer
        else:
            motion_detected = False

    # Store the current frame for the next comparison
    last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If no motion detected for 5 seconds, send "bye" command
    if  motion_detected and (current_time - last_person_time > motion_timeout):
        send_command("bye")
        last_person_time = current_time  # Reset to avoid sending multiple "bye" messages

    # Display the frame
    cv2.imshow('Humanoid Face Interaction', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
arduino.close()
