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
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.8)
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

            if abs(wrist.y - middle_finger_tip.y) > 0.2:
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
            if detection.score[0] > 1:  # Assuming higher confidence correlates with a smile
                if current_time - last_smile_time > smile_cooldown:
                    send_command("happy_hello")
                    last_smile_time = current_time

    # Heart sign detection - when both palms are close together
    if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) == 2:
        hand1_landmarks = hand_results.multi_hand_landmarks[0]
        hand2_landmarks = hand_results.multi_hand_landmarks[1]

        # Get the position of both palms (wrist and index tip)
        palm1 = hand1_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        palm2 = hand2_landmarks.landmark[mp_hands.HandLandmark.WRIST]

        # Calculate the Euclidean distance between both palms
        distance = np.linalg.norm(np.array([palm1.x, palm1.y]) - np.array([palm2.x, palm2.y]))

        # If palms are very close, assume heart sign
        if distance < 0.6:  # Adjust this threshold as needed
            send_command("love")
            time.sleep(10)
    # No light or dark detection (brightness-based)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    average_brightness = np.mean(gray_frame)

    # If average brightness is too low (dark), send "bye"
    if average_brightness < 50:  # You can adjust the threshold for darkness
        send_command("bye")
        time.sleep(10)
    # Display the frame
    cv2.imshow('Humanoid Face Interaction', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
arduino.close()
