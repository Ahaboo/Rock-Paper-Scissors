import cv2
import mediapipe as mp

# Function to check the status of a finger
def is_finger_raised(hands_module, hand_landmarks, finger_name):
    finger_id_map = {'INDEX': 8, 'MIDDLE': 12, 'RING': 16, 'PINKY': 20}

    finger_tip_y = hand_landmarks.landmark[finger_id_map[finger_name]].y
    finger_dip_y = hand_landmarks.landmark[finger_id_map[finger_name] - 1].y
    finger_mcp_y = hand_landmarks.landmark[finger_id_map[finger_name] - 2].y

    return finger_tip_y < finger_dip_y and finger_dip_y < finger_mcp_y

# Function to check if the thumb is raised
def is_thumb_raised(hands_module, hand_landmarks):
    thumb_tip_x = hand_landmarks.landmark[hands_module.HandLandmark.THUMB_TIP].x
    thumb_mcp_x = hand_landmarks.landmark[hands_module.HandLandmark.THUMB_TIP - 2].x
    thumb_ip_x = hand_landmarks.landmark[hands_module.HandLandmark.THUMB_TIP - 1].x

    return thumb_tip_x > thumb_ip_x > thumb_mcp_x

# Function to start capturing video and recognize gestures
def start_video():
    drawingModule = mp.solutions.drawing_utils
    hands_module = mp.solutions.hands

    # Open the webcam
    cap = cv2.VideoCapture(0)

    with hands_module.Hands(static_image_mode=False, min_detection_confidence=0.7,
                            min_tracking_confidence=0.4, max_num_hands=2) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            recognized_gesture = "UNKNOWN"
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on the frame
                    drawingModule.draw_landmarks(frame, hand_landmarks, hands_module.HAND_CONNECTIONS)

                    current_state = ""
                    thumb_raised = is_thumb_raised(hands_module, hand_landmarks)
                    current_state += "1" if thumb_raised else "0"

                    index_raised = is_finger_raised(hands_module, hand_landmarks, 'INDEX')
                    current_state += "1" if index_raised else "0"

                    middle_raised = is_finger_raised(hands_module, hand_landmarks, 'MIDDLE')
                    current_state += "1" if middle_raised else "0"

                    ring_raised = is_finger_raised(hands_module, hand_landmarks, 'RING')
                    current_state += "1" if ring_raised else "0"

                    pinky_raised = is_finger_raised(hands_module, hand_landmarks, 'PINKY')
                    current_state += "1" if pinky_raised else "0"

                    if current_state == "00000":
                        recognized_gesture = "Rock"
                    elif current_state == "11111":
                        recognized_gesture = "Paper"
                    elif current_state == "01100":
                        recognized_gesture = "Scissors"
                    else:
                        recognized_gesture = "UNKNOWN"

                    # Display the recognized gesture on the frame
                    cv2.putText(frame, "Gesture: " + recognized_gesture, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            cv2.imshow('Rock Paper Scissors!', frame)

            # Exit the loop by pressing 'e' key
            if cv2.waitKey(1) & 0xFF == ord('e'):
                break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    start_video()