import cv2
import mediapipe as mp
import socket
import pickle

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

def start_server():
    drawingModule = mp.solutions.drawing_utils
    hands_module = mp.solutions.hands

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 4915))  # Use an appropriate IP and port
    server_socket.listen(1)

    print("Waiting for a client to connect...")
    client_socket, client_address = server_socket.accept()
    print(f"Connection established with {client_address}")

    cap_player1 = cv2.VideoCapture(0)  # Initialize player 1's webcam capture

    mp_hands = mp.solutions.hands

    with mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7,
                        min_tracking_confidence=0.4, max_num_hands=2) as hands:
        while True:
            ret_player1, frame_player1 = cap_player1.read()
            if not ret_player1:
                continue

            results = hands.process(cv2.cvtColor(frame_player1, cv2.COLOR_BGR2RGB))

            recognized_gesture_player1 = "UNKNOWN"
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    thumb_raised = is_thumb_raised(mp_hands, hand_landmarks)
                    index_raised = is_finger_raised(mp_hands, hand_landmarks, 'INDEX')
                    middle_raised = is_finger_raised(mp_hands, hand_landmarks, 'MIDDLE')
                    ring_raised = is_finger_raised(mp_hands, hand_landmarks, 'RING')
                    pinky_raised = is_finger_raised(mp_hands, hand_landmarks, 'PINKY')

                    if thumb_raised and index_raised and middle_raised and ring_raised and pinky_raised:
                        recognized_gesture_player1 = "Paper"
                    else:
                        recognized_gesture_player1 = "UNKNOWN"

            # Send player 1's gesture to player 2
            client_socket.send(pickle.dumps(recognized_gesture_player1))

            # Display player 1's webcam feed and gesture locally
            cv2.putText(frame_player1, f"Player 1 Gesture: {recognized_gesture_player1}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Player 1', frame_player1)

            if cv2.waitKey(1) & 0xFF == ord('e'):
                break

    cap_player1.release()
    cv2.destroyAllWindows()
    client_socket.close()

if __name__ == "__main__":
    start_server()