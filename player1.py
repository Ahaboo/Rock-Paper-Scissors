import socket
import cv2
import pickle
import mediapipe as mp
import numpy as np

# Function to check if a finger is raised
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

    # Create a socket for server-client communication
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Change IP and Port Here
    server_socket.bind(('localhost', 49156))
    server_socket.listen(1)

    print("Waiting for a client to connect...")
    client_socket, client_address = server_socket.accept()
    print(f"Connection established with {client_address}")

    # Initialize webcam capture for Player 1
    cap_player1 = cv2.VideoCapture(0)

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
                    # Draw landmarks on Player 1's frame
                    drawingModule.draw_landmarks(frame_player1, hand_landmarks, hands_module.HAND_CONNECTIONS)

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
                        recognized_gesture_player1 = "Rock"
                    elif current_state == "11111":
                        recognized_gesture_player1 = "Paper"
                    elif current_state == "01100":
                        recognized_gesture_player1 = "Scissors"
                    else:
                        recognized_gesture_player1 = "UNKNOWN"

            # Send Player 1's gesture to Player 2 via the client socket
            client_socket.send(pickle.dumps(recognized_gesture_player1))

            # Display Player 1's webcam feed and recognized gesture locally
            cv2.putText(frame_player1, f"Player 1 Gesture: {recognized_gesture_player1}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Player 1', frame_player1)

            # Press e on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('e'):
                break

    # Release resources and close sockets
    cap_player1.release()
    cv2.destroyAllWindows()
    client_socket.close()

if __name__ == "__main__":
    start_server()