import socket
import cv2
import pickle

def start_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 4915))  # Replace 'server_ip' with the actual server IP and port

    cv2.namedWindow('Player 2', cv2.WINDOW_NORMAL)  # Create a window for player 2
    cv2.resizeWindow('Player 2', 640, 480)  # Set window size

    cap_player2 = cv2.VideoCapture(0)  # Initialize player 2's webcam capture

    while True:
        data = client_socket.recv(1024)
        if not data:
            print("Server closed the connection.")
            break

        recognized_gesture_player1 = pickle.loads(data)

        ret_player2, frame_player2 = cap_player2.read()

        # Display player 1's gesture received from player 1
        cv2.putText(frame_player2, f"Player 1 Gesture: {recognized_gesture_player1}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display both player 1 and player 2 webcam feeds side by side
        combined_frame = cv2.hconcat([frame_player2, frame_player1])

        # Display the combined frame
        cv2.imshow('Player 2', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    cap_player2.release()
    cv2.destroyAllWindows()
    client_socket.close()

if __name__ == "__main__":
    start_client()