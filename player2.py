import socket
import cv2
import pickle
import numpy as np

def start_client():
    # Create a socket for client-server communication
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Change IP and Port to match the "Player 1" configuration
    client_socket.connect(('localhost', 49156))

    while True:
        # Receive the recognized gesture from Player 1
        recognized_gesture_player1 = pickle.loads(client_socket.recv(1024))

        # Create a black screen to display Player 1's recognized gesture
        frame_player2 = np.zeros((480, 640, 3), dtype=np.uint8)

        # Display Player 1's recognized gesture on the black screen
        cv2.putText(frame_player2, f"Player 1 Gesture: {recognized_gesture_player1}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Player 2', frame_player2)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close the socket and destroy the window
    client_socket.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_client()