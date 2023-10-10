import socket
import cv2
import pickle
import numpy as np

def start_client():
    # Create a socket for client-server communication
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Change IP and Port Here
    client_socket.connect(('localhost', 4915)) 

    # Create a window for Player 2
    cv2.namedWindow('Player 2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Player 2', 640, 480)

     # Receive data from the server (Player 1)
    while True:
        data = client_socket.recv(1024)
        if not data:
            print("Server closed the connection.")
            break
        
        # Deserialize the received data
        recognized_gesture_player1 = pickle.loads(data)

        # Create a blank frame with a white background
        frame_player2 = 255 * np.ones((480, 640, 3), dtype=np.uint8)

        # Display Player 1's gesture received from Player 1
        cv2.putText(frame_player2, f"Player 1 Gesture: {recognized_gesture_player1}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with text
        cv2.imshow('Player 2', frame_player2)

        # Press e on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    cv2.destroyAllWindows()
    client_socket.close()  # Close the client socket when done

if __name__ == "__main__":
    start_client()