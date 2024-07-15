import cv2

def capture_board_image():
    cap = cv2.VideoCapture(0)  # Open the default camera

    while True:
        ret, frame = cap.read()
        cv2.imshow('Press Space to capture', frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()

    return frame

board_image = capture_board_image()
cv2.imwrite("captured_board.png", board_image)  # Save the captured image
