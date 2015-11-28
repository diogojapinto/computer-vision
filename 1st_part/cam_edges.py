import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    cv2.imshow('cam', cv2.Canny(frame, 50, 150))

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('frame.jpg', frame)

cap.release()
cv2.destroyAllWindows()
