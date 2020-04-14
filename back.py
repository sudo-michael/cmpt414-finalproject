import cv2
import numpy as np

def background_sub(frame, last_frame):
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    last_frame = cv2.GaussianBlur(last_frame, (5, 5), 0)
    last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

    output = cv2.absdiff(frame, last_frame)
    _, output = cv2.threshold(output, 25, 255, cv2.THRESH_BINARY)
    return output

    
if __name__ == "__main__":
    c = 0
    capture = cv2.VideoCapture(0)
    ret, last_frame = capture.read()
    while True:
        c += 1
        ret, frame = capture.read()
        if frame is None:
            break
        
        output = background_sub(frame, last_frame) 
        last_frame = frame
        cv2.imshow('Frame', frame)
        cv2.imshow('FG Mask', output)
        
        if c == 100:
            cv2.imwrite('background_sub.jpg', output)
            exit()
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break