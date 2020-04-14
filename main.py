import cv2
import time
import numpy as np
import pyautogui
import keras
from cnn import CNN

###############################
### Constants
##############################
## Hand Segmentation
BLUE_GLOVE_LOW = (100, 100, 35)
BLUE_GLOVE_HIGH = (120, 255, 255)
# skin_low =  (115, 44, 0)
# skin_high =  (179, 148, 149)

# input image dimensions
img_rows, img_cols = 64, 64

## Windows
CAMERA_CAPTURE = "camera"
HAND_CAPTURE = "hand"
TH = "threashold"

## Screen Information
FRAME_WIDTH = 600
FRAME_HEIGHT = 350

SCREEN_WIDTH = 3000
SCREEN_HEIGHT = 2000

FONT = cv2.FONT_HERSHEY_SIMPLEX
# Video Setup
video  = cv2.VideoCapture(0)
# video.set(cv2.CAP_PROP_FPS, 15)

cv2.namedWindow(CAMERA_CAPTURE)
# cv2.namedWindow(HAND_CAPTURE)
# cv2.namedWindow(TH)


USE_KERAS = True
if USE_KERAS:
    model = keras.models.load_model('cnn_keras.h5')
else:
    cnn = CNN()
    cnn.load_weights()

LABELS = ["fist", "one", "two", "three", "four", "five"]

def camera_to_screen_pos(y, x):
    """
    y := height
    x := width

    returns position of (y,x) in screen space
    """
    global FRAME_HEIGHT, FRAME_WIDTH, SCREEN_HEIGHT, SCREEN_WIDTH

    return int((y / FRAME_HEIGHT) * SCREEN_HEIGHT), int((x / FRAME_WIDTH) * SCREEN_WIDTH)

def main():
    hand_score = [np.zeros(6) for _ in range(3)]
    last_prediction = None
    in_a_row = 0
    while True:
        check, frame = video.read()
        original = frame.copy()

        # Hand Segmentation
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        # cv2.imshow('blur', frame)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow('hsv', frame_hsv)
        frame_th = cv2.inRange(frame_hsv, BLUE_GLOVE_LOW, BLUE_GLOVE_HIGH)
        # frame_th = cv2.inRange(frame_hsv, skin_low, skin_high)
        # cv2.imshow('th', frame_th)
        frame_erode = cv2.erode(frame_th, None, iterations=2)
        # cv2.imshow('erode', frame_erode)
        frame_dilate = cv2.dilate(frame_erode, None, iterations=2)
        # cv2.imshow('dilate', frame_dilate)

        contours, _ = cv2.findContours(frame_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hand_bw = np.zeros((128,128,3), np.uint8)
        found = False
        predictions = hand_score[0]
        area = 0
        if len(contours) > 0:
            hand = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(hand)

        if area >= 1000:
            found = True
            # hand = cv2.convexHull(hand)
            # cv2.drawContours(original, [hand], -1, (0, 255, 0), 3)
            m = cv2.moments(hand)
            hand_y = int(m['m01']/m['m00']) 
            hand_x = int(m['m10']/m['m00'])
            screen_y, screen_x = camera_to_screen_pos(hand_y - 100, hand_x)
            # make it so mouse movement is mirrord
            # COMMENT TO DISABLE MOUSE TRACKING
            # pyautogui.moveTo(3000 - screen_x, screen_y, 1)

            cv2.circle(original, (hand_x, hand_y), 3, (0, 255, 255), 2)

            x, y, w, h = cv2.boundingRect(hand)
            hand = original[y:y+h, x:x+w]
            hand = cv2.resize(hand, (64, 64))
            hand_bw = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)

            # send hand to cnn to detect gesture
            if USE_KERAS:
                hand_tmp = np.array([hand_bw])
                hand_input = hand_tmp.reshape(1, img_rows, img_cols, 1)
                hand_input = hand_input.astype('float32')
                hand_input /= 255
                hand_input -= 0.5
                probabilities = model.predict(np.expand_dims(hand_input[0], axis=0))
            else:
                label, probabilities = cnn.predict(hand_bw)

            # moving avg
            # hand_score.append(probabilities)
            # hand_score.pop(0)
            # avg = sum(hand_score) / len(hand_score)
            # label = LABELS[np.argmax(avg)]
            hand_score = probabilities
            # print(hand_score)
            label = LABELS[np.argmax(hand_score)]
            if last_prediction == label:
                in_a_row += 1
                if in_a_row == 10:
                    in_a_row = 0
                    # COMMENT TO IGNORE
                    '''
                    if label == "one":
                        pyautogui.press('volumeup')
                    elif label == "two":
                        pyautogui.press('volumedown')
                    # elif label == "three":
                        # pyautogui.press('win')
                    # elif label == "four":
                        # pyautogui.click(button='right)
                    elif label == "five":
                        pyautogui.press('space')
                    '''
            else:
                last_prediction = label
                in_a_row = 1

            predictions = [f"{p:0.2f}" for p in hand_score[0]]

            cv2.rectangle(original, (x,y), (x+w, y+h), (255, 0, 0), 3)
            original = cv2.flip(original, 1)
            cv2.putText(original, label, (400-x+w, y+h), FONT, 1, (0, 0, 255), 2, cv2.LINE_AA)


        if not found:
            original = cv2.flip(original, 1)
            cv2.imshow(CAMERA_CAPTURE, original)
        else:
            predictions *= 100
            output = f"fist: {predictions[0]}%, one: {predictions[1]}%, two: {predictions[2]}%"
            output2 = f"three: {predictions[3]}%, four: {predictions[4]}%, five: {predictions[5]}%"
            cv2.putText(original, output, (0, 25), FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(original, output2, (0, 55), FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(CAMERA_CAPTURE, original)

        # cv2.imshow(HAND_CAPTURE, hand_bw)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('m'):
            move = True
        elif key == ord('n'):
            move = False
        elif key == ord('l'):
            label = "1"
            print(f"saving {label}")
            cv2.imwrite(f"data/{label}/{label}_{int(time.time())}.jpg", hand_bw)


    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()