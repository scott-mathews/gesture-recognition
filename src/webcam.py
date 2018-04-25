''' Methods and Utilities for the Webcam '''

import numpy as np
import cv2


def feed(size=28):
    ''' generates a feed from webcam 0 using opencv, yields two images '''
    cap = cv2.VideoCapture(2)

    while True:

        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape

        newX, newY = size, size
 
        small = cv2.resize(gray, (int(newX), int(newY)))

        #  if cv2.waitKey(1) & 0xFF == ord('q'):
            #  break

        yield frame, small

        #  cv2.imshow('frame', small)

        #  if cv2.waitKey(1) & 0xFF == ord('q'):
            #  break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    generator = feed()

    for small in generator:
        cv2.imshow('frame', small)

