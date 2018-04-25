''' Collect training data '''

import time
from webcam import feed
import cv2
import numpy as np

vid = feed(size=28)

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

print('What letter would you like to collect data for?')
letter_no = letters.index(input('Letter (uppercase): ').strip())

collected_data = []
for bigframe, frame in vid:

    cv2.imshow('frame', bigframe)
    
    val = cv2.waitKey(1)
    if val & 0xFF == ord('q'):
        break
    elif val & 0xFF == ord(' '):
        print('Frame captured!')
        inv_frame = cv2.bitwise_not(frame)
        frame = np.insert(frame, 0, letter_no)
        inv_frame = np.insert(inv_frame, 0, letter_no)
        collected_data.append(frame)
        collected_data.append(inv_frame)
        continue

print('Collected ' + str(len(collected_data)) + ' samples.')

f = open('../data/small/train_augmented.csv', 'ab')
np.savetxt(f, collected_data, delimiter=',', fmt='%i')

print('Appended samples to ../data/small/train_augmented.csv')
