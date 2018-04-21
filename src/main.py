from keras.models import load_model
import numpy as np
import time
import cv2

from webcam import feed

model = load_model('../models/asl_model_2.h5')

vid = feed(size=28)

# J is not in ASL alphabet, but inexplicably is included as a label
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

for bigframe, frame in vid:
    begin = time.time()

    cv2.imshow('frame', bigframe)
    # invert image - seems to make things worse
    frame = cv2.bitwise_not(frame)
    # blur frame
    #  frame = cv2.GaussianBlur(frame, (3,3), 0)
    frame = frame.reshape(1, 28, 28, 1)
    pred = model.predict(frame)

    end = time.time()
    print('took ' + str(end - begin) + ' seconds to load and evaluate frame')

    chosen_index = np.argmax(pred[0])

    chosen_letter = letters[int(chosen_index)]
    print(chosen_letter)


#  print(model.predict(next(vid).reshape(1, 28, 28, 1)))
#  print(next(vid).reshape(1, 28, 28, 1))
