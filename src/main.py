from keras.models import load_model
import numpy as np
import time
import cv2
import heapq
from operator import itemgetter

from webcam import feed

model = load_model('../models/asl_model_alt_2.h5')

vid = feed(size=28)

# J is not in ASL alphabet, but inexplicably is included as a label
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

for bigframe, frame in vid:
    #  begin = time.time()

    #  font = cv2.FONT_HERSHEY_SIMPLEX
    # invert image - seems to make things worse
    frame_inv = cv2.bitwise_not(frame)
    # blur frame
    #  frame = cv2.GaussianBlur(frame, (3,3), 0)

    frame = frame.astype('float32')
    frame_inv = frame_inv.astype('float32')
    frame /= 255
    frame_inv /= 255

    frame = frame.reshape(1, 28, 28, 1)
    frame_inv = frame_inv.reshape(1, 28, 28, 1)
    pred = model.predict(frame)
    pred1 = model.predict(frame_inv)

    top5 = np.argpartition(pred[0], -2)[-2:]
    top51 = np.argpartition(pred1[0], -2)[-2:]

    print(str(list(map(lambda x: letters[x], top5))) + ' vs ' + str(list(map(lambda x: letters[x], top51))))

    funky = np.concatenate((pred[0],pred1[0]))
    chosen_index = np.argmax(funky)
    chosen_letter = letters[int(chosen_index%25)]
    text = chosen_letter

    #  end = time.time()
    #  print('took ' + str(end - begin) + ' seconds to load and evaluate frame')

    #  if sum(pred[0]) > 1:
        #  text = 'No letter'
    #  else:
    #  chosen_index = np.argmax(pred[0])
    #  chosen_letter = letters[int(chosen_index)]
    #  text = chosen_letter

    cv2.putText(bigframe, text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
    cv2.imshow('frame', bigframe)
    val = cv2.waitKey(1)
    if val & 0xFF == ord('q'):
        break


#  print(model.predict(next(vid).reshape(1, 28, 28, 1)))
#  print(next(vid).reshape(1, 28, 28, 1))
