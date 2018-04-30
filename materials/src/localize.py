''' Perform localization of hands in images '''

from keras.models import load_model
import numpy as np
import cv2

class Localizer:
    ''' use to localize objects in images '''

    def __init__(self, model_file='../models/localization_model_2.h5'):
        self.model = load_model(model_file)
        self.indices = None

    def get_block_indices(self, shape, block_size):
        ''' get valid block indices for image shape and block size '''
        indices = []

        width = shape[1] // block_size
        height = shape[0] // block_size

        for h in range(height):
            for w in range(width):
                top_left_row = block_size * h
                top_left_col = block_size * w
                bot_right_row = top_left_row + block_size
                bot_right_col = top_left_col + block_size

                indices.append(((top_left_row, top_left_col), (bot_right_row, bot_right_col)))

        self.indices = indices

        return indices

    def extract_view(self, image, block):
        return image[block[0][0]:block[1][0], block[0][1]:block[1][1]]

    def recognized(self, raw_pred):
        raw_pred = raw_pred[0]
        #  print(raw_pred)
        return raw_pred[1] > 0.01
        #  return np.argmax(raw_pred) == 1


    def localize(self, image, block_size=100, model_input=100):
        '''
        Splits image into blocks of size,
        reports which blocks test positive for hands
        '''

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if not self.indices:
            block_indices = self.get_block_indices(image.shape, block_size)
        else:
            block_indices = self.indices

        blocks = []

        for block in block_indices:
            view = image[block[0][0]:block[1][0], block[0][1]:block[1][1]]
            view = cv2.resize(view, (model_input, model_input))
            view = view.astype('float32')
            view /= 255
            view = view.reshape(1,model_input,model_input,1)
            blocks.append(view)

        predictions = []

        for block in blocks:
            predictions.append(self.recognized(self.model.predict(block)))

        return block_indices, predictions

if __name__ == '__main__':
    loc = Localizer()

    print(loc.get_block_indices((500,600), 100))
