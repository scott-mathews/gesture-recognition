'''
File for loading localization dataset
and processing into usable form
'''

import json
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from numba import jit
import pickle
import time
from localize import Localizer

locer = Localizer()


class Localization:
    '''
    Object for interacting with localization dataset
    '''

    def __init__(self, filename='masks.p',  verbose=True):
        self.filename = filename
        self.verbose = verbose
    
        self.print('Loading Masks...')
        with open(self.filename, 'rb') as f:
            self.masks = pickle.load(f)
        self.print('Done Loading Masks')

    def get_dataset(self, block_size=100, num_examples=999999999):
        '''
        Create a training dataset from the main dataset,
        by partitioning images into subimage squares of block_size
        until you have num_examples of those.
        '''
        
        # tracks the data points we have
        # List of (numpy matrix, label) pairs
        # where label = 0 is negative, label=1 is positive
        examples = []

        self.print('Creating training examples...')
        while len(examples) < num_examples and len(self.masks) > 0:
            next_image = self.masks.pop()

            block_indices = locer.get_block_indices(next_image[0].shape, block_size)

            for block in block_indices:
                view = next_image[0][block[0][0]:block[1][0], block[0][1]:block[1][1]]
                label_view = next_image[1][block[0][0]:block[1][0], block[0][1]:block[1][1]]
                num_pixels = view.shape[0] * view.shape[1]
                num_hand_pixels = np.sum(label_view)
                pixel_ratio = num_hand_pixels / num_pixels
                #  if pixel_ratio > 0.1:
                    #  cv2.imshow('frame', label_view)
                    #  cv2.waitKey(0)
                label = 1 if pixel_ratio > 0.1 else 0
                examples.append((label, view))

        self.print('We have ' + str(len(examples)) + ' images.')
        self.print(str(len([lab for lab in examples if lab[0] == 1])) + ' positive examples.')

        return examples

    def save_csv(self, data, filename='localization.csv', size=100):
        '''
        resize each image in data to size,
        then flatten them and save them
        in a csv file
        '''

        self.print('Resizing...')

        resized_data = []

        for example in data:
            small = cv2.resize(example[1], (size, size))
            flat = small.flatten()
            resized_data.append( np.insert(flat, 0, example[0]) )

        self.print('Saving...')

        #  self.print(resized_data[0])

        with open('localization.csv', 'wb') as f:
            np.savetxt(f, resized_data, delimiter=',', fmt='%i')
            
        self.print('Saved to ' + filename)
            

    def print(self, text):
        '''
        Print only if verbose
        '''
        if self.verbose:
            print(text)


def get_filenames(path='.', suffix='.mat', with_path=True):
    '''
    this function modified from:
    https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    '''
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlysuffix = [f for f in onlyfiles if suffix in f]

    if with_path:
        onlysuffix = [(f, path+f) for f in onlysuffix]

    return onlysuffix

def load_image(filename):
    img = cv2.imread(filename, 0)
    assert img is not None

    return img

def mask_image(img, boxes):
    '''
    return a numpy array the same size as the image, which contains 0 for non hand pixels and 1 for hand pixels

    args:
        img (np.matrix): a matrix containing pixels values for the image
        boxes (list): a list of boxes within the given image
    '''

    mask = np.zeros(shape=img.shape)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):

            in_a_box = False

            # convert boxes 
            for box in boxes:
                box = (box['pt1'], box['pt2'], box['pt3'], box['pt4'])

                if ray_tracing(x,y,box):
                    in_a_box = True
                    break

            if in_a_box:
                mask[x][y] = 1.0

    return mask


# THE FOLLOWING FUNCTION FROM
# https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
@jit(nopython=True)
def ray_tracing(x,y,poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def make_localization():
    '''
    Used this to make the localization data
    '''
    directory = '../data/localization/hand_dataset/training_dataset/training_data/'
    img_directory  = directory + 'images/'

    annotations = json.load(open(directory+'annotations_training.json'))
    filenames = get_filenames(img_directory, suffix='.jpg')

    data = []
    #  images = {}
    ctr = 0
    for filename in filenames:
        start = time.time()
        image = load_image(filename[1])
        #  images[filename[0]] = image

        # find matching annotation
        annot_name = str.join('', filename[0].split('.')[0:-1]) + '.mat'
        annot = annotations[annot_name]
        mask = mask_image(image, annot)

        data.append((image, mask))
        end = time.time()
        ctr += 1
        print('Image #' + str(ctr) + ' (out of 4096) done in ' + str(end - start) + 's.')

    pickle.dump( data, open('masks.p', 'wb') )

if __name__ == '__main__':
    loc = Localization(filename = '../../../../Desktop/masks.p')
    dataset = loc.get_dataset()
    loc.save_csv(dataset)
