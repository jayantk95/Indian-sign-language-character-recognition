import cv2
import numpy as np
import random
import os
import csv

# PATH or folder name of dataset
PATH = 'data'

# please reduce TOTAL_IMAGES value to 800 or less if you are facing memory issues.
TOTAL_IMAGES = 700

# Total number of classes to be classified
N_CLASSES = 12

# clustering factor
CLUSTER_FACTOR = 5

START = (300, 75)
END = (600, 400)
##
IMG_SIZE = 128


def get_canny_edge(image):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert from RGB to HSV
    HSVImaage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Finding pixels with itensity of skin
    lowerBoundary = np.array([0, 40, 30], dtype="uint8")
    upperBoundary = np.array([43, 255, 254], dtype="uint8")
    skinMask = cv2.inRange(HSVImaage, lowerBoundary, upperBoundary)

    # blurring of gray scale using medianBlur
    skinMask = cv2.addWeighted(skinMask, 0.5, skinMask, 0.5, 0.0)
    skinMask = cv2.medianBlur(skinMask, 5)
    skin = cv2.bitwise_and(grayImage, grayImage, mask=skinMask)
    # cv2.imshow("masked2",skin)

    # . canny edge detection
    canny = cv2.Canny(skin, 60, 60)
    # plt.imshow(img2, cmap = 'gray')
    return canny, skin


def get_all_gestures():
    gestures = []
    for (dirpath, dirnames, filenames) in os.walk(PATH):
        dirnames.sort()
        for label in dirnames:
            # print(label)
            if not (label == '.DS_Store'):
                for (subdirpath, subdirnames, images) in os.walk(PATH + '/' + label + '/'):
                    random.shuffle(images)
                    # print(label)
                    imagePath = PATH + '/' + label + '/' + images[0]
                    # print(imagePath)
                    img = cv2.imread(imagePath)
                    img = cv2.resize(img, (int(IMG_SIZE * 3 / 4), int(IMG_SIZE * 3 / 4)))
                    img = cv2.putText(img, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                                      cv2.LINE_AA)
                    gestures.append(img)

    # print('length of gesatures {}'.format(len(gestures)))
    im_tile = concat_tile(gestures, (5,6))
    # im_tile = concat_tile(gestures, (2, 2))

    ''' im_tile = concat_tile([[gestures[0], gestures[1], gestures[2], gestures[3], gestures[4]],
                           [gestures[5], gestures[6], gestures[7], gestures[8], gestures[9]],
                           [gestures[10], gestures[11], gestures[12], gestures[13], gestures[14]],
                           [gestures[15], gestures[16], gestures[17], gestures[18], gestures[19]],
                           [gestures[20], gestures[21], gestures[22], gestures[23], gestures[24]],
                           [gestures[25], gestures[26], gestures[27], gestures[28], gestures[29]],
                           [gestures[30], gestures[31], gestures[32], gestures[33], gestures[34]]])'''
    return im_tile


def concat_tile(im_list_2d, size):
    count = 0
    all_imgs = []
    # print(size)
    for row in range(size[1]):
        imgs = []
        for col in range(size[0]):
            # print(count)
            imgs.append(im_list_2d[count])
            count += 1
        all_imgs.append(imgs)
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in all_imgs])


def get_labels():
    class_labels = []
    for (dirpath, dirnames, filenames) in os.walk(PATH):
        dirnames.sort()
        for label in dirnames:
            if not (label == '.DS_Store'):
                class_labels.append(label)
    # print(class_labels)
    return class_labels


# Loading dataset images and labels from csv files
def load_dataset(filename, n, h, w):
    data = []
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting each data row one by one
        for row in csvreader:
            data.append(row)

    x_data = np.zeros((n, h * w), dtype=float)
    y_data = []
    path = "/home/jayant/PycharmProjects/Indian sign language character recognition/"
    i = 0
    for row in data:
        current_image_path = path + row[0]
        y_data.append(int(row[1]))
        current_image = cv2.imread(current_image_path)
        canny_image = get_canny_edge(current_image)[0]
        # normalize and store the image
        x_data[i] = (np.asarray(canny_image).reshape(1, 128 * 128)) / 255
        i += 1
    return x_data, y_data
