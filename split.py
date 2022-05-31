import os

import numpy as np
import PIL

from PIL import Image


# Class that represents a single image with its path
class RootImage:

    def __init__(self, path, mask):
        self.path = path
        self.mask = mask


# Class that represents all the images classified by density
class AllImages:

    def __init__(self):
        self.less_than_ten = []
        self.ten_thirty = []
        self.thirty_fifty = []
        self.more_than_fifty = []

    def write_down(self):
        f = open("less_than_10.txt", "x")
        for line in self.less_than_ten:
            f.write(line)
        f.close()
        f = open("10_to_30.txt", "x")
        for line in self.ten_thirty:
            f.write(line)
        f.close()
        f = open("30_to_50.txt", "x")
        for line in self.thirty_fifty:
            f.write(line)
        f.close()
        f = open("more_than_50.txt", "x")
        for line in self.more_than_fifty:
            f.write(line)
        f.close()


# Data loader
def load_data_from_file(path, file):

    masks = []

    file = open(path + file, 'r')
    Lines = file.readlines()

    for line in Lines:
        full_path = (path + line).strip()
        full_path = full_path.replace('/M', '/mask_M').replace('.jpg', '.png')

        mask = PIL.Image.open(os.path.join(path, full_path))
        mask = np.array(mask) > 127
        masks.append(RootImage(line, mask))

    return masks


# Function that computes density of a given image
def calculate_density(mask):
    m, n = mask.shape
    size = m*n
    nb_true = 0

    for line in mask:
        for element in line:
            if element:
                nb_true += 1

    return nb_true / size


# Function that creates the AllImages object and fill it with all our images
def classify_masks(masks):

    classified_images = AllImages()

    for mask in masks:
        density = calculate_density(mask.mask)
        if density < 0.1:
            classified_images.less_than_ten.append(mask.path)
        elif density < 0.3:
            classified_images.ten_thirty.append(mask.path)
        elif density < 0.5:
            classified_images.thirty_fifty.append(mask.path)
        else:
            classified_images.more_than_fifty.append(mask.path)

    return classified_images


path = "/mnt/roots/TUBE_all_photos/"
file = "all_good_images.txt"
masks = load_data_from_file(path, file)
classified_images = classify_masks(masks)
classified_images.write_down()
