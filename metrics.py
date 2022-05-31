### Common code to get results and metrics on a given saved network
from itertools import count
import os
import argparse
import PIL
import time

import sklearn.metrics as m
import numpy as np

from tensorflow import keras

from matplotlib import pyplot as plt


# Class to represent the data (should not be useful for other applications)
class Roots(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, datas, labels):
        self.batch_size = batch_size
        self.img_size = (img_size[1],img_size[0])
        self.datas = datas
        self.labels = labels

    def __len__(self):
        return len(self.labels) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_datas = self.datas[i : i + self.batch_size]
        batch_labels = self.labels[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, data in enumerate(batch_datas):
            x[j] = data #np.transpose(data,(1,0,2))
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, label in enumerate(batch_labels):
            y[j] = np.expand_dims(label, 2) #np.trannspose(np.expand_dims(label, 2),(1,0,2))
            # Switch from grayscale to label
            y[j] = y[j] > 127
        return x, y

# Personnal data loader TO ADAPT
def load_data_from_file(img_path, file_path, img_size):

    datas = []
    labels = []

    file = open(file_path, 'r')
    Lines = file.readlines()

    for line in Lines:
            
            line = (img_path + line).strip()

            image = PIL.Image.open(line)
            image = image.resize(
                img_size,
                resample=PIL.Image.BILINEAR
            )
            datas.append(np.array(image))
            data_f = line.replace('/M','/mask_M').replace('.jpg','.png')
            mask = PIL.Image.open(os.path.join(img_path,data_f))
            mask = mask.resize(
                img_size,
                resample=PIL.Image.BILINEAR
            )
            labels.append(np.array(mask))

    return datas, labels

# Computes Accuracy, Precision and Recall
def get_metrics(y_true, y_pred):
    acc = m.accuracy_score(y_true, y_pred)
    prec = m.precision_score(y_true, y_pred, pos_label=1)
    rec = m.recall_score(y_true, y_pred, pos_label=1)
    return acc, prec, rec

# Plots the Precision-Recall curve
def display_PR_curve(precisions, recalls):
    plt.plot(recalls, precisions, linewidth=4, color="red")
    plt.xlabel("Recall", fontsize=12, fontweight='bold')
    plt.ylabel("Precision", fontsize=12, fontweight='bold')
    plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    plt.savefig("./metrics/PR_train.png")


## Main code
if __name__ == '__main__':

    previous_time = time.time()
    # Parsing
    parser = argparse.ArgumentParser(description='Generate metrics of a given model')
    parser.add_argument('--model', nargs='?', dest ='model_path', help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', dest ='testing_path', help='Path to the testingset file')
    args = parser.parse_args()

    # Other variables initialiazation
    starting_threshold = 0.2
    ending_threshold = 0.8
    nb_thresholds = 30
    batch_size = 8
    img_size = (768,448)
    img_path = '/mnt/roots/TUBE_all_photos/'

    # Loading data TO ADAPT
    datas, labels = load_data_from_file(img_path, args.testing_path, img_size)
    current_time = time.time()
    print("Loading done in {} seconds".format(current_time-previous_time))
    previous_time = current_time

    # Preprocessing data TO ADAPT
    to_predict = Roots(batch_size, img_size, datas, labels)
    vectorized_labels = []
    b,h,w = len(labels), len(labels[0]), len(labels[0][0])
    for k in range(b):
        for i in range(h):
            for j in range(w):
                vectorized_labels.append(labels[k][i][j] > 127)
    current_time = time.time()
    print("Preprocessing done in {} seconds".format(current_time-previous_time))
    previous_time = current_time

    # Loading the model
    model = keras.models.load_model(args.model_path)

    # Inference
    prediction = model.predict(to_predict)
    current_time = time.time()
    print("Prediction done in {} seconds".format(current_time-previous_time))
    previous_time = current_time

    # Post processing  TO ADAPT
    b,h,w,_ = prediction.shape
    infered_labels = []
    for threshold in np.linspace(starting_threshold,ending_threshold, nb_thresholds):
        threshold_labels = []
        for k in range(b):
            for i in range(h):
                for j in range(w):
                    threshold_labels.append(prediction[k,i,j,1] > threshold)
        infered_labels.append(threshold_labels)
    current_time = time.time()
    print("Postprocessing done in {} seconds".format(current_time-previous_time))
    previous_time = current_time

    # Metrics computation for 0.5 threshold
    acc, prec, rec = get_metrics(vectorized_labels, infered_labels[15])
    print("Accuracy for a treshold of 0.5 : {}".format(acc))
    print("Precision for a treshold of 0.5 : {}".format(prec))
    print("Recall for a treshold of 0.5 : {}".format(rec))

    # Curve drawing
    precs = []
    recs = []
    for labels in infered_labels:
        _, prec, rec = get_metrics(vectorized_labels, labels)
        precs.append(prec)
        recs.append(rec)
    display_PR_curve(precs,recs)