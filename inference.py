from tensorflow import keras
import numpy as np
import PIL
import random
import os
from matplotlib import pyplot as plt
from PIL import ImageOps

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

# def display_mask(i, val_preds):
#     """Quick utility to display a model's prediction."""
#     mask = np.argmax(val_preds[i], axis=-1)
#     mask = np.expand_dims(mask, axis=-1)
#     mask *= 255
#     img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
#     plt.imshow(img)
#     plt.savefig('inference.png')

path = '/mnt/roots/TUBE_all_photos/'
file = '/home/GTL/jmorata/special_problem/testing_set/testing_set.txt'
img_size = (768,448)

model = keras.models.load_model('./model/sigmoid')
file = open(file, 'r')
Lines = file.readlines()

for ind in range(20):
# ind = random.randint(0, len(Lines)-1)
    line = (path + Lines[ind]).strip()

    image = PIL.Image.open(line)
    resized_image = image.resize(
                    img_size,
                    resample=PIL.Image.BILINEAR
                )
    data = [np.array(resized_image)]
    mask = PIL.Image.open(os.path.join(path,line.replace('/M','/mask_M').replace('.jpg','.png')))
    resized_mask = mask.resize(
        img_size,
        resample=PIL.Image.BILINEAR
    )
    label = [np.array(resized_mask)]

    to_predict = Roots(1, img_size, data, label)
    prediction = model.predict(to_predict)

    # print(data[0].shape)    # displays (768, 448, 3)
    # print(label[0].shape)   # displays (768, 448)
    # print(prediction.shape) # displays (1, 768, 448, 2)
    _,h,w,_ = prediction.shape
    inferered_mask_smooth = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            inferered_mask_smooth[i,j] = 255*(prediction[0,i,j,0])
        # print(prediction[0,i,:,0])
    # inferered_mask_hf = np.zeros((h,w))
    # for i in range(h):
    #     for j in range(w):
    #         inferered_mask_hf[i,j] = 255*(prediction[0,i,j,0] > prediction[0,i,j,1])

    # print(prediction[0,550,400,0])
    # print(prediction[0,550,400,1])

    # print(line)
    plt.imshow(image)
    plt.savefig('results/sigmoid/test/{}_input.png'.format(ind))
    plt.imshow(mask)
    plt.savefig('results/sigmoid/test/{}_mask.png'.format(ind))
    plt.imshow(inferered_mask_smooth,cmap='gray', vmin=0, vmax=255)
    plt.savefig('results/sigmoid/test/{}_inference_smooth.png'.format(ind))
    # plt.imshow(inferered_mask_hf, cmap='gray', vmin=0, vmax=255)
    # plt.savefig('results/sigmoid/fine_tuned/test/{}_inference_hf.png'.format(ind))