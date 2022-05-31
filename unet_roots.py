### First UNet Implementation to work on roots segmentation
import os
import configparser
import random
import math
import PIL
import numpy as np
import tensorflow as tf
# import albumentations as A

from os import path
from tensorflow import keras
# from IPython.display import Image, display
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt

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




# Get all the data from a folder, returns 2 np.array's lists : 1 with the data and 1 with the labels
def load_data(path, img_size):

    datas = []
    labels = []

    for f in os.listdir(path): 
        if 'mask_' in f:
            mask = PIL.Image.open(os.path.join(path,f))
            mask = mask.resize(
                img_size,
                resample=PIL.Image.BILINEAR
            )
            labels.append(np.array(mask))
            data_f = f.replace('mask','image').replace('.png','.jpg')
            image = PIL.Image.open(os.path.join(path,data_f))
            image = image.resize(
                img_size,
                resample=PIL.Image.BILINEAR
            )
            datas.append(np.array(image))

        # elif '_mask' in f:

        #     mask = PIL.Image.open(os.path.join(path,f)).convert('L')
        #     mask = mask.resize(
        #         img_size,
        #         resample=PIL.Image.BILINEAR
        #     )
        #     labels.append(mask)#np.array(mask))

        #     data_f = f.replace('_mask','').replace('.png','.jpg')
        #     mask2 = PIL.Image.open(os.path.join(path,data_f))
        #     mask2 = mask2.resize(
        #         img_size,
        #         resample=PIL.Image.BILINEAR
        #     )
        #     datas.append(np.array(mask2))

    # np_datas = np.array([datas[0]/255])
    # np_labels = np.array([labels[0]/255])
    # for i in range(1,len(labels)):
    #     np_datas = np.append(np_datas,[datas[i]/255],0)
    #     np_labels = np.append(np_labels,[labels[i]/255],0)

    return datas, labels

# Get all the data from a folder, returns 2 np.array's lists : 1 with the data and 1 with the labels
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

def augmentation(data, label):

    augmenter = A.Compose([
        A.VerticalFlip(p=0.3),
        A.HorizontalFlip(p=0.3),
        # A.RandomRotate90(p=0.3),
        A.RandomContrast(limit=0.2, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.RandomBrightness(limit=0.2, p=0.3)]
    )
    augmented = augmenter(image=data, mask=label)
    return augmented['image'], augmented['mask']

# Taken from https://github.com/keras-team/keras-io/blob/master/examples/vision/ipynb/oxford_pets_image_segmentation.ipynb

def get_model(img_size, num_classes, l2):
    inputs = tf.keras.Input(shape=img_size)#+ (3,))
    ### [First half of the network: downsampling inputs] ###

    # Entry blockdtype=float32
    x = layers.Conv2D(32, 3, strides=2, padding="same", kernel_regularizer=regularizers.l2(l2))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residuals

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same", kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same", kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="sigmoid", padding="same")(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    return model



# Train the network with the given parameters
def training(model, learning_rate, validation_split, nb_epochs, batch_size, metrics, data, label):

    optim = tf.keras.optimizers.Adam(lr = learning_rate)
    model.compile(optimizer=optim, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=metrics)
    print("V_split : {} // Nb_epochs : {} // Batch_size : {}".format(validation_split, nb_epochs, batch_size))
    history = model.fit(data, label, validation_split, nb_epochs, batch_size)

    return history

def display_metrics(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure()
    plt.plot(history.epoch, loss, 'r', label='Training loss')
    plt.plot(history.epoch, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig('curve_sigmoid_retrained_bis.png')

#### Main code ####

# Hyperparameters import
config = configparser.ConfigParser()
config.read("config_without_reg.ini")
path = config.get("myvars", "path")
filename = config.get("myvars", "filename")
batch_size = int(config.get("myvars", "batch_size"))
nb_epochs = int(config.get("myvars", "nb_epochs"))
validation_split = float(config.get("myvars", "validation_split"))
learning_rate = float(config.get("myvars", "learning_rate"))
metrics = config.get("myvars", "metrics")
nb_classes = int(config.get("myvars", "nb_classes"))
img_size = (int(config.get("myvars", "img_height")),int(config.get("myvars", "img_width")))
aug_factor = int(config.get("myvars", "augmentation_factor"))
l2 = float(config.get("myvars", "l2"))

# Data loading
datas, labels = load_data_from_file(path, filename, img_size)
# datas, labels = load_data(path, img_size)

# Splitting
val_samples = int(len(labels)*validation_split)
random.Random(1337).shuffle(datas)
random.Random(1337).shuffle(labels)
train_datas = datas[:-val_samples]
train_labels = labels[:-val_samples]
val_datas = datas[-val_samples:]
val_labels = labels[-val_samples:]

# Instantiating data Sequences for each split
# print(len(train_datas))
# augmented_datas = []
# augmented_labels = []
# for k in range(aug_factor):
#     for i in range(len(train_datas)):
#         aug_train_data, aug_train_label = augmentation(train_datas[i], train_labels[i])
#         augmented_datas.append(aug_train_data)
#         augmented_labels.append(aug_train_label)
# train_datas = train_datas + augmented_datas
# train_labels = train_labels + augmented_labels
# print(len(train_datas))
# # Image saving
# for i in range(len(train_datas)):
#     image = PIL.Image.fromarray(train_datas[i])
#     mask = PIL.Image.fromarray(train_labels[i])
#     image.save("augmented_dataset/image_{}.jpg".format(i))
#     mask.save("augmented_dataset/mask_{}.png".format(i))

# random.Random(1337).shuffle(train_datas)
# random.Random(1337).shuffle(train_labels)
train_gen = Roots(batch_size, img_size, train_datas, train_labels)
val_gen = Roots(batch_size, img_size, val_datas, val_labels)


# plt.imshow(train_gen.datas[0])
# plt.savefig("visualization/data.png")
# plt.imshow(train_gen.labels[0])
# plt.savefig("visualization/labels.png")
# Running the model

#print("Data's shape : {}".format(datas.shape))
input_shape = np.shape(datas[0])
#print("Input shape : {}".format(input_shape))
# model = get_model(input_shape, nb_classes, l2)
model = keras.models.load_model('./model/sigmoid')
print("model initialised")
model.summary()
model.compile(optimizer="rmsprop", loss="binary_crossentropy")
history = model.fit(train_gen, epochs=nb_epochs, validation_data=val_gen)#, callbacks=callbacks)
#history = training(model, learning_rate, validation_split, nb_epochs, batch_size, ["accuracy"], datas, labels)
model.save_weights('./checkpoints/sigmoid_retrained_bis')
model.save('./model/sigmoid_retrained_bis')
# imgs = model.predict(train_gen)
# print(type(imgs))
# print(imgs.shape)
# a,b,c,d = imgs.shape
# for i in range(a):
#     for j in range(b):
#         for k in range(c):
#             for l in range(d):
#                 if not(math.isnan(imgs[i,j,k,l])):
#                     print('A number at place {}'.format((i,j,k,l)))

# plt.imshow(imgs[0])
# plt.imshow(val_labels[0])
display_metrics(history)
