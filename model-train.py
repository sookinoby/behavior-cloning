from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.utils.visualize_util import plot

import os
from scipy import misc
import cv2
import pickle
import csv
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import keras
import random

# for reproducibility of result
random.seed(42)

# height width, channels of final image that is provided as input for the dnn
height = 64;
width = 64;
channels =3;

#function to read a file
def read_image_from_file_name(image_name):
    image_name = image_name.replace("\\", "/")
    image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
    return image

#crop and resize the image
def make_roi(image):
    crop_img = image[60:140, 0:320, :]
    crop_img = cv2.resize(crop_img, (height, width),cv2.INTER_AREA)
    return crop_img



def read_csv_and_parse(image_steer,file_name,folder,path_split=True):
    counter = 0
    with open(folder+"/"+file_name) as f:
        reader = csv.reader(f)
        for row in reader:
            if (not path_split):
                temp = row[0].replace("C:\\Users\\sookinoby\\Dropbox\\ML\\udacity\\project3\\" + folder +"\\","")
                image_name = folder + "\\" + temp
            else:
                image_name = folder+"\\"+(row[0])
            angle = float(row[3])
            image_steer[image_name] = angle
            counter = counter + 1
        print(counter)
    return image_steer

# reads left,center,right images from the specified folder. Compensates adding .25 to
# steering angle for left and right camera image
def read_csv_and_parse_three_images(image_steer,file_name,folder,path_split=True):
    counter = 0
    with open(folder+"/"+file_name) as f:
        reader = csv.reader(f)
        for row in reader:
            image_name = folder+"/"+(row[0])
            angle = float(row[3])
            image_steer[image_name] = angle
            image_name = folder + "/" + (row[1]).strip()
            image_steer[image_name] = angle + .25

            image_name = folder + "/" + (row[2]).strip()
            image_steer[image_name] = angle - .25
            counter = counter + 1
        print(counter)
    return image_steer

image_steer = {}

#load only the udacity data
image_steer = read_csv_and_parse_three_images(image_steer,'driving_log.csv','data3')

#addtional data- I found this data in not that useful.
# image_steer = read_csv_and_parse(image_steer,'driving_log.csv','data4')
# image_steer = read_csv_and_parse(image_steer,'driving_log.csv','data5')
# image_steer = read_csv_and_parse(image_steer,'driving_log.csv','data6')
# image_steer = read_csv_and_parse(image_steer,'driving_log.csv','data7')
# image_steer = read_csv_and_parse(image_steer,'driving_log.csv','data8')
# image_steer = read_csv_and_parse(image_steer,'driving_log.csv','data9')
# image_steer = read_csv_and_parse(image_steer,'driving_log.csv','data10')


#constructs the training sets
def construct_train_test(image_steer):
    X_input = []
    Y_input = []
    counter = 0
    for key, value in image_steer.items():
        image = read_image_from_file_name(key)
        crop_img =image
        X_input.append(crop_img)
        Y_input.append(value)
    return np.array(X_input), np.array(Y_input)


X_input, Y_input = construct_train_test(image_steer)


# load the data for validation. I loaded these data already
X_valid = np.load("X_valid.dat")
Y_valid = np.load("Y_valid.dat")

# load the data for testing
X_test = np.load("X_test.dat")
Y_test = np.load("Y_test.dat")


# load the data for training. All the data is cropped to remove the horizon (skyline). They are resized into 32x32 image

# normalise the train data

X_train, y_train = X_input, Y_input;

# Flips the image and steering angle
# Adjusted the bais towards left.
def random_flip(image, steering):
    n = np.random.randint(0, 2)
    if n == 0 and steering == 0:
        image, steering = cv2.flip(image, 1), -steering
    return image, steering

#radnomly selects images from training set.
# It prefers to select( 4/5) images which has steering angle other than zero
# It eliminates bias towards zero steering angle
def select_random(X_train,y_train,bias):
    m = np.random.randint(0, len(y_train))
    n = np.random.randint(0, 80)
    image = X_train[m]
    steering = y_train[m];
    while ((steering > -bias and steering < bias) and n > 40):
        m = np.random.randint(0, len(y_train))
        image = X_train[m]
        steering = y_train[m]
    return image, steering

# randomly translates the images
# Generates additional data with translated image and steering angle.
def random_trans(image, steer, trans_range):
    rows,cols,_ = image.shape;
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

    return image_tr, steer_ang

# build a Dnn model
# The model accepts 64X64X3 as the input. The model is adpated from Nvidia.
# It has 6 convultion, and 4 fully connected layer.
# It uses exponential linear unit as the activation layer.
# In the final output is real valued number
# I have used a dropout layer. Dropout layer was used to prevent overfitting

model = Sequential()
model.add(Lambda(lambda x:  x/127.5 - 1.0, input_shape=(height,width,channels), name="noramlise"))
model.add(Convolution2D(24, 3, 3))
model.add(ELU())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(48, 4, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(1164))
model.add(ELU())
model.add(Dense(50))
model.add(ELU())
model.add(Dense(10))
model.add(Dense(1))

#to summarise and plot the model as a graph
#model.summary();
#plot(model, to_file='model.png')

# batch size of 64 was used since it easily fits in my GPU
batch_size = 64

# I trained for 10 Epoch. I noticed that after 10 epoch the training accuray doesnt decrease further
nb_epoch = 10;

# I used adam optimiser . Mse loss function is used
model.compile('adam', 'mse')


# The acutal function that generates agumented data
def generate_training_example(X_train, y_train,bias):
    image, steering =  select_random(X_train,y_train,bias)
    #image, steering = random_flip(image, steering)
    image, steering = random_trans(image, steering,20)
    return image, steering

# a python generator that select images at random with some random augmentation
def generate_train_batch(X_train, y_train, batch_size=32):
    batch_images = np.zeros((batch_size, height, width, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        bias = 0
        for i_batch in range(batch_size):
            bias = i_batch / (i_batch + 10)
            x, y = generate_training_example(X_train, y_train,bias)
            image = make_roi(x)
            batch_images[i_batch] = image
            batch_steering[i_batch] = y
        yield batch_images, batch_steering

#  a function to generate random training data of given batch size. This is passed to model.fit_generator function.
train_generator = generate_train_batch(X_train, y_train, batch_size)
#the actual training
model.fit_generator(train_generator,
                    samples_per_epoch=40000,
                    nb_epoch=nb_epoch,
                    validation_data=(X_valid, Y_valid)
                    )

#evaluation on test data. Not a useful measure.
result = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, sample_weight=None)

# save the model
model_filename = "model.json";
model_weights = "model.h5"
model_json = model.to_json()

# remove old models
try:
    os.remove(model_json)
    os.remove(model_weights)
except OSError:
    pass

#write new model
with open(model_filename, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(model_weights)
print("Saved model to disk")
