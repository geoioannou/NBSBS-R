import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Model
import numpy as np


def Dense_2Layer(input_shape, classes):
    inp = Input(shape=(input_shape,))
    x = Dense(100, activation="relu")(inp)
    x = Dense(30, activation="relu")(x)
    x = Dense(classes, activation="softmax")(x)

    model = Model(inp, x)
    return model

def Big_Dense(input_shape, classes):
    inp = Input(shape=(input_shape,))
    x = Dense(200, activation="relu")(inp)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation="softmax")(x)

    model = Model(inp, x)
    return model

def Small_Conv_net():
    inp = Input(shape=(28,28,1,))
    x = Conv2D(32, (3,3), activation="relu")(inp)
    x = Conv2D(64, (3,3), activation="relu")(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation="softmax")(x)

    model = Model(inp, x)
    return model


def Big_Conv_net():
    inp = Input(shape=(32,32,3,))
    x = Conv2D(32, (3,3), activation="relu")(inp)
    x = Conv2D(64, (3,3), activation="relu")(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(128, (3,3), activation="relu")(x)
    x = Conv2D(128, (3,3), activation="relu")(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation="softmax")(x)

    model = Model(inp, x)
    return model