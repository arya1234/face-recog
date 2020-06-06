import os
import numpy as np
from os import listdir
from os.path import isfile, join
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.models import Model
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

def train_model(train_generator, validation_generator, nb_train_samples, nb_validation_samples):

    checkpoint = ModelCheckpoint("face.h5",
                                 monitor="val_loss",
                                 mode="min",
                                 save_best_only = True,
                                 verbose=1)

    earlystop = EarlyStopping(monitor = 'val_loss', 
                              min_delta = 0, 
                              patience = 3,
                              verbose = 1,
                              restore_best_weights = True)

    # we put our call backs into a callback list
    callbacks = [earlystop, checkpoint]

    # We use a very small learning rate of 0.001
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = RMSprop(lr = 0.001),
                  metrics = ['accuracy'])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch = nb_train_samples // batch_size,
        epochs = epochs,
        callbacks = callbacks,
        validation_data = validation_generator,
        validation_steps = nb_validation_samples // batch_size)

    return history.history['accuracy'][-1]


def add_layer(bottom_model, num_classes):

    top_model = bottom_model.output

    top_model = (Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))(top_model)
    top_model = (Conv2D(64, (3, 3), activation='relu'))(top_model)
    top_model = (MaxPooling2D(pool_size=(2, 2)))(top_model)
    top_model = (Dropout(0.25))(top_model)
    top_model = (Flatten())(top_model)
    top_model = (Dense(256, activation='relu'))(top_model)
    top_model = (Dropout(0.5))(top_model)

    top_model = (Dense(num_classes, activation='softmax'))(top_model)
    
    return top_model


#Loading The VGG16 Model.
def load_my_model():
    
    Vgg16 = VGG16(weights = 'imagenet', include_top = False, input_shape = (img_rows, img_cols, 3))

    for layer in Vgg16.layers:
        layer.trainable = False

    return Vgg16


def load_data():
    train_data_dir = 'C:/Users/Arya/Desktop/MLOPS/photos/99505_234911_bundle_archive/images/train'
    validation_data_dir = 'C:/Users/Arya/Desktop/MLOPS/photos/99505_234911_bundle_archive/images/validation'

    train_datagen = ImageDataGenerator(
          rescale=1./255,
          rotation_range=45,
          width_shift_range=0.3,
          height_shift_range=0.3,
          horizontal_flip=True,
          fill_mode='nearest')
    
    validation_datagen = ImageDataGenerator(rescale=1./255)

    print("Train Data:")
    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='categorical')

    print("Test Data:")
    validation_generator = validation_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='categorical')

    num_classes = len(np.unique(train_generator.classes))
    nb_train_samples = len(train_generator.classes)
    nb_validation_samples = len(validation_generator.classes)

    return train_generator, validation_generator, nb_train_samples, nb_validation_samples, num_classes





epochs = 1
batch_size = 16
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)


Vgg16 = load_my_model()

#Loading Our Dataset
train_gen, test_gen, nb_train_sample, nb_test_sample, num_classes = load_data()

#Adding our Layers to the Model.
FC_Head = add_layer(Vgg16, num_classes)
model = Model(inputs = Vgg16.input, outputs = FC_Head)


#Training Our Model.
acc = train_model(train_gen, test_gen, nb_train_sample, nb_test_sample)

#Printing the accuracy to acc.txt file
f = open("acc.txt", "w")
f.write(str(acc * 100))
f.close()