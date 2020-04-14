# code derived from sample keras code
# https://keras.io/examples/mnist_cnn/ 

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
import numpy as np
import glob
import cv2

train = True
if train:
    batch_size = 128
    num_classes = 6
    epochs = 10 

    # input image dimensions
    img_rows, img_cols = 64, 64

    labels = ["fist", "1", "2", "3", "4", "5"]
    images = []
    for idx, l in enumerate(labels):
        path = f"data/{l}/*"
        images.extend([(cv2.imread(file, cv2.IMREAD_GRAYSCALE), idx) for file in glob.glob(path)])

    split = int(len(images) * 0.75)
    np.random.shuffle(images)
    images = np.array(images)
    train_set = images[:split]
    test_set = images[split:]

    x_train = np.array([im[0] for im in train_set])
    y_train = np.array([im[1] for im in train_set])

    x_test = np.array([im[0] for im in test_set])
    y_test = np.array([im[1] for im in test_set])

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train -= 0.5
    x_test -= 0.5
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),
                    activation='relu',
                    input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu')) 
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(8, (3, 3), activation='relu')) 
    model.add(Dropout(0.03))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(6, activation='softmax'))

    '''
    # copy of my own cnn implementation
    model.add(Conv2D(8, kernel_size=(3, 3),
                    activation='linear',
                    input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    '''

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test)
    print('loss', score[0], 'acc', score[1])
    model.save('cnn_keras.h5')


if __name__ == "__main__":
    model = keras.models.load_model('cnn_keras.h5')
    model.summary()
    batch_size = 128
    num_classes = 6
    epochs = 10 

    # input image dimensions
    img_rows, img_cols = 64, 64

    labels = ["fist", "1", "2", "3", "4", "5"]
    images = []
    for idx, l in enumerate(labels):
        path = f"data/{l}/*"
        images.extend([(cv2.imread(file, cv2.IMREAD_GRAYSCALE), idx) for file in glob.glob(path)])

    split = int(len(images) * 0.75)
    np.random.shuffle(images)
    images = np.array(images)
    train_set = [images[0]]
    test_set = [images[0]]

    x_train = np.array([im[0] for im in train_set])
    y_train = np.array([im[1] for im in train_set])

    x_test = np.array([im[0] for im in test_set])
    y_test = np.array([im[1] for im in test_set])

    if K.image_data_format() == 'channels_first':
        print('c first')
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        print(x_train.shape)
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train -= 0.5
    x_test -= 0.5
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(np.argmax(model.predict(np.expand_dims(x_test[0], axis=0))))
    
