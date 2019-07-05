from keras.datasets import fashion_mnist
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization
import keras.utils
import os
from util import ensure_pre_dirs_exists

batch_size = 128
epochs = 30
num_classes = 10
img_rows, img_cols = 28, 28

my_model_path = os.path.join(os.path.dirname(__file__), "../model/my_model.h5")

ensure_pre_dirs_exists(my_model_path)


def my_model(inputs):
    outputs = Conv2D(32, kernel_size=(5, 5),
                     activation='relu', name="conv_1")(inputs)
    outputs = MaxPooling2D(pool_size=(2, 2), name="pool_1")(outputs)
    outputs = Conv2D(64, kernel_size=(5, 5),
                     activation='relu', name="conv_2")(outputs)
    outputs = MaxPooling2D(pool_size=(2, 2), name="pool_2")(outputs)
    outputs = Flatten(name="flatten")(outputs)
    outputs = Dense(1024, activation='relu', name="dense_1")(outputs)
    outputs = Dense(1024, activation='relu', name="dense_2")(outputs)
    outputs = BatchNormalization(name="bn_1")(outputs)
    outputs = Dense(num_classes, activation='softmax', name="dense_3")(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def train_my_model():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    inputs = Input(shape=input_shape)

    model = my_model(inputs)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    model.save_weights(my_model_path)


def get_my_model_output(tensor):
    inputs = Input(tensor=tensor)
    model = my_model(inputs)
    model.load_weights(my_model_path, by_name=True)
    for layer in model.layers:
        layer.trainable = False
    return model.output
