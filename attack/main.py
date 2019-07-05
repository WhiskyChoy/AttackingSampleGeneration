from keras.datasets import fashion_mnist
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Input, BatchNormalization
from keras import backend as K
import keras.utils
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from util import ensure_pre_dirs_exists

from my_model import get_my_model_output

attack_model_path = os.path.join(os.path.dirname(__file__), "../model/attack_model.h5")
test_image_path = os.path.join(os.path.dirname(__file__), "../test_image/")

c = 0.02
t = 0.25
num_classes = 10
img_rows, img_cols = 28, 28

ensure_pre_dirs_exists(attack_model_path, test_image_path)


def encrypt(img_true, label_true):
    return np.concatenate((img_true, label_true), axis=1)


def my_loss(y_true, y_pred):
    img_true, label_true, y_pred = decrypt(y_true, y_pred)
    label_pred = get_my_model_output(y_pred)
    return -K.sqrt(K.mean((label_true - label_pred) ** 2)) + c * (tf.image.ssim(img_true, y_pred, 255) + 1) / 2


def my_acc(y_true, y_pred):
    img_true, label_true, y_pred = decrypt(y_true, y_pred)
    a = K.argmax(label_true, axis=1)
    b = K.argmax(get_my_model_output(y_pred), axis=1)
    return K.mean(K.not_equal(a, b))


def my_ssim(y_true, y_pred):
    img_true, label_true, y_pred = decrypt(y_true, y_pred)
    return (tf.image.ssim(img_true, y_pred, 255) + 1) / 2


def my_score(y_true, y_pred):
    img_true, label_true, y_pred = decrypt(y_true, y_pred)
    a = K.argmax(label_true, axis=1)
    b = K.argmax(get_my_model_output(y_pred), axis=1)
    return K.mean(tf.cast(K.not_equal(a, b), tf.float32) * (tf.image.ssim(img_true, y_pred, 255) + 1) / 2)


def decrypt(y_true, y_pred):
    img_true, label_true = tf.split(
        y_true, [img_rows * img_cols, num_classes], 1)

    img_true = tf.reshape(img_true, (-1, img_rows, img_cols, 1))
    label_true = tf.reshape(label_true, (-1, num_classes))

    min_val = tf.reshape(tf.reduce_min(y_pred, axis=1), (-1, 1))
    max_val = tf.reshape(tf.reduce_max(y_pred, axis=1), (-1, 1))
    y_pred = ((y_pred - min_val) / (max_val - min_val) - 0.5) * 2 * t * 255
    y_pred = tf.reshape(y_pred, (-1, img_rows, img_cols, 1))
    y_pred = tf.clip_by_value(img_true + y_pred, 0, 255)

    return img_true, label_true, y_pred


def train_attack_model():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(-1, img_rows * img_cols)
    x_test = x_test.reshape(-1, img_rows * img_cols)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    y_train = encrypt(x_train, y_train)
    y_test = encrypt(x_test, y_test)

    x_train = x_train.reshape(-1, img_rows, img_cols, 1)
    x_test = x_test.reshape(-1, img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    inputs = Input(shape=input_shape)
    outputs = Flatten()(inputs)
    outputs = Dense(1024, activation='relu')(outputs)
    outputs = Dense(1024, activation='relu')(outputs)
    outputs = Dense(img_rows * img_cols, activation='relu'
                    )(outputs)
    outputs = BatchNormalization()(outputs)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss=my_loss,
                  optimizer=keras.optimizers.Adam(),
                  metrics=[my_acc, my_ssim, my_score])
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=20,
              verbose=1,
              validation_data=(x_test, y_test))

    model.save(attack_model_path)


def ai_test(images, shape):
    model = load_model(attack_model_path, custom_objects={
        "my_ssim": my_ssim, "my_loss": my_loss, "my_acc": my_acc, "my_score": my_score})
    mask = model.predict(images)
    mask = mask.reshape(np.append(-1, shape))
    generate_images = images + mask

    return generate_images


def train():
    # train_my_model()
    train_attack_model()


def test():
    (_, _), (x_test, y_test) = fashion_mnist.load_data()
    x_test = x_test[0:1000].reshape(-1, img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    generate_images = ai_test(x_test, input_shape)

    for i in range(10):
        Image.fromarray(x_test[i].reshape(img_rows, img_cols)).convert(
            'RGB').save(test_image_path + 'origin' + str(i) + '.jpg')
        Image.fromarray(generate_images[i].reshape(img_rows, img_cols)).convert(
            'RGB').save(test_image_path + 'generate' + str(i) + '.jpg')


# train()
test()
