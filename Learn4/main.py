from tkinter.constants import X
from sympy import false
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.layers import Dense as layer
from keras.utils import to_categorical
from tensorflow.python.keras.metrics import accuracy

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28*28])
x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28*28])

y_train = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

model = keras.Sequential([
    layer(128, activation='relu', input_shape=(28*28,), name='layer1'),
    layer(10, activation='softmax', name='layer2')
])

model.summary()

# model.trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=5)

#print(model.evaluate(x_test, y_test_cat))
loss, accuracy = model.evaluate(x_test, y_test_cat)
print(f'loss: {loss:.5f}, accuracy: {accuracy:.5f}')