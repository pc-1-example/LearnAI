import pathlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#plt.style.use('dark_background') темная тема отстой

dataset_dir = pathlib.Path("Learn3/dataset/flower_photos")

batch_size = 32
img_width = 180
img_height = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

class_names = train_ds.class_names
print(f"Class names: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)
model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    # аугментация данных 
    # layers.experimental.preprocessing. устарел и больше не существует. Он был удален в TensorFlow 2.11
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),

    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# График точности
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Обучающая точность')
plt.plot(epochs_range, val_acc, label='Валидационная точность')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.title('Точность на обучении и валидации')
plt.legend(loc='lower right')
plt.grid(True)

# График потерь
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Обучающая потеря')
plt.plot(epochs_range, val_loss, label='Валидационная потеря')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.title('Потери на обучении и валидации')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()

# Сохранение модели
model.save_weights('flower_model')
print('Model saved to flower_model')