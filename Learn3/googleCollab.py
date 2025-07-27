import pathlib
import os
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Sequential

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
dataset_dir = "Learn3/dataset"

os.makedirs(dataset_dir, exist_ok=True)
archive_path = os.path.join(dataset_dir, "flower_photos.tgz")

urllib.request.urlretrieve(url, archive_path)

with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(path=dataset_dir)

dataset_dir = os.path.join(dataset_dir, "flower_photos")


batch_size = 64
img_size = (180, 180)

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
)

class_names = train_ds.class_names
print("Class names:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
])

num_classes = len(class_names)
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),

    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

epochs = 20
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

model.save("flower_model.h5")

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(epochs)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Обучающая точность")
plt.plot(epochs_range, val_acc, label="Валидационная точность")
plt.xlabel("Эпоха")
plt.ylabel("Точность")
plt.title("Точность на обучении и валидации")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Обучающая потеря")
plt.plot(epochs_range, val_loss, label="Валидационная потеря")
plt.xlabel("Эпоха")
plt.ylabel("Потери")
plt.title("Потери на обучении и валидации")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
