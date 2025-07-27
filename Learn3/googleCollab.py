import argparse
import logging
import os
from pathlib import Path
import tarfile
import urllib.request

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Sequential, callbacks


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def download_and_extract(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / url.split('/')[-1]
    if not archive_path.exists():
        logging.info(f"Downloading dataset from {url}")
        urllib.request.urlretrieve(url, archive_path)
    else:
        logging.info(f"Archive already exists at {archive_path}")

    extracted_folder = dest_dir / archive_path.stem
    if not extracted_folder.exists():
        logging.info(f"Extracting {archive_path}")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)
    else:
        logging.info(f"Dataset already extracted at {extracted_folder}")

    return extracted_folder


def prepare_datasets(
    data_dir: Path,
    img_size: tuple[int, int],
    batch_size: int,
    validation_split: float = 0.2,
    seed: int = 123,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )
    class_names = train_ds.class_names
    logging.info(f"Found classes: {class_names}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    return train_ds, val_ds, class_names


def build_model(img_size: tuple[int, int], num_classes: int) -> tf.keras.Model:
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.2),
    ], name="data_augmentation")

    model = Sequential([
        layers.Input(shape=(*img_size, 3)),
        data_augmentation,
        layers.Rescaling(1.0 / 255.0, name="rescaling"),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name="flower_classifier")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def plot_history(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plot_path = output_dir / "training_history.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved training history plot to {plot_path}")


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Train a flower classification model.")
    parser.add_argument(
        "--data_url",
        type=str,
        default="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
        help="URL for the flower photos dataset",
    )
    parser.add_argument(
        "--dest_dir",
        type=Path,
        default=Path("data"),
        help="Directory to download and extract data",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[180, 180],
        help="Image size (height width)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--output_model",
        type=Path,
        default=Path("flower_model.h5"),
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for plots and logs",
    )
    args = parser.parse_args()

    tf.debugging.set_log_device_placement(False)
    logging.info(f"GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    data_folder = download_and_extract(args.data_url, args.dest_dir)
    train_ds, val_ds, class_names = prepare_datasets(
        data_folder, tuple(args.img_size), args.batch_size
    )

    model = build_model(tuple(args.img_size), num_classes=len(class_names))
    model.summary(print_fn=logging.info)

    # Callbacks
    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=args.output_model,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    cb_earlystop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )
    cb_tensorboard = callbacks.TensorBoard(log_dir=args.output_dir / "logs")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[cb_checkpoint, cb_earlystop, cb_tensorboard],
    )

    plot_history(history, args.output_dir)
    logging.info("Training complete.")


if __name__ == "__main__":
    main()
