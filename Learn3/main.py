import pathlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tenserflow as tf
from tenserflow import keras
from tenserflow.keras import layers
from tenserflow.keras.models import Sequential

dataset_dir = pathlib.Path("Learn3/dataset/flower_photos")

image_count = len(list(dataset_dir.glob('*/*.jpg')))
print(f"Total images: {image_count}")