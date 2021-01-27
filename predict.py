import PIL
from PIL import Image

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from skimage import io
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import keras
import pathlib
import os
import glob
import copy
import math


from model import Model
import utilities

# 1. Load images
X, y, class_names = utilities.load_images()

# 2. Load model
model = Model(model_name="simple_cnn", epochs=2)

# 3. train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model.skf_fit(X_train, y_train, X_test, y_test, save_model=True, verbose=True)