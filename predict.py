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

# 0. Parameters
epochs = 15
filename = "full_test_simple_cnn.txt"

# 1. Load images
X, y, class_names = utilities.load_images()

X_an_av, y_an_av, class_names_an_av = utilities.select_classes(X, y, [0,1], class_names)
X_an_in, y_an_in, class_names_an_in = utilities.select_classes(X, y, [0,2], class_names)
X_av_in, y_av_in, class_names_av_in = utilities.select_classes(X, y, [1,2], class_names)


# 2. Load model
model_name = "simple_cnn"
model = Model(model_name=model_name, epochs=epochs, n_classes=3)
model_an_av = Model(model_name=model_name, epochs=epochs, n_classes=2)
model_an_in = Model(model_name=model_name, epochs=epochs, n_classes=2)
model_av_in = Model(model_name=model_name, epochs=epochs, n_classes=2)

# 3. train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train_an_av, X_test_an_av, y_train_an_av, y_test_an_av = train_test_split(X_an_av, y_an_av, test_size=0.1, random_state=42)
X_train_an_in, X_test_an_in, y_train_an_in, y_test_an_in = train_test_split(X_an_in, y_an_in, test_size=0.1, random_state=42)
X_train_av_in, X_test_av_in, y_train_av_in, y_test_av_in = train_test_split(X_av_in, y_av_in, test_size=0.1, random_state=42)


acc_list, mean_acc, std_acc, loss_list, mean_loss, std_loss, best_model = model.skf_fit(X_train, y_train, X_test, y_test, save_model=True, verbose=True)
acc_list_an_av, mean_acc_an_av, std_acc_an_av, loss_list_an_av, mean_loss_an_av, std_loss_an_av, best_model_an_av = model_an_av.skf_fit(X_train_an_av, 
																					y_train_an_av, X_test_an_av, y_test_an_av, save_model=True, verbose=True)
acc_list_an_in, mean_acc_an_in, std_acc_an_in, loss_list_an_in, mean_loss_an_in, std_loss_an_in, best_model_an_in = model_an_in.skf_fit(X_train_an_in, 
																					y_train_an_in, X_test_an_in, y_test_an_in, save_model=True, verbose=True)
acc_list_av_in, mean_acc_av_in, std_acc_av_in, loss_list_av_in, mean_loss_av_in, std_loss_av_in, best_model_av_in = model_av_in.skf_fit(X_train_av_in, 
																					y_train_av_in, X_test_av_in, y_test_av_in, save_model=True, verbose=True)


# 4. Write results

utilities.write_results(filename, class_names, acc_list, loss_list, model_name)
utilities.write_results(filename, class_names_an_av, acc_list_an_av, loss_list_an_av, model_name)
utilities.write_results(filename, class_names_an_in, acc_list_an_in, loss_list_an_in, model_name)
utilities.write_results(filename, class_names_av_in, acc_list_av_in, loss_list_av_in, model_name)
