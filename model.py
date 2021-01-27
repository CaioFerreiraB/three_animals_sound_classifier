
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import keras

from skimage import io
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import seaborn as sn
import pandas as pd
import numpy as np

import pathlib
import os
import glob
import copy
import math


class Model(object):
	"""
	Atributes
		model_name: the name of the model to be  utilized. 
					- "simple_cnn"
		epochs: number of epochs to the trainning process. 
		n_classes: Nomber of classes to clasify			
	"""
	def __init__(self, model_name=None, epochs=10, n_classes=3):
		super(Model, self).__init__()

		self.classes = []
		self.learning_rate = 0.0000001
		self.epochs = epochs
		self.model = None
		self.logs_path = None
		self.model_name = None
		self.n_classes = n_classes

		if model_name is None:
			raise RuntimeError ('A model_name must be set to instantiate a model')
		else:
			self.model_name = model_name
			if model_name == 'simple_cnn' : self.model = self.simple_cnn(epochs=self.epochs, num_classes=n_classes)
			


	def simple_cnn(self, num_classes, learning_rate=0.0000001, epochs=None):

		model = tf.keras.Sequential([
			layers.experimental.preprocessing.Rescaling(1./255),

			layers.Conv2D(32, 3, activation='relu'),
			layers.MaxPool2D((3,3), strides=(2,2), padding='same'),
			layers.Flatten(),
			
			layers.Dense(128, activation='relu'),
			layers.Dropout(0.4),

			layers.Dense(num_classes)
		])

		model.compile(
			optimizer='adam',
			loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
			metrics=['accuracy']
		)
		
		return model

	"""
	Standard way to fit the model. Customs fit can be done with model.model.fit(...)
	"""					
	def standard_fit(self, X_train, y_train, X_val, y_val, epochs=None, model=None):
		n_epochs = None
		model_fit = None

		if epochs is None: n_epochs = self.epochs
		else: n_epochs = epochs

		if model is None: model_fit = self.model
		else: model_fit = model

		history = model_fit.fit(
			x=X_train, 
			y=y_train, 
			validation_data=(X_val, y_val),
			epochs=n_epochs)

		return history

	def skf_fit(self, X_train, y_train, X_test, y_test, n_folds=5, epochs=None, verbose=False, save_model=False):
		n_epochs = None

		if epochs is None: n_epochs = self.epochs
		else: n_epochs = epochs

		best_model = None
		model = None
		best_acc = -1
		best_loss = math.inf
		acc_list = []
		loss_list = []

		skf = StratifiedKFold(n_splits=n_folds)
		for train_index, test_index in skf.split(X_train, y_train):

			# 1. Create an instance of the model
			if self.model_name == 'simple_cnn' : model = self.simple_cnn(epochs=n_epochs, num_classes=self.n_classes)
			
			# 2. Fit the models
			history = self.standard_fit(X_train[train_index.tolist()], y_train[train_index.tolist()], X_train[test_index.tolist()], y_train[test_index.tolist()], model=model)

			# 3. Evaluate the fited model
			loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
			if verbose: print(f"Accuracy: {accuracy}\t|\tLoss:{loss}")

			if accuracy > best_acc:
				best_acc = accuracy
				best_model = model
				
			acc_list.append(history.history['accuracy'])
			loss_list.append(history.history['loss'])

		mean_acc = np.mean(acc_list)
		std_acc = np.std(acc_list)

		mean_loss = np.mean(loss_list)
		std_loss = np.std(loss_list)

		if verbose:
			print("\n\n\t\t| Value\t\t\tstd_dev")
			print("-------------------------------------------------------------------")
			print(f"Mean Accuracy\t| {mean_acc}\t(+/- {std_acc})")
			print(f"Mean Loss\t| {mean_loss}\t(+/- {std_loss})")

		self.model = best_model

		return acc_list, mean_acc, std_acc, loss_list, mean_loss, std_loss, best_model