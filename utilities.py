import glob
import numpy as np
from skimage import io


def load_images():
	class_names = ["anuros", "aves", "insetos"]

	X = []
	y = []
	for filename in glob.glob('C:/Users/caiof/Google Drive/Mestrado/IC/analise_exploratoria/spectrograms/anuros/*.jpg'): #assuming jpg
		img = io.imread(filename)
		X.append(img)
		y.append(0)

	for filename in glob.glob('C:/Users/caiof/Google Drive/Mestrado/IC/analise_exploratoria/spectrograms/aves/*.jpg'): #assuming jpg
		img = io.imread(filename)
		X.append(img)
		y.append(1)
		
	for filename in glob.glob('C:/Users/caiof/Google Drive/Mestrado/IC/analise_exploratoria/spectrograms/insetos/*.jpg'): #assuming jpg
		img = io.imread(filename)
		X.append(img)
		y.append(2)

	X = np.array(X)
	y = np.array(y)

	return X, y, class_names

def select_classes(X, y, classes, class_names):
	select_class_names = np.array(class_names)[np.array(classes)-1].tolist()

	indexes = np.array([])
	for class_id in classes:
		indexes = np.append(indexes, np.where(y[:] == class_id))

	y_selected = y[indexes.astype(int)] - (y[indexes.astype(int)] / len(classes)).astype(int)
	X_selected = X[indexes.astype(int)]

	return X_selected, y_selected, select_class_names

def write_results(filename, classes, acc_list, loss_list, model_name):
	f = open(filename, "a+")

	mean_acc = np.mean(acc_list)
	std_acc = np.std(acc_list)

	mean_loss = np.mean(loss_list)
	std_loss = np.std(loss_list)

	f.write("\n===========================================================================================================\n")
	f.write(f"Model: {model_name}\nClasses: {classes}")
	f.write("\n\n\t\t\t\t| Value\t\t\t\t\tstd_dev\n")
	f.write("-------------------------------------------------------------------\n")
	f.write(f"Mean Accuracy\t| {mean_acc}\t(+/- {std_acc})\n")
	f.write(f"Mean Loss\t\t| {mean_loss}\t(+/- {std_loss})\n")


