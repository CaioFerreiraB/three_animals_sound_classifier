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