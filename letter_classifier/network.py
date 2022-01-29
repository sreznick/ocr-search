from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

import numpy as np
import os
import cv2
import random

class TensorFlowNetwork():

	def __init__(self, number_of_classes, dataset_path, sep=0.1, epochs=10):
		self.number2letter = dict()
		self.letter2number = dict()
		((X_train, y_train), (X_test, y_test)) = self.load_data(dataset_path, sep)

		self.X_train = np.asarray(X_train).astype(np.float64)
		self.X_test = np.asarray(X_test).astype(np.float64)
		self.y_train = np.asarray(y_train).astype(int)
		self.y_test = np.asarray(y_test).astype(int)

		print("Train:", self.X_train.shape[0])
		print("Test: ", self.X_test.shape[0])

		# defining model
		self.model=Sequential()
		# adding convolution layer
		self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
		# adding pooling layer
		self.model.add(MaxPool2D(2, 2))
		# adding convolution layer
		self.model.add(Conv2D(64, (3, 3), activation='relu'))
		# adding pooling layer
		self.model.add(MaxPool2D(2, 2))
		# adding fully connected layer
		self.model.add(Flatten())
		self.model.add(Dense(100, activation='relu'))
		# adding output layer
		self.model.add(Dense(number_of_classes, activation='softmax'))
		# compiling the model
		self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		# fitting the model
		self.model.fit(self.X_train, self.y_train, epochs=epochs)

	def evaluate(self):
		# evaluting the model
		self.model.evaluate(self.X_test, self.y_test)

	def predict(self, X_pred):
		X_pred = np.array(X_pred)
		X_pred = X_pred.reshape((X_pred.shape[0], X_pred.shape[1], X_pred.shape[2], 1))
		return self.model.predict(np.asarray(X_pred).astype(np.float64))

	def get_letter(self, number):
		return self.number2letter[number]

	def load_data(self, dataset_path, sep):
		#letter2number = dict()
		#number2letter = dict()
		data = []  # np.array([[]], dtype=np.ndarray)
		target = []  # np.array([[]], dtype=np.ndarray)

		ii = 0
		for letter_path in os.listdir(dataset_path):
			if letter_path not in "йцукенгшщзхъфывапролджэячсмитьбю?.,":
				# "йцукенгшщзхъфывапролджэячсмитьбю":
				continue
			path = dataset_path + "/" + letter_path + "/"
			target_letter = open(path + "target.txt", "r")
			letter = str(target_letter.read())
			self.number2letter[ii] = letter
			self.letter2number[letter] = ii

			files = os.listdir(path)
			for j, picture_path in enumerate(files):
				if picture_path.endswith(".png"):
					img = cv2.imread(path + "/" + picture_path)
					gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
					picture = 1 - gray/255
					data.append(np.array(picture, dtype=np.float32))  # .reshape(-1, 1)
					target.append(ii)  # vectorized_result(letter))
			ii += 1
		d = np.array(data, dtype=np.ndarray)
		print(len(data), d.shape)

		d = d.reshape((d.shape[0], d.shape[1], d.shape[2], 1))
		t = np.array(target, dtype=np.ndarray)

		# (X_, y_) = pickle.load(dataset_path)
		# print(X_.shape, y_.shape)
		d = np.array(d, dtype=np.ndarray)
		t = np.array(t, dtype=np.ndarray)
		num = int(t.shape[0] * (1-sep))
		rand_ids = random.sample(range(t.shape[0]), t.shape[0])
		train_ids = rand_ids[:num]
		test_ids = rand_ids[num:]
		# number_of_classes = np.max(t) + 1

		X_train = d[train_ids, :, :]
		y_train = t[train_ids]
		X_test = d[test_ids, :, :]
		y_test = t[test_ids]
		return (X_train, y_train), (X_test, y_test)


def vectorized_result(letter):
		e = np.zeros((number_of_classes, 1), dtype=np.float32)
		e[letter2number[letter]] = 1.0
		return e


#dataset_file_read = open("/home/alex/Proga/Project/second_dataset.pickle", "rb")
'''dataset_path = "/home/alex/Proga/Project/second_dataset"
number_of_classes = 91
network = TensorFlowNetwork(number_of_classes, dataset_path)
network.evaluate()'''

'''
number_of_classes = 10
#loading data
(X_train,y_train) , (X_test,y_test)=mnist.load_data()
#reshaping data
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))[:400, :, :, :]
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))[:400, :, :, :]
y_train = y_train[:400]
y_test = y_test[:400]
#normalizing the pixel values
X_train=X_train/255
X_test=X_test/255 
#checking the shape after reshaping
print(X_train.shape, X_train.dtype, y_train)#, X_train[:1, :, :, :])
print(X_test.shape, y_test.shape, y_test)'''