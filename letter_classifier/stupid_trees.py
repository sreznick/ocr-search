from sklearn.ensemble import RandomForestClassifier

import numpy as np
import os
import cv2
import random
from sklearn import metrics


class StupidTrees():

	def __init__(self, dataset_path, sep=0.1):
		self.number2letter = dict()
		self.letter2number = dict()
		((X_train, y_train), (X_test, y_test)) = self.load_data(dataset_path, sep)

		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test

		print("Train:", self.X_train.shape[0])
		print("Test: ", self.X_test.shape[0])

		self.model = RandomForestClassifier(n_estimators=70, n_jobs=4, max_depth=6)
		self.model.fit(self.X_train, self.y_train)

	def predict(self, X_pred):
		#X_pred = [pict.flatten() for pict in X_pred]
		#X_pred = X_pred.reshape((X_pred.shape[0], X_pred.shape[1], X_pred.shape[2], 1))
		return self.model.predict(X_pred)

	def evaluate(self):
		print(f'accuracy: {metrics.accuracy_score(self.predict(self.X_test), self.y_test)}')

	def get_letter(self, number):
		return self.number2letter[number]

	def load_data(self, dataset_path, sep):
		#letter2number = dict()
		#number2letter = dict()
		data = []  # np.array([[]], dtype=np.ndarray)
		target = []  # np.array([[]], dtype=np.ndarray)

		ii = 0
		for letter_path in os.listdir(dataset_path):
			if letter_path not in "укенгшзывапролджячсмитьб":
				#"йцукенгшщзхъфывапролджэячсмитьбю?.,":
				continue
			'''print(letter_path, ii)'''
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
					data.append(np.array(picture.flatten(), dtype=np.float32))  # .reshape(-1, 1)
					target.append(ii)  # vectorized_result(letter))
			ii += 1
		d = np.array(data, dtype=np.ndarray)
		#d = d.reshape((d.shape[0], d.shape[1], d.shape[2], 1))
		t = np.array(target, dtype=int)

		# (X_, y_) = pickle.load(dataset_path)
		# print(X_.shape, y_.shape)
		'''d = np.array(d, dtype=np.ndarray)
		t = np.array(t, dtype=np.ndarray)'''
		num = int(t.shape[0] * (1-sep))
		rand_ids = random.sample(range(t.shape[0]), t.shape[0])
		train_ids = rand_ids[:num]
		test_ids = rand_ids[num:]
		# number_of_classes = np.max(t) + 1

		X_train = d[train_ids, :]
		y_train = t[train_ids]
		X_test = d[test_ids, :]
		y_test = t[test_ids]
		return (X_train, y_train), (X_test, y_test)
