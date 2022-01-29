from djvu_utils.djvu2pngs import book2pngs
from crop_letters.crop_letters import contour_letters_cut_28x28
from letter_classifier.network import TensorFlowNetwork
from letter_classifier.stupid_trees import StupidTrees

#from tqdm import tqdm

import cv2
import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import shutil
import random

from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile


def download_and_unzip(url, extract_to):
	http_response = urlopen(url)
	zipfile = ZipFile(BytesIO(http_response.read()))
	zipfile.extractall(path=extract_to)


def read_page(path):
	img = cv2.imread(path)
	'''coeff = 1 # 5000/img.shape[0]
	if coeff < 1:
		img = cv2.resize(img, (int(img.shape[1]*coeff), int(img.shape[0]*coeff)), cv2.INTER_AREA)
	else:
		img = cv2.resize(img, (int(img.shape[1]*coeff), int(img.shape[0]*coeff)), cv2.INTER_LINEAR)'''

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return gray


def main():
	#dataset_path = "/home/alex/Proga/Project/good_dataset_books/"
	dataset_path = "/home/alex/Proga/Project/second_dataset"
	network = TensorFlowNetwork(35, dataset_path, 0.15, 5)
	#network = StupidTrees(dataset_path, 0.2)
	network.evaluate()

	book_path = "/home/alex/Proga/Project/books/"
	'''good_links = pd.read_csv("/home/alex/Proga/Project/djvu_links.csv")
	ids = random.sample(range(len(good_links)), 1)
	print(ids)

	for i, link in enumerate(good_links["djvu_link"][ids]):
		if os.path.exists(book_path):
			shutil.rmtree(book_path)

		download_and_unzip(link, book_path)
		for f in os.listdir(book_path):
			if f.endswith(".djvu"):
				book = os.path.join(book_path, f)
				print(book)
				book2pngs(book, book_path, range(14, 15))'''

	for num, f in enumerate(os.listdir(book_path)):
		if f.endswith(".png"):
			page = os.path.join(book_path, f)
			result = os.path.join(book_path, "results/" + str(num) + ".txt")
			result_file = open(result, "w")

			print(page)

			page_img = read_page(page)

			words = contour_letters_cut_28x28(page_img)
			h = words.shape[0]
			w = np.max([words[i].shape[0] for i in range(h)])

			# plt.figure(figsize=(20, 20))
			for i, word in enumerate(words):
				'''if i > 15:
					continue'''
				for j in range(word.shape[0]):
					letter = np.array(word[j], dtype=np.float64)
					#letter = 1 - letter/255
					prediction = np.argmax(network.predict([1 - letter/255]))
					predicted = network.get_letter(prediction)
					#print("word:", i, "letter:", predicted)
					#print(prediction)
					folder_name = os.path.join(book_path, "preds/" + str(num) + "/")
					if not os.path.exists(folder_name):
						os.mkdir(folder_name)
					path = os.path.join(folder_name, str(i) + "_" + str(j) + "=" + predicted + ".png")
					result_file.write(predicted)
					cv2.imwrite(path, letter)
					#print(network.get_letter(np.argmax(network.predict([letter]))))
					# plt.subplot(h, w, i*w + j+1)
					# plt.title(network.predict([letter])[0])
					# plt.imshow(np.array(letter, dtype='float'), cmap='gray')
				result_file.write(" ")
				#print("---------------------")
			# plt.show()


if __name__ == '__main__':
	main()
