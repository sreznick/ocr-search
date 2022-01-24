from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

import numpy as np
import os
import cv2
import random
import pickle

#dataset_file_read = open("/media/alex/KINGSTON/second_dataset.pickle", "rb")
#number_of_classes = 91

class TensorFlowNetwork():
    
    def __init__(self, number_of_classes, dataset_path, sep=0.1, epochs=10):
        ((X_train, y_train), (X_test, y_test)) = load_data(dataset_path, sep)

        self.X_train = np.asarray(X_train).astype(np.float64)
        self.X_test = np.asarray(X_test).astype(np.float64)
        self.y_train = np.asarray(y_train).astype(int)
        self.y_test = np.asarray(y_test).astype(int)

        print("Train:", self.X_train.shape, self.y_train.shape)
        print("Test: ", self.X_test.shape, self.y_test.shape)
        
        #defining model
        self.model=Sequential()
        #adding convolution layer
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        #adding pooling layer
        self.model.add(MaxPool2D(2, 2))
        #adding fully connected layer
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))
        #adding output layer
        self.model.add(Dense(number_of_classes, activation='softmax'))
        #compiling the model
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #fitting the model
        self.model.fit(self.X_train, self.y_train, epochs=epochs)
        
    def evaluate(self):
        #evaluting the model
        self.model.evaluate(self.X_test, self.y_test)
        
    def predict(self, X_pred):
        self.model.predict(np.asarray(X_pred).astype(np.float64))
    
def load_data(dataset_path, sep):
    letter2number = dict()
    number2letter = dict()
    data = [] #np.array([[]], dtype=np.ndarray)
    target = [] #np.array([[]], dtype=np.ndarray)
    
    for i, letter_path in enumerate(os.listdir(dataset_path)):
        '''if (i > 10):
            break'''
        path = dataset_path + "/" + letter_path + "/"
        target_letter = open(path + "target.txt", "r") 
        letter = str(target_letter.read())
        number2letter[i] = letter
        letter2number[letter] = i

        files = os.listdir(path)
        for j, picture_path in enumerate(files):
            if picture_path.endswith(".png"):
                img = cv2.imread(path + "/" + picture_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                picture = 1 - gray/255
                data.append(np.array(picture, dtype=np.float32)) #.reshape(-1, 1)
                target.append(i)#vectorized_result(letter))
    d = np.array(data, dtype=np.ndarray)
    d = d.reshape((d.shape[0], d.shape[1], d.shape[2], 1))
    t = np.array(target, dtype=np.ndarray)
    
    #(X_, y_) = pickle.load(dataset_path)
    #print(X_.shape, y_.shape)
    d = np.array(d, dtype=np.ndarray)
    t = np.array(t, dtype=np.ndarray)
    num = int(t.shape[0] * (1-sep))
    rand_ids = random.sample(range(t.shape[0]), t.shape[0])
    train_ids = rand_ids[:num]
    test_ids = rand_ids[num:]
    #number_of_classes = np.max(t) + 1
    
    X_train = d[train_ids, :, :] 
    y_train = t[train_ids]
    X_test = d[test_ids, :, :]
    y_test = t[test_ids]
    return ((X_train, y_train), (X_test, y_test))

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