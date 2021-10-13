from ctypes import resize
import os
import sys
import random
from time import time
import cv2
import pickle
from numpy.lib.type_check import imag
import tensorflow as tf
import pandas as pd
import numpy as np
import pprint
import matplotlib.pyplot as plot
from PySide2 import QtWidgets
from PySide2 import QtGui, QtCore
from PySide2.QtGui import *

from tensorflow.keras.utils import normalize
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Conv2D, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD, Adam

valid_exts = ('.png', '.jpeg')
current_dir = os.path.dirname(__file__)
categories = ( 'A', 'B', 'C', 'D', 'E', 'F', )

image_size = 50

alpha_training_data_dir = os.path.join(current_dir, 'alphabets')
alphabet_test_data_dir = os.path.join(current_dir, 'testing_data')
alpha_training_data_one = []
testing_data = []
features_set = []
labels_set = []
test_feature_set = []
trained_aplhabet_features_model_path = os.path.join(current_dir, 'features_set.pickle')
trained_aplhabet_label_model_path = os.path.join(current_dir, 'labels_set.pickle')
trained_alphabet_model_path = os.path.join(current_dir, 'alphabet_reader.model')

class Predictor(QtWidgets.QDialog):

    def __init__(self, parent=None):

        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle('Writer')
        self.setWindowFlags(self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)

        self.main_layout = QtWidgets.QVBoxLayout()
        input_horizontal_layout = QtWidgets.QHBoxLayout()
        self.input_label = QtWidgets.QLabel('Input Image :')
        input_horizontal_layout.addWidget(self.input_label)
        self.input_line_edit = QtWidgets.QLineEdit()
        self.input_line_edit.setReadOnly(True)
        input_horizontal_layout.addWidget(self.input_line_edit)
        self.file_browser = QtWidgets.QPushButton(' ... ',clicked=self.set_file_path)
        input_horizontal_layout.addWidget(self.file_browser)
        self.main_layout.addLayout(input_horizontal_layout)
        
        self.predict_button = QtWidgets.QPushButton(' Predict ', clicked=self.predict_clicked)
        self.main_layout.addWidget(self.predict_button)
        self.result_label = QtWidgets.QLabel(' Result ')
        self.result_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.result_label)
        self.result_window = QtWidgets.QLabel()
        self.result_window.setAlignment(Qt.AlignCenter)
        self.result_window.setFont(QtGui.QFont('Times', weight=QtGui.QFont.Bold))
        self.main_layout.addWidget(self.result_window)

        self.setLayout(self.main_layout)
        self.show()


    def set_file_path(self):
     
        browse_path = QtWidgets.QFileDialog.getOpenFileName(None, 'Select File')
        old_path = self.input_line_edit.text()
        self.input_line_edit.setText(str(browse_path[0]) if browse_path else old_path)
        file_path, file_path_ext = os.path.splitext(self.input_line_edit.text())
        if file_path_ext not in valid_exts:
            print ('The selected file is not a valid Image. Please select png or jpeg')
            self.input_line_edit.setText(old_path)
        
    
    @staticmethod
    def processing_user_input(input_data):

        bad_input_image = []
        input_data = input_data
        print (input_data)
        try:
            alphabet_data = cv2.imread(input_data, cv2.IMREAD_GRAYSCALE)
            resized_alphabet_data = cv2.resize(alphabet_data, (image_size, image_size))
            testing_data.append(resized_alphabet_data)
        except:
            bad_input_image.append(input_data)
        
        if bad_input_image:
            print('Following test image data is bad and cant be read /n{}'.format(bad_input_image))
        
        new_test_feature_set = np.array(testing_data).reshape(-1, image_size, image_size, 1)
        alpha_x_test = new_test_feature_set/255.0
        alpha_x_test = np.array(alpha_x_test)

        return alpha_x_test


    def predict_clicked(self):
        predictions = None
        if self.input_line_edit.text() == '':
            raise ValueError('Please provide a input image')

        if not os.path.exists(trained_alphabet_model_path):
            Trainer.training_alphabet_model()
        
        reload_alphabet_model_one = load_model(trained_alphabet_model_path)
        
        alpha_x_test = Predictor.processing_user_input(self.input_line_edit.text())

        predictions = reload_alphabet_model_one.predict([alpha_x_test])
        print('Result -----> ', categories[int(predictions)])
        self.result_window.setText(categories[int(predictions[0])])

        
class Trainer():
    def __init__(self) -> None:
        pass
    
    def training_alphabet_model():
        
        bad_images = []
        for category in categories:
            path = os.path.join(alpha_training_data_dir,category).replace('\\', '/')
            class_num = categories.index(category)
            for image in os.listdir(path):
                image_full_path = os.path.join(path, image)
                try:
                    alpha_train_data = cv2.imread(image_full_path, cv2.IMREAD_GRAYSCALE)
                    new_alpha_train_data = cv2.resize(alpha_train_data, (image_size, image_size))
                    alpha_training_data_one.append((new_alpha_train_data, class_num))
                except Exception as e:
                    bad_images.append(image_full_path)
        if bad_images:
            print('Following training image data are bad and cant read them /n{}'.format(bad_images))

        random.shuffle(alpha_training_data_one)

        for features, label in alpha_training_data_one:
            features_set.append(features)
            labels_set.append(label)
       
        new_features_set = np.array(features_set).reshape(-1, image_size, image_size, 1)
        if not os.path.exists(trained_aplhabet_features_model_path):
            pickle_out_features = open(trained_aplhabet_features_model_path, 'wb')
            pickle.dump(new_features_set, pickle_out_features)
            pickle_out_features.close()
        if not os.path.exists(trained_aplhabet_label_model_path):
            pickle_out_labels = open(trained_aplhabet_label_model_path, 'wb')
            pickle.dump(labels_set, pickle_out_labels)
            pickle_out_labels.close()
        
        alpha_x_train = pickle.load(open(trained_aplhabet_features_model_path, 'rb'))
        alpha_y_train = pickle.load(open(trained_aplhabet_label_model_path, 'rb'))
        alpha_x_train = np.array(alpha_x_train)
        alpha_y_train = np.array(alpha_y_train)

        alpha_model = Sequential()

        alpha_model.add(Conv2D(64, (3,3), input_shape = alpha_x_train.shape[1:], activation=tf.nn.relu))
        alpha_model.add(MaxPooling2D(pool_size=(2,2)))

        alpha_model.add(Conv2D(64, (3,3),  activation=tf.nn.relu))
        alpha_model.add(MaxPooling2D(pool_size=(2,2)))

        alpha_model.add(Flatten())

        alpha_model.add(BatchNormalization())
        alpha_model.add(Dense(64, activation = tf.nn.relu))
        alpha_model.add(BatchNormalization())
        alpha_model.add(Dense(64, activation = tf.nn.relu))
        alpha_model.add(BatchNormalization())
        alpha_model.add(Dense(1, activation = tf.nn.sigmoid))
        

        alpha_model.compile(optimizer=Adam(learning_rate=1e-6), loss='binary_crossentropy', metrics=['accuracy'])
        alpha_model.fit(alpha_x_train, alpha_y_train, batch_size=10, epochs=10)
        validation_loss, validation_accuracy = alpha_model.evaluate(alpha_x_train, alpha_y_train)

        alpha_model.save(trained_alphabet_model_path)

    def training_number_model():
        numeric_data = tf.keras.datasets.mnist # images 28x28 of hand written digits 0-9
        (num_x_train, num_y_train) = numeric_data.load_data()
        num_x_train = normalize(num_x_train, axis= 1)

        if not os.path.exists(trained_number_model_path):
            num_model = Sequential() # its a feed forward model hence sequential
            num_model.add(Flatten())
            num_model.add(Dense(128, activation= tf.nn.relu)) # activation function that defines the neuron to fire
            num_model.add(Dense(128, activation= tf.nn.relu))
            num_model.add(Dense(36, activation= tf.nn.softmax))

            num_model.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics= ['accuracy'])
            num_model.fit(num_x_train, num_y_train, epochs= 3)

            num_model.save(trained_number_model_path)

def main():
    app = QtWidgets.QApplication(sys.argv)
    GUI = Predictor()
    GUI.show()
    sys.exit(app.exec_())

main()