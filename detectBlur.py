import cv2
import json
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from skimage.filters import sobel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Dropout, GlobalAveragePooling2D, MaxPooling2D


class DetectBlur():
    """An Image Blur Detection"""

    def __init__(self, method='cnn'):
        self.method = method
        if method == 'cnn':
            self.threshold = 0.5
        elif method == 'tenengrad':
            self.threshold = 0.00039
    
    def predict(self, image_path):
        
        # Load the image
        image = cv2.imread(image_path)

        # Image Preprocessing
        image = self.preprocessing(image)

        # Predict the image. if pred < 0.5, the image is considered as blurry, vice versa
        if self.method == 'cnn':
            # load weight model
            model_path = 'model_weights/cnn_model.h5'
            model = self.build_model()
            model.load_weights(model_path)

            image = np.array([image])
            pred = model.predict(image)[0][0]

        elif self.method == 'tenengrad':
            pred = self.tenengrad(image)
        
        return pred < self.threshold


    def preprocessing(self, img):
        scale = 1/255

        # convert image from bgr to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # pixel value normalization
        img = img * scale

        if self.method == 'cnn':
            img = np.expand_dims(img, axis=2)
            img = tf.image.resize(img, size=(256, 256), method='nearest')
        elif self.method == 'tenengrad':
            img = img
        
        return img
    
    def tenengrad(self, image, param='var', norm=False): # Sobel Operator
        img = sobel(image)
        if norm:
            img = (img - min(img)) / (max(img) - min(img))
        
        if param=='var':
            return img.var()
        elif param=='max':
            return img.max()
        elif param=='mean':
            return img.mean()
        elif param=='image':
            return img

    def build_model(self):
        input_shape = (256, 256, 1)
        model = Sequential()
        model.add(Conv2D(64, 3, input_shape=input_shape, activation='relu'))
        model.add(Conv2D(64, 3, activation='relu'))
        model.add(MaxPooling2D(2))
        model.add(Conv2D(32, 3, activation='relu'))
        model.add(Conv2D(32, 3, activation='relu'))
        model.add(MaxPooling2D(2))
        model.add(Conv2D(16, 3, activation='relu'))
        model.add(Conv2D(16, 3, activation='relu'))
        model.add(Flatten())
        
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        return model

root = tk.Tk()
root.withdraw()

if __name__ == '__main__':
    path = filedialog.askopenfilename()

    file = path.split('/')[-1]
    blur_detector = DetectBlur(method='cnn')
    
    isBlur = blur_detector.predict(path)
    
    if isBlur:
        blurred = "True"
        print('{} \nis blurry image'.format(file))
    else:
        blurred = "False"
        print('{} \nis sharp image'.format(file))

    result = {"file": file, "blur": blurred}
    # save predict result in json file
    json_string = json.dumps(result)
    json_file = open("result.json", "w")
    json_file.write(json_string)
    json_file.close()
