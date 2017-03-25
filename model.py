# Model parameters
csv_filename = 'data/driving_log_1.csv'
zero_frac = 0.25   # fraction of 0 steering records to be left after undersampling
m_frac = 0.6       # fraction of -1 steering records to be left after undersampling 
p_frac = 0.6       # fraction of 1 steering records to be left after undersampling
correction = 0.01   # steering correction for left and right camera images
top_crop, bottom_crop, left_crop, right_crop = 60, 25, 0, 0    # image crop parameters
layers = 2         # number of layers in input images
batch_size = 32    # initial batch size (is doubled by adding flipped images)
EPOCHS = 5         # number of training epocha
model_filename = 'model_p.h5'

# BehavioralCloning Class

import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split
import sklearn

class BehavioralCloning:
    def __init__(self, csv_filename, model_filename, layers, zero_frac, m_frac, p_frac, correction, top_crop, bottom_crop, left_crop, right_crop, batch_size, EPOCHS):
        self.layers = layers
        self.zero_frac = zero_frac
        self.m_frac = m_frac
        self.p_frac = p_frac
        self.correction = correction
        self.top_crop = top_crop
        self.bottom_crop = bottom_crop
        self.left_crop = left_crop
        self.right_crop = right_crop
        self.batch_size = batch_size
        self.EPOCHS = EPOCHS
        self.driving_log = pd.read_csv(csv_filename, usecols=['center', 'left', 'right', 'steering'])
        self.model_filename = model_filename
        self.__defineCNN()
    
    # CNN definition
    def __defineCNN(self):
        self.model = Sequential()
        self.model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,self.layers)))
        self.model.add(Cropping2D(cropping=((self.top_crop, self.bottom_crop), (self.left_crop, self.right_crop)),
                                  input_shape=(160,320,self.layers)))
        self.model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
        self.model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
        self.model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
        self.model.add(Convolution2D(64,3,3, activation='relu'))
        self.model.add(Convolution2D(64,3,3, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(100))
        self.model.add(Dense(50))
        self.model.add(Dense(10))
        self.model.add(Dense(1))        
    
    # Run the whole pipeline
    def run(self):
        self.undersample()
        self.augment_center()
        self.split_train_validation()
        self.train()
     
    # Preview input data
    def preview(self):
        return self.driving_log.head()
    
    # Statistics of input data
    def describe(self, column):
        return self.driving_log[column].describe()
    
    # Histogram for an input data column
    def histogram(self, column):
        _ = self.driving_log[column].hist(bins=101, figsize=(15,5))
    
    # Undersample specified value leaving a fraction of it
    def __undersample(self, log, value, frac):
        log_nonvalue = log[log['steering']!=value]
        log_value = log[log['steering']==value]

        if len(log_value)>0:
            log_value = log_value.sample(frac=frac)

        log = log_value.append(log_nonvalue)
        log = log.reset_index(drop=True)
        return log
    
    # Full undersampling
    def undersample(self):
        self.driving_log = self.__undersample(self.driving_log, 0, self.zero_frac)
        self.driving_log = self.__undersample(self.driving_log, -1, self.m_frac)
        self.driving_log = self.__undersample(self.driving_log, 1, self.p_frac)
   
    # Shuffle a DataFrame
    def __shuffle(self, data):
        return data.sample(frac=1).reset_index(drop=True)
        
    # Build new log taking into account correction for left and right camera images 
    def augment_center(self):
        self.driving_log = pd.DataFrame({'image':self.driving_log['center']
                                        .append(self.driving_log['left'])
                                        .append(self.driving_log['right']),
                                        'steering':self.driving_log['steering']
                                        .append(self.driving_log['steering'] + self.correction)
                                        .append(self.driving_log['steering'] - self.correction)})
        #Shuffle
        self.driving_log = self.__shuffle(self.driving_log)

    # Image preprocessing (except normalization and cropping that are done in CNN layers)
    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:,:,1:]
        return image
    
    # Get X,y from the training or validation log. Preprocessing and augmentation with flipped images are applied
    def __get_data(self, log):    
        images = []
        steerings = []

        for _, row in log.iterrows():
            filename = row['image'].strip()
            image = self.preprocess(mpimg.imread(filename))
            image_flip = cv2.flip(image, 1)
            if layers==1:
                image_flip = image_flip[:,:,None]

            steering = row['steering']

            images.extend([image, image_flip])
            steerings.extend([steering, -steering])

        X, y = np.array(images), np.array(steerings)
        X, y = sklearn.utils.shuffle(X, y)

        return X, y    
    
    # Generator
    def __generator(self, sample_log):
        n_rows = len(sample_log)
        while 1: # Loop forever so the generator never terminates
            sample_log = self.__shuffle(sample_log)  #shuffle sample_log DataFrame

            for offset in range(0, n_rows, self.batch_size):
                batch_log = sample_log[offset : offset + self.batch_size]

                X, y = self.__get_data(batch_log)

                yield X, y        
    
    # Split the driving log to train and validation sets, return number of records in each set
    def split_train_validation(self):
        self.train_log, self.validation_log = train_test_split(self.driving_log, test_size=0.2)
        self.train_log = self.train_log.reset_index(drop=True)
        self.validation_log = self.validation_log.reset_index(drop=True)
        
        # Train and Validation generators
        self.train_generator = self.__generator(self.train_log)
        self.validation_generator = self.__generator(self.validation_log)        
        
        return len(self.train_log), len(self.validation_log )
   
    # Train the model and save the result
    def train(self):
        self.model.compile(loss='mse', optimizer='adam')

        self.history_object = self.model.fit_generator(self.train_generator,
                                     samples_per_epoch = len(self.train_log),
                                     validation_data = self.validation_generator, 
                                     nb_val_samples=len(self.validation_log),
                                     nb_epoch=self.EPOCHS,
                                     verbose=1)
        self.model.save(self.model_filename)

    # Plot the training and validation loss for each epoch    
    def plot_loss(self):
        plt.plot(self.history_object.history['loss'])
        plt.plot(self.history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        _ = plt.legend(['training set', 'validation set'], loc='upper right')
            
            

if __name__ == '__main__':
    cloningModel = BehavioralCloning(csv_filename, model_filename, layers, zero_frac, m_frac, p_frac, correction, top_crop, bottom_crop, left_crop, right_crop, batch_size, EPOCHS)
    
    cloningModel.run()
    
    print('Done')
        
        
            