import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import cv2

import tensorflow as tf
from tensorflow.keras import layers
keras = tf.keras

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from datetime import datetime
from joblib import Parallel,delayed

from scipy import signal
from scipy.signal import resample,hilbert
#from keras.wrappers.scikit_learn import KerasClassifier


############################# USER INPUT #############################

# set CNN data paths

# path to image dataset
dir = '/Users/Siddharth/Desktop/IIITB/bashok_srip/STEAD_dataset/images/training_spectrograms'

# path to stead dataset
data_dir = '/Users/Siddharth/Desktop/IIITB/bashok_srip/STEAD_dataset'


# paths to csv and hdf5 (waveform/signal) files
noise_csv_path = data_dir+'/chunk1.csv'
noise_sig_path = data_dir+'/chunk1.hdf5'
eq1_csv_path = data_dir+'/chunk2.csv'
eq1_sig_path = data_dir+'/chunk2.hdf5'
eq2_csv_path = data_dir+'/chunk3.csv'
eq2_sig_path = data_dir+'/chunk3.hdf5'
eq3_csv_path = data_dir+'/chunk4.csv'
eq3_sig_path = data_dir+'/chunk4.hdf5'
eq4_csv_path = data_dir+'/chunk5.csv'
eq4_sig_path = data_dir+'/chunk5.hdf5'
eq5_csv_path = data_dir+'/chunk6.csv'
eq5_sig_path = data_dir+'/chunk6.hdf5'

# read the noise and earthquake csv files into separate dataframes:
earthquakes_1 = pd.read_csv(eq1_csv_path,low_memory=False)
earthquakes_2 = pd.read_csv(eq2_csv_path,low_memory=False)
earthquakes_3 = pd.read_csv(eq3_csv_path,low_memory=False)
earthquakes_4 = pd.read_csv(eq4_csv_path,low_memory=False)
earthquakes_5 = pd.read_csv(eq5_csv_path,low_memory=False)
noise = pd.read_csv(noise_csv_path,low_memory=False)

full_csv = pd.concat([earthquakes_1,earthquakes_2,earthquakes_3,earthquakes_4,earthquakes_5,noise])

# making lists of trace names for the earthquake sets
eq1_list = earthquakes_1['trace_name'].to_list()
eq2_list = earthquakes_2['trace_name'].to_list()
eq3_list = earthquakes_3['trace_name'].to_list()
eq4_list = earthquakes_4['trace_name'].to_list()
eq5_list = earthquakes_5['trace_name'].to_list()

# making a list of trace names for the noise set
noise_list = noise['trace_name'].to_list()


##############################################################################################################################################


class SeismicCNN():

    def __init__(self,model_type,target,choose_dataset_size,full_csv,dir):
        self.model_type = model_type
        self.target = target
        self.choose_dataset_size = choose_dataset_size
        self.full_csv = full_csv
        self.dir = dir
        self.traces_array = []
        self.img_dataset = []
        self.labels = []
        self.imgs = []
        self.imgs_names = []
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        self.model = []
        self.test_loss = []
        self.test_acc = []
        self.predicted_classes = []
        self.predicted_probs = []
        self.cm = []
        self.epochs = []
        self.history = []


        if self.model_type == 'classification':
            # create list of traces in the image datset
            print('Creating seismic trace list')
            for filename in os.listdir(dir): # loop through image directory and get filenames
                if filename.endswith('.png'):
                    self.traces_array.append(filename[0:-4]) # remove the .png from filename
                    #print(f'self.traces_array is-- {self.traces_array}')

            if choose_dataset_size == 'full':
                # select only the rows in the metadata dataframe which correspond to images
                print('Selecting traces matching images in directory')
                self.img_dataset = self.full_csv.loc[self.full_csv['trace_name'].isin(self.traces_array)] # select rows from the csv that have matching image files
                self.labels = self.img_dataset['trace_category'] # target variable, 'earthquake' or 'noise'
                self.labels = np.array(self.labels.map(lambda x: 1 if x == 'earthquake_local' else 0)) # transform target variable to numerical categories
                print(f'The number of traces in the directory is {len(self.img_dataset)}')
                count = 0
                for i in range(0,len(self.img_dataset['trace_name'])): # loop through images and read them into the imgs array
                    count += 1
                    #print(f'Working on trace # {count}')
                    img= cv2.imread(self.dir+'/'+self.img_dataset['trace_name'].iloc[i]+'.png',0) # read in image as grayscale image
                    self.imgs.append(img)
                self.imgs = np.array(self.imgs)
                print(f'shape is {self.imgs.shape}')

            elif isinstance(choose_dataset_size, int):
                seismic_dataset = self.full_csv.loc[self.full_csv['trace_name'].isin(self.traces_array)] # get rows of csv dataset that have corresponding images in directory
                if seismic_dataset.empty:
                    raise ValueError("The seismic dataset is empty. Check if the CSV files and directory contain matching trace names.")
                choose_seismic_dataset = np.random.choice(np.array(seismic_dataset['trace_name']),choose_dataset_size,replace=False)
                self.img_dataset = seismic_dataset.loc[seismic_dataset['trace_name'].isin(choose_seismic_dataset)] # random choice of images from the directory
                self.labels = self.img_dataset['trace_category'] # target variable, 'earthquake' or 'noise'
                self.labels = np.array(self.labels.map(lambda x: 1 if x == 'earthquake_local' else 0)) # transform target variable to numerical categories
                print(f'The number of traces in the directory is {len(self.img_dataset)}')
                count = 0
                for i in range(0,len(self.img_dataset['trace_name'])): # loop through trace names in filtered dataframe and append images to imgs array
                    count += 1
                    #print(f'Working on trace # {count}')
                    img= cv2.imread(self.dir+'/'+self.img_dataset['trace_name'].iloc[i]+'.png',0) # read in image as grayscale image
                    self.imgs_names.append(self.img_dataset['trace_name'].iloc[i])
                    self.imgs.append(img)
                print(f'the count is: {count}')
                self.imgs = np.array(self.imgs)
                print(f'shape is {self.imgs.shape}')

            else:
                print('Error: please choose either "full" for variable choose_dataset_size to use the full dataset, or provide an integer number of random samples to take from the dataset')

        elif self.model_type == 'regression':

            for filename in os.listdir(dir): # loop through every file in the directory and get trace names from the image files
                if filename.endswith('.png'):
                    self.traces_array.append(filename[0:-4]) # remove .png from image file names
            print(f'The number of all traces in the directory including noise is {len(self.traces_array)}')
            local_quakes = self.full_csv[self.full_csv['trace_category'] == 'earthquake_local'] # get only signals corresponding to local earthquakes, not noise
            print(f'size of local_quakes is- {len(local_quakes)}')
            local_quakes_data = local_quakes.loc[local_quakes['trace_name'].isin(self.traces_array)]
            print(f'size of local_quakes_data is- {len(local_quakes_data)}')

            if choose_dataset_size == 'full':
                self.img_dataset = local_quakes.loc[local_quakes['trace_name'].isin(self.traces_array)]
                print(f'The number of all earthquakes in the directory excluding noise is {len(self.img_dataset)}')
                self.labels = self.img_dataset[target] # target variable

                count = 0
                for i in range(0,len(self.img_dataset['trace_name'])): # loop through the images dataframe and read in the images with matching trace names
                    count += 1
                    print(f'Working on trace # {count}')
                    img= cv2.imread(dir+'/'+self.img_dataset['trace_name'].iloc[i]+'.png',0) # read in image as grayscale image
                    self.imgs.append(img)
                self.imgs = np.array(self.imgs)
                print(self.imgs.shape)

            elif type(choose_dataset_size) == int:

                choose_local_quakes = np.random.choice(np.array(local_quakes_data['trace_name']),choose_dataset_size,replace=False)
                self.img_dataset = local_quakes_data.loc[local_quakes_data['trace_name'].isin(choose_local_quakes)]
                self.labels = self.img_dataset[self.target] # target variable

                count = 0
                for i in range(0,len(self.img_dataset['trace_name'])): # loop through dataframe and read in images corresponding to trace names in data frame
                    count += 1
                    #print(f'Working on trace # {count}')
                    img= cv2.imread(dir+'/'+self.img_dataset['trace_name'].iloc[i]+'.png',0) # read in image as grayscale image
                    self.imgs_names.append(self.img_dataset['trace_name'].iloc[i])
                    self.imgs.append(img)
                self.imgs = np.array(self.imgs)
                print(self.imgs.shape)

            else:
                print('Error: please choose either "full" for variable choose_dataset_size to use the full dataset, or provide an integer number of random samples to take from the dataset')

        else:
            print('Error: please choose either "classification" or "regression" for CNN model type')

    #####################################################################################################################################################################################################################

    def train_test_split(self,test_size,random_state):
            self.train_images, self.test_images, self.train_labels, self.test_labels = train_test_split(self.imgs,self.labels,random_state=random_state,test_size=test_size)

            # test_image_filenames = [os.path.basename(path) for path in self.test_images]

            # # Optionally, save to a file
            # with open('test_image_filenames.txt', 'w') as f:
            #     for filename in test_image_filenames:
            #         f.write(f"{filename}\n")

            print(f'The training images set is size: {self.train_images.shape}')
            print(f'The training labels set is size: {self.train_labels.shape}')
            print(f'The testing images set is size: {self.test_images.shape}')
            print(f'The testing labels set is size: {self.test_labels.shape}')

            print('Scaling image intensity')
            self.train_images = self.train_images/255.0 # scale intensity to between 0 and 1
            self.test_images = self.test_images/255.0 # scale intensity to between 0 and 1

            img_height = self.train_images.shape[1] # get height of each image in pixels
            img_width = self.train_images.shape[2] # get width of each image in pixels

            print('Resizing images')
            self.train_images = self.train_images.reshape(-1,img_height,img_width,1) # reshape to input into CNN which requires a 4-tensor
            self.test_images = self.test_images.reshape(-1,img_height,img_width,1) # reshape to input into CNN which requires a 4-tensor

    ####################################################################################################################################################################################

    def classification_cnn(self,epochs):
            self.epochs = epochs

            # set callbacks so that the model will be saved after each epoch
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    filepath=f'./saved_models/specs_{str(self.choose_dataset_size)}dataset_{self.model_type}_{self.target}_epochs{self.epochs}_{format(datetime.now().strftime("%Y%m%d"))}',
                    save_freq='epoch')
            ]

            # build CNN on dataset
            print('Building CNN model')

            model = keras.Sequential()
            model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), activation = 'relu', padding = 'same'))
            model.add(keras.layers.MaxPool2D(2,2))
            model.add(keras.layers.Dropout(0.25))
            model.add(keras.layers.Flatten(input_shape=(self.imgs.shape[1],self.imgs.shape[2])))
            model.add(keras.layers.Dense(64,activation='relu'))
            model.add(keras.layers.Dense(16,activation='relu'))
            model.add(keras.layers.Dense(2,activation='softmax'))

            opt = keras.optimizers.Adam(learning_rate=1e-6)
            model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics='accuracy')
            self.history = model.fit(self.train_images,self.train_labels,batch_size=64,epochs=epochs,callbacks=callbacks,validation_split=0.2)

            print(model.summary())

            # Set model save path
            saved_model_path = f'./saved_models/specs_{str(self.choose_dataset_size)}dataset_{self.model_type}_{self.target}_epochs{self.epochs}_{format(datetime.now().strftime("%Y%m%d"))}'
            # Save entire model to a HDF5 file
            model.save(saved_model_path)

            self.model = model
            #-------------------------------------------------------------------------------
            # output_file = 'predicted_trace_categories.txt'
            # loaded_model = keras.models.load_model(saved_model_path)
    
            # # Make predictions on the test dataset
            # predicted_probs = loaded_model.predict(self.test_images)
            
            # # Convert probabilities to class labels
            # predicted_classes = np.argmax(predicted_probs, axis=-1)
            
            # # Store the predicted trace categories in a text file
            # with open(output_file, 'w') as f:
            #     for image_name, label in zip(self.test_images, predicted_classes):
            #         if label == 1:
            #             f.write(f"{image_name}: earthquake signal\n")
            #         else:
            #             f.write(f"{image_name}: noise\n")         
            
            # print(f"Predicted trace categories have been saved to {output_file}")
            #  #-------------------------------------------------------------------------------

    ########################################################################################################################################################################################################################

    def regression_cnn(self,target,epochs):
            self.epochs = epochs
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    filepath=f'./saved_models/specs_{str(self.choose_dataset_size)}dataset_{self.model_type}_{self.target}_epochs{self.epochs}_{format(datetime.now().strftime("%Y%m%d"))}',
                    save_freq='epoch')
            ]

            # build CNN on dataset
            print('Building Regression CNN model')
            model = keras.Sequential()
            model.add(keras.layers.Conv2D(64, kernel_size=(5, 5), activation = 'relu', padding = 'same'))
            model.add(keras.layers.MaxPool2D(2,2))
            model.add(keras.layers.Dropout(0.50))
            model.add(keras.layers.Flatten(input_shape=(self.imgs.shape[1],self.imgs.shape[2])))
            model.add(keras.layers.Dense(16,activation='relu')) #take output of this and feed it to svm. Insert conv2d and maxppool after dropout layers
            model.add(keras.layers.Dense(1))

            opt = keras.optimizers.Adam(learning_rate=1e-5)
            model.compile(optimizer=opt,loss='mse')
            self.history = model.fit(self.train_images,self.train_labels,epochs=epochs,callbacks=callbacks,validation_split=0.2,batch_size=64)

            print(model.summary())

            # Set model save path
            saved_model_path = f'./saved_models/specs_{str(self.choose_dataset_size)}dataset_{self.model_type}_{self.target}_epochs{self.epochs}_{format(datetime.now().strftime("%Y%m%d"))}' # _%H%M%S
            # Save entire model to a HDF5 file
            model.save(saved_model_path)

            self.model = model
            #---------------------------------------------------------------------------
            # if(target == "source_magnitude"):
            #     # Load the model
            #     output_file = 'predicted_magnitudes.txt'
            #     loaded_model = keras.models.load_model(saved_model_path)

            #     # Make predictions on the test dataset
            #     predictions = loaded_model.predict(self.test_images)

            #     # Save the predictions to a text file
            #     with open(output_file, 'w') as f:
            #         for image_name, pred in zip(self.test_images, predictions):
            #             f.write(f"{image_name}: Predicted Magnitude: {pred[0]}\n")      
                
            #     print(f"Predicted magnitudes have been saved to {output_file}")
            
            # if(target == 'p_arrival_sample'):
            #     # Load the model
            #     output_file = 'p_arrival_sample.txt'
            #     loaded_model = keras.models.load_model(saved_model_path)

            #     # Make predictions on the test dataset
            #     predictions = loaded_model.predict(self.test_images)

            #     # Save the predictions to a text file
            #     with open(output_file, 'w') as f:
            #         for image_name, pred in zip(self.test_images, predictions):
            #             f.write(f"{image_name}: Predicted p_arrival_sample: {pred[0]}\n")      
                
            #     print(f"Predicted p_arrival_sample have been saved to {output_file}")

            # if(target == 's_arrival_sample'):
            #     # Load the model
            #     output_file = 's_arrival_sample.txt'
            #     loaded_model = keras.models.load_model(saved_model_path)

            #     # Make predictions on the test dataset
            #     predictions = loaded_model.predict(self.test_images)

            #     # Save the predictions to a text file
            #     with open(output_file, 'w') as f:
            #         for image_name, pred in zip(self.test_images, predictions):
            #             f.write(f"{image_name}: Predicted s_arrival_sample: {pred[0]}\n")      
                
            #     print(f"Predicted s_arrival_sample have been saved to {output_file}")
            # #-------------------------------------------------------------------------

    #########################################################################################################################################################################################################

    def evaluate_classification_model(self):
            print('Evaluating model on test dataset')
            self.test_loss, self.test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=1) # get model evaluation metrics
            print("\nTest data, accuracy: {:5.2f}%".format(100*self.test_acc))

            print('Finding predicted classes and probabilities to build confusion matrix')
            self.predicted_classes = np.argmax(self.model.predict(self.test_images),axis=-1) # predict the class of each image
            self.predicted_probs = self.model.predict(self.test_images) # predict the probability of each image belonging to a class

            # create confusion matrix
            print('Building confusion matrix')
            self.cm = confusion_matrix(self.test_labels,self.predicted_classes) # compare target values to predicted values and show confusion matrix
            print(self.cm)
            accuracy = accuracy_score(self.test_labels,self.predicted_classes)
            precision = precision_score(self.test_labels,self.predicted_classes, average='weighted')
            recall = recall_score(self.test_labels,self.predicted_classes, average='weighted')
            print(f'The accuracy of the model is {accuracy}, the precision is {precision}, and the recall is {recall}.')


            # plot confusion matrix
            plt.style.use('default')
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=self.cm, display_labels=['noise','earthquake'])
        
            disp.plot(cmap='Blues', values_format='',ax=ax)
            plt.title(f'Classification CNN Results ({self.epochs} epochs)')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png')
            plt.show()

            # plot accuracy history
            plt.style.use('ggplot')
            fig, ax = plt.subplots(figsize=(7,7))
            ax.plot(self.history.history['accuracy'])
            ax.plot(self.history.history['val_accuracy'])
            ax.set_title('Model Accuracy')
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Epoch')
            ax.legend(['train','test'])
            plt.savefig('model_accuracy.png')
            plt.show()

    ####################################################################################################################################################################################
    
    def evaluate_regression_model(self):
            print('Evaluating model on test dataset')
            self.test_loss = self.model.evaluate(self.test_images, self.test_labels, verbose=1)
            print(f'Test data loss: {self.test_loss}')

            print('Getting predictions')
            self.predicted = self.model.predict(self.test_images)

            plt.style.use('ggplot')
            fig, ax = plt.subplots(figsize=(7,7))
            ax.scatter(self.test_labels,self.predicted,alpha=0.25)
            point1 = [0,0]
            point2 = [6,6]
            xvalues = [point1[0], point2[0]]
            yvalues = [point1[1], point2[1]]
            ax.plot(xvalues,yvalues,color='blue')
            ax.set_ylabel('Predicted Value',fontsize=14)
            ax.set_xlabel('Observed Value',fontsize=14)
            ax.set_title(f'Regression CNN Results 100k dataset | ({self.epochs} epochs)')
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            plt.tight_layout()
            plt.savefig('true_vs_predicted.png')
            plt.show()

            fig, ax = plt.subplots(figsize=(7,7))
            ax.plot(self.history.history['loss'])
            ax.plot(self.history.history['val_loss'])
            ax.set_title('Model Accuracy')
            ax.set_ylabel('Loss (MSE)')
            ax.set_xlabel('Epoch')
            ax.legend(['train','test'])
            plt.savefig('model_loss.png')
            plt.show()

    #####################################################################################################################################################################

# Using the class for a classification CNN
# model_cnn_c1 = SeismicCNN('classification','trace_category',75001,full_csv,dir) # initialize the class
# model_cnn_c1.train_test_split(test_size=0.00002667,random_state=44) # train_test_split
# model_cnn_c1.classification_cnn(50) # use the regression cnn method with 15 epochs with a target variable
# model_cnn_c1.evaluate_classification_model() # evaluate the model

# run the model using CNN regression to predict the earthquake magnitude
# model_cnn_rm = SeismicCNN('regression','source_magnitude',75001,full_csv,dir); # initialize the class #try putting 10000 instead?
# model_cnn_rm.train_test_split(test_size=0.00002667,random_state=44) # train_test_split
# model_cnn_rm.regression_cnn('source_magnitude',20) # use the classification cnn method with 15 epochs
#model_cnn_rm.evaluate_regression_model() # evaluate the model

# run the model using CNN regression to predict the p-wave arrival time
# model_cnn_rp = SeismicCNN('regression','p_arrival_sample',75001,full_csv,dir); # initialize the class
# model_cnn_rp.train_test_split(test_size=0.00002667,random_state=44) # train_test_split
# model_cnn_rp.regression_cnn('p_arrival_sample',20) # use the classification cnn method with 15 epochs
#model_cnn_rp.evaluate_regression_model() # evaluate the model

# # run the model using CNN regression to predict the s-wave arrival time
model_cnn_rs = SeismicCNN('regression','s_arrival_sample',75001,full_csv,dir); # initialize the class
model_cnn_rs.train_test_split(test_size=0.00002667,random_state=44) # train_test_split
model_cnn_rs.regression_cnn('s_arrival_sample',20) # use the classification cnn method with 15 epochs
#model_cnn_rs.evaluate_regression_model() # evaluate the model