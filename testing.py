import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import cv2
import re
import math
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers
keras = tf.keras

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from datetime import datetime
from joblib import Parallel,delayed

from scipy import signal
from scipy.signal import resample,hilbert

#####################################   USER INPUT   ###################################################################

testing_dir = 'path_to_testing_dir'
data_dir = 'path_to_stead'

saved_model_path_TraceCategory = 'path_to_saved_model'
saved_model_path_SourceMagnitude = 'path_to_saved_model'
saved_model_path_Parrival = 'path_to_saved_model'
saved_model_path_Sarrival = 'path_to_saved_model'

test_trace_names = ['KAN16.GS_20150424061553_EV','KAN16.GS_20151111113626_EV'] #Enter the names of earthquakes which you want to find correlations of
####################################################################################################

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

test_files = [
    f'{testing_dir}/{test_trace_names[0]}.png',
    f'{testing_dir}/{test_trace_names[1]}.png'
]


def custom_data_generator(file_list, target_size):
    images = []
    for file in file_list:
        image = load_img(file, target_size=target_size,color_mode='grayscale')
        image = img_to_array(image)
        image /= 255.0
        images.append(image)
    print(len(np.array(images)))
    return np.array(images)


def testing(target,test_images):

    if(target == "trace_category"):
        # Load the model
        output_file = 'predicted_trace_categories.txt'
        loaded_model = keras.models.load_model(saved_model_path_TraceCategory)

        #steps = len(test_files) // batch_size

        # Make predictions on the test dataset
        predictions = loaded_model.predict(test_images)

        # Convert probabilities to class labels
        predicted_classes = np.argmax(predictions, axis=-1)

        # Save the predictions to a text file
        with open(output_file, 'w') as f:
            for image_name,label in zip(test_trace_names, predicted_classes):
                if label == 1:
                        f.write(f"{image_name}: earthquake\n")
                else:
                    f.write(f"{image_name}: noise\n")      
        
        print(f"Predicted trace categories have been saved to {output_file}")

    if(target == "source_magnitude"):
        # Load the model
        output_file = 'predicted_magnitudes.txt'
        loaded_model = keras.models.load_model(saved_model_path_SourceMagnitude)

        #steps = len(test_files) // batch_size

        # Make predictions on the test dataset
        predictions = loaded_model.predict(test_images)

        # Save the predictions to a text file
        with open(output_file, 'w') as f:
            for image_name, pred in zip(test_trace_names, predictions):
                f.write(f"{image_name}: Predicted Magnitude: {pred[0]}\n")      
        
        print(f"Predicted magnitudes have been saved to {output_file}")
                
    if(target == 'p_arrival_sample'):
        # Load the model
        output_file = 'p_arrival_sample.txt'
        loaded_model = keras.models.load_model(saved_model_path_Parrival)

        # Make predictions on the test dataset
        predictions = loaded_model.predict(test_images)

        # Save the predictions to a text file
        with open(output_file, 'w') as f:
            for image_name, pred in zip(test_trace_names, predictions):
                f.write(f"{image_name}: Predicted p_arrival_sample: {pred[0]}\n")      
        
        print(f"Predicted p_arrival_sample have been saved to {output_file}")

    if(target == 's_arrival_sample'):
        # Load the model
        output_file = 's_arrival_sample.txt'
        loaded_model = keras.models.load_model(saved_model_path_Sarrival)

        # Make predictions on the test dataset
        predictions = loaded_model.predict(test_images)

        # Save the predictions to a text file
        with open(output_file, 'w') as f:
            for image_name, pred in zip(test_trace_names, predictions):
                f.write(f"{image_name}: Predicted s_arrival_sample: {pred[0]}\n")      
        
        print(f"Predicted s_arrival_sample have been saved to {output_file}")

#####################################################################################################################################  

#batch_size_regression = 64
img_height, img_width = 100,150
test_gen = custom_data_generator(test_files, target_size=(img_height, img_width))

testing('trace_category',test_gen)
testing('source_magnitude',test_gen)
testing('p_arrival_sample',test_gen)
testing('s_arrival_sample',test_gen)
