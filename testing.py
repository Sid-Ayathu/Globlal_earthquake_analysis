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

#####################################################################################################################################

testing_dir = '/Users/Siddharth/Desktop/IIITB/bashok_srip/STEAD_dataset/images/testing_spectrograms'
data_dir = '/Users/Siddharth/Desktop/IIITB/bashok_srip/STEAD_dataset'

saved_model_path_TraceCategory = './saved_models/specs_75001dataset_classification_trace_category_epochs50_20240721'
saved_model_path_SourceMagnitude = './saved_models/specs_75001dataset_regression_source_magnitude_epochs20_20240721'
saved_model_path_Parrival = './saved_models/specs_75001dataset_regression_p_arrival_sample_epochs20_20240722'

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

test_trace_names = ['KAN16.GS_20150424061553_EV','KAN16.GS_20151111113626_EV']
test_files = [
    f'{testing_dir}/{test_trace_names[0]}.png',
    f'{testing_dir}/{test_trace_names[1]}.png'

    #these gave decent predictions:
    # f'{testing_dir}/imagesB086.PB_20120613131555_EV.png',
    # f'{testing_dir}/imagesB086.PB_20070926054633_EV.png'
]


#####################################################################################################################################

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
        loaded_model = keras.models.load_model(saved_model_path)

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
# regression_testing('s_arrival_sample',test_gen)

###############################################################################################################################
#THIS gives me exact date and time:
# trace_start_times = earthquakes_4.loc[earthquakes_4['trace_name'] == 'KAN16.GS_20151111113626_EV', 'trace_start_time']
# datetime_value = trace_start_times.values[0]

# date_time_parts_str = datetime_value.split()
# date_str = date_time_parts_str[0]
# time_str = date_time_parts_str[1]
# time_str_split = time_str.split('.')
# time_str_rounded_down = time_str_split[0]
# print(f'date = {date_str}')
# print(f'time = {time_str_rounded_down}')

#THIS gives me exact latitude and long:
# receiver_latitude = earthquakes_4.loc[earthquakes_4['trace_name'] == test_trace_names[0], 'receiver_latitude']
# lat1 = receiver_latitude.values[0]
# print(f'lat1: {lat1}')
# receiver_long = earthquakes_4.loc[earthquakes_4['trace_name'] == test_trace_names[0], 'receiver_longitude']
# long1 = receiver_long.values[0]
# print(f'lat1: {long1}')

#-----------------------------------------------------------------------------------------------

# def haversine_distance(lat1, lon1, lat2, lon2):
#         """
#         Calculate the great-circle distance between two points 
#         on the Earth specified by their latitude and longitude.

#         Parameters:
#         lat1, lon1: Latitude and longitude of the first point in degrees.
#         lat2, lon2: Latitude and longitude of the second point in degrees.

#         Returns:
#         Distance in kilometers between the two points.
#         """
#         # Convert latitude and longitude from degrees to radians
#         lat1 = math.radians(lat1)
#         lon1 = math.radians(lon1)
#         lat2 = math.radians(lat2)
#         lon2 = math.radians(lon2)

#         # Haversine formula
#         dlat = lat2 - lat1
#         dlon = lon2 - lon1
#         a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
#         c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

#         # Radius of Earth in kilometers (mean radius)
#         R = 6371.0
#         return R * c
    
# def correlationScore( distance_between_stations, mag1,mag2,L , alpha  ):
#         """
#         L = is a characteristic length scale
#         alpha = scaling factor
#         """
       
#         # # Cross-correlation component
#         # C_ij = (1 / N) * np.sum((t_i - mu_i) * (t_j - mu_j))        #NEED TO CHANGE THIS!!!!!!!
        
#         # Spatial correlation component
#         S_ij = np.exp(-distance_between_stations / L)
        
#         # Magnitude dependency component
#         W_ij = np.exp(alpha * (mag1 - mag2))
        
#         # Combined correlation
#         correlation = S_ij * W_ij
#         # # Combined correlation
#         # correlation = C_ij * S_ij * W_ij
#         return correlation

# predicted_magnitudes = []
# # Open the file and read its contents
# with open('predicted_magnitudes.txt', 'r') as file:
#     for line in file:
#         # Use regular expression to find the number in the line
#         match = re.search(r'Predicted Magnitude: ([0-9.]+)', line)
#         if match:
#             # Extract the number and convert it to float
#             magnitude = float(match.group(1))
#             # Append the magnitude to the list
#             predicted_magnitudes.append(magnitude)
    
# mag1 = predicted_magnitudes[0]
# mag2 = predicted_magnitudes[1]
# print(f'magnitude1: {mag1}')
# print(f'magnitude2: {mag2}')


# lat = full_csv.loc[full_csv['trace_name'] == test_trace_names[0], 'receiver_latitude']
# receiver_latitude1 = lat.values[0]
# long = full_csv.loc[full_csv['trace_name'] == test_trace_names[0], 'receiver_longitude']
# receiver_longitude1 = long.values[0]
# lat = full_csv.loc[full_csv['trace_name'] == test_trace_names[1], 'receiver_latitude']
# receiver_latitude2 = lat.values[0]
# long = full_csv.loc[full_csv['trace_name'] == test_trace_names[1], 'receiver_longitude']
# receiver_longitude2 = long.values[0]

# print(f'lat1 is {receiver_latitude1} ,lon1 is {receiver_longitude1}, lat2 is {receiver_latitude2}, lon2 is {receiver_longitude2}')

# distance_between_stations = haversine_distance(receiver_latitude1,receiver_longitude1,receiver_latitude2,receiver_longitude2)
# correlation = correlationScore(distance_between_stations, mag1,mag2,L =50, alpha =0.1) # L = 50 km
# print(f'The correlation score is: {correlation}')