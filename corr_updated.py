import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import cv2
import math
from geopy.geocoders import Nominatim
import opencage
from opencage.geocoder import OpenCageGeocode
import re
from datetime import datetime, timedelta

############################     USER INPUT   ####################################

# path to stead dataset
data_dir = 'D:/bashok_srip/STEAD_dataset'
testing_dir = 'D:/bashok_srip/STEAD_dataset/images/testing_spectrograms'

api_key = 'afe9311b9a5d4b96877fb8a92eb048e3'

test_trace_names = ['PAH.NN_20151216135336_EV','PAH.NN_20151216142205_EV']
    #['KAN16.GS_20150424061553_EV','KAN16.GS_20151111113626_EV']
#######################################################################################

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

test_images = [
    f'{testing_dir}/{test_trace_names[0]}.png',
    f'{testing_dir}/{test_trace_names[1]}.png'
]

class Correlate:
    def __init__(self, mag1, p_sample_arrival1, s_sample_arrival1, mag2, p_sample_arrival2, s_sample_arrival2,trace_start_time1,trace_start_time2,api_key,trace_start_date1 ,trace_start_date2):
        self.mag1 = mag1
        self.p_sample_arrival1 = p_sample_arrival1
        self.s_sample_arrival1 = s_sample_arrival1
        self.start_date1 = trace_start_date1
        self.start_time1 = trace_start_time1
        
        self.mag2 = mag2
        self.p_sample_arrival2 = p_sample_arrival2
        self.s_sample_arrival2 = s_sample_arrival2
        self.start_date2 = trace_start_date2
        self.start_time2 = trace_start_time2

        self.distance_between_stations = 0
        self.distance_between_sources =0
        self.distance_station1_to_epicenter = 0
        self.distance_station2_to_epicenter = 0

        # Convert strings to datetime objects
        time1_obj = datetime.strptime(self.start_time1, '%H:%M:%S')
        time2_obj = datetime.strptime(self.start_time2, '%H:%M:%S')

        # Calculate the difference
        difference = abs(time2_obj - time1_obj)

        # Convert the difference to total hours
        self.difference_in_hours = difference.total_seconds() / 3600
        self.difference_in_minutes = difference.total_seconds() / 60
        self.difference_in_seconds = difference.total_seconds()
        # Convert strings to datetime objects
        date1_obj = datetime.strptime(self.start_date1, '%Y-%m-%d')
        date2_obj = datetime.strptime(self.start_date2, '%Y-%m-%d')

        # Calculate the difference
        difference = abs(date2_obj - date1_obj)
        self.days_gap = difference.days
        
        self.higher_magnitude = max(self.mag1, self.mag2)
        self.lower_magnitude = min(self.mag1, self.mag2)

        self.api_key = api_key

    
    def haversine_distance(self,lat1, lon1, lat2, lon2,source_latitude1,source_longitude1,source_latitude2,source_longitude2):
        """
        Calculate the great-circle distance between two points 
        on the Earth specified by their latitude and longitude.

        Parameters:
        lat1, lon1: Latitude and longitude of the first point in degrees.
        lat2, lon2: Latitude and longitude of the second point in degrees.

        Returns:
        Distance in kilometers between the two points.
        """
        # Convert latitude and longitude from degrees to radians
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        sourcelat1 = math.radians(source_latitude1)
        sourcelon1 = math.radians(source_longitude1)
        sourcelat2 = math.radians(source_latitude2)
        sourcelon2 = math.radians(source_longitude2)

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        dlat_source = sourcelat2 - sourcelat1
        dlon_source = sourcelon2 - sourcelon1
        source_a = math.sin(dlat_source / 2)**2 + math.cos(sourcelat1) * math.cos(sourcelat2) * math.sin(dlon_source / 2)**2
        source_c = 2 * math.atan2(math.sqrt(source_a), math.sqrt(1 - source_a))

        # Radius of Earth in kilometers (mean radius)
        R = 6371.0
        self.distance_between_stations = R * c
        self.distance_between_sources = R * source_c
        print(f'distance between stations is:{self.distance_between_stations}')
        print(f'distance between sources is:{self.distance_between_sources}')

    def __calculate_distance_to_epicenter__(self):  #check again
        """
        Estimate the distance from the earthquake epicenter to a seismic station using the difference in arrival times of P-wave and S-wave.

        Parameters:
        p_sample_arrival (float): Arrival time of the P-wave in seconds.
        s_sample_arrival (float): Arrival time of the S-wave in seconds.

        Returns:
        float: Estimated distance to the earthquake epicenter in kilometers.
        """
        time_difference1 = self.s_sample_arrival1 - self.p_sample_arrival1
        time_difference2 = self.s_sample_arrival2 - self.p_sample_arrival2
        if time_difference1 <= 0 or time_difference2 <= 0:
            raise ValueError("S-wave arrival time must be after P-wave arrival time.")
        # USING seismological emperical formula
        self.distance_station1_to_epicenter = 8 * time_difference1
        self.distance_station2_to_epicenter = 8 * time_difference2

        print(f'diatance from station1 to its epi is {self.distance_station1_to_epicenter}')
        print(f'diatance from station2 to its epi is {self.distance_station2_to_epicenter}')


    def are_earthquakes_related(self):
        """
        Determines if two earthquakes are related based on their magnitudes and the distance between stations.

        Parameters:
        mag1 (float): Magnitude measured at the first station.
        mag2 (float): Magnitude measured at the second station.
        distance (float): Distance between the two seismic stations in kilometers.

        Returns:
        bool: True if the earthquakes are related, False if they are unrelated.
        """
        higher_magnitude = max(self.mag1, self.mag2)
        
        if self.mag1 < 2.5 and self.mag2 < 2.5:
            if self.distance_between_sources > 800:
                return False
        elif 2.5 < higher_magnitude <= 4.0:
            if self.distance_between_sources > 1200:
                return False
        elif 4.0 < higher_magnitude <= 6.0:
            if self.distance_between_sources > 1600:
                return False
        elif 6.0 < higher_magnitude <= 8.0:
            if self.distance_between_sources > 2000:
                return False
        elif higher_magnitude > 8.0:
            if self.distance_between_sources > 3000:
                return False
        
        return True
    
    def earthquakes_days_gap(self):
        
        print(f'gap in days is:{self.days_gap}')
       
        mag_diff = abs(self.higher_magnitude - self.lower_magnitude)
        
        if self.mag1 < 2.5 and self.mag2 < 2.5:
            if self.difference_in_hours > 2:
                return False
        elif 2.5 < self.higher_magnitude <= 4.0:
            if self.days_gap > 1 :
                return False
        elif 4.0 < self.higher_magnitude < 5.0 and mag_diff < 1.5:
            if self.days_gap > 3:
                return False
        elif 5.0 <= self.higher_magnitude <= 6.0 and mag_diff < 3:
            if self.days_gap > 7:
                return False
        elif 6.0 < self.higher_magnitude <= 7.0 and mag_diff < 3:
            if self.days_gap > 15: 
                return False
        elif 7.0 < self.higher_magnitude <= 8.0 and mag_diff < 4:
            if self.days_gap > 30: 
                return False

    def are_earthquakes_related_by_epicenter(self):
        """
        Determines if two seismic signals are related based on their magnitudes and distances.

        Parameters:
        mag1 (float): Magnitude measured at the first station.
        mag2 (float): Magnitude measured at the second station.
        distance_between_stations (float): Distance between the two seismic stations in kilometers.
        distance_station1_to_epicenter (float): Distance from the first station to the epicenter in kilometers.
        distance_station2_to_epicenter (float): Distance from the second station to the epicenter in kilometers.

        Returns:
        bool: True if the seismic signals are related and have the same epicenter, False otherwise.
        """
        
        if (self.mag1 == self.mag2) and (self.distance_between_stations == (self.distance_station1_to_epicenter + self.distance_station2_to_epicenter)):
            return True
        # IMPORTANT ------ add a case where the same station records two earthquakes at different times and might be related by its aftershocks.
        else:
            return False
    
    def check_direct_interaction(self): 
        """
        Check if two earthquakes or their aftershocks interact based on arrival times.
        """
        if self.days_gap < 1:
            if self.mag1 <= 2.5 and self.mag2 <= 2.5:
                if self.difference_in_seconds < 20 :
                    return True
                else:
                    return False
            elif 2.5 < self.higher_magnitude <= 4.0:
                if self.difference_in_seconds < 50:
                    return True
                else:
                    return False
            elif 4.0 < self.higher_magnitude <= 6.0 :
                if self.difference_in_seconds <= 100:
                    return True
                else:
                    return False
            elif 6.0 < self.higher_magnitude <= 8.0:
                if self.difference_in_minutes <= 3:
                    return True
                else:
                    return False
            elif self.higher_magnitude > 8.0:
                if self.difference_in_minutes <= 6:
                    return True
                else:
                    return False
        else:
            return False
        
    def check_Mainshock_triggering(self):
        if self.days_gap < 1:
            if self.mag1 <= 2.5 and self.mag2 <= 2.5:
                return False
            elif 2.5 < self.higher_magnitude <= 4.0:
                if self.difference_in_seconds < 1500:
                    return True
                else:
                    return False
            elif 4.0 < self.higher_magnitude <= 6.0 :
                if self.difference_in_minutes <= 100:
                    return True
                else:
                    return False
            elif 6.0 < self.higher_magnitude <= 8.0:
                if self.difference_in_minutes <= 200:
                    return True
                else:
                    return False
            elif self.higher_magnitude > 8.0:
                if self.difference_in_minutes <= 300:
                    return True
                else:
                    return False
        else:
            return False
                
    def get_region_from_coordinates(self, latitude, longitude):
        # Implementation using self.api_key for API calls
        # For example, using OpenCage API
        
        geocoder = OpenCageGeocode(self.api_key)
        result = geocoder.reverse_geocode(latitude, longitude)
        # Extract region from result
        region = result[0]['components']['state']  # Adjust based on the actual response structure
        return region
    
    @staticmethod
    def get_attenuation_coefficient(region):
        """
        Get the attenuation coefficient for a given region.
        
        Parameters:
        region (str): Name of the region.
        
        Returns:
        float: Attenuation coefficient for the region.
        """
        # Example dictionary of regions and their attenuation coefficients
        attenuation_coefficients = {
            'California': 0.003,
            'Nevada': 0.004,
            'Alaska': 0.002,
            'Washington': 0.005,
            'Oregon': 0.004,
            'Utah': 0.006,
            'Hawaii': 0.007,
            'Texas': 0.003,
            'Florida': 0.001,
            'Missouri': 0.005,
            'Montana': 0.004,
            'Tennessee': 0.005,
            'Virginia': 0.002,
            'Kansas' : 0.0046 ,
            'Oklahoma': 0.0033,
            'South Carolina': 0.0037,
            'New York': 0.0031,
            'Massachusetts': 0.0030,
            'Idaho': 0.0039,
            'Wyoming': 0.0036,
            'Colorado': 0.0040,
            'New Mexico': 0.0043,
            'Arizona': 0.0042,
            'Puerto Rico': 0.02
            # Add more regions and their coefficients as needed
        }
        return attenuation_coefficients.get(region,0.003) #default value 

    def magnitude_at_distance(self,M0, alpha):  #let alpha be 0.05 for ex , but actually depends on geology of that region
        """
        Calculate the magnitude of an earthquake at a given distance considering attenuation.

        Parameters:
        M0: Magnitude at the epicenter
        r: Distance from the earthquake epicenter (in km)
        alpha: Attenuation coefficient

        Returns:
        Magnitude at the given distance.
        """
        # Initial amplitude (A0) corresponding to magnitude M0
        A0 = 10**M0
        
        # Amplitude at distance r considering attenuation
        A_r = A0 * np.exp(-alpha * self.distance_between_sources)
        
        # Convert amplitude back to magnitude
        M_r = np.log10(A_r)
        
        return M_r
    #===============================================================================================================
    def find_critical_stress_threshold(self,region):
        # Example dictionary of regions and their critical stress thresholds(in MPa)
        critical_thresholds = {
            'California': 3,
            'Nevada': 2,
            'Alaska': 2.4,
            'Washington': 2,
            'Oregon': 2,
            'Utah': 1.5,
            'Hawaii': 1,
            'New York': 0.75,
            'Texas': 0.75,
            'Florida': 0.5,
            'Missouri': 0.75,
            'Montana': 1.25,
            'Tennessee': 1.5,
            'South Carolina': 1.25,
            'Virginia': 0.75,
            'Arizona': 1.5,
            'Colorado': 1.2,
            'Illinois': 1,
            'Kansas': 0.9,
            'Kentucky': 1.1,
            'Michigan': 0.8,
            'Minnesota': 0.7,
            'Ohio': 0.85,
            'Oklahoma': 1.3,
            'Pennsylvania': 0.9,
            'Puerto Rico': 1.7
        }
        return critical_thresholds.get(region)

    def seismic_moment(magnitude):
        """ Calculate the seismic moment from magnitude. """
        return 10**(1.5 * magnitude + 9.1)

    def static_stress_change(self,magnitude, distance_km):
        """ Calculate the static stress change in MPa. """
        M0 = self.seismic_moment(magnitude)
        distance_m = distance_km * 1000  # Convert km to meters
        delta_sigma = M0 / (distance_m**3)  # Static stress change
        return delta_sigma / 1e6  # Convert to MPa

    def evaluate_triggering(self,magnitude_at_new_location, critical_threshold):
        """ Evaluate the impact based on static stress change. """
        delta_sigma = self.static_stress_change(magnitude_at_new_location, self.distance_between_stations)
        if delta_sigma >= critical_threshold:
            return "Mainshock of stronger earthquake has triggered the weaker earthquake", delta_sigma
        else:
            return "Minimal effect , no triggering", delta_sigma
    #========================================================================================================
    def calculate_largest_aftershock(self,mainshock_magnitude): #based on an emperical formula / bath's law
        """
        Calculate the magnitude of the largest aftershock based on the mainshock magnitude.
        
        Parameters:
        mainshock_magnitude (float): The magnitude of the mainshock.
        
        Returns:
        float: The magnitude of the largest aftershock.
        """
        if mainshock_magnitude < 1.5:
            if self.difference_in_hours <=1:
                aftershock_magnitude = mainshock_magnitude
                return aftershock_magnitude
            if 1 < self.difference_in_hours <=2:
                aftershock_magnitude = mainshock_magnitude -0.2
                return aftershock_magnitude
        
        if 1.5<= mainshock_magnitude < 2.5:
            if self.difference_in_hours <=2:
                aftershock_magnitude = mainshock_magnitude 
                return aftershock_magnitude
            if 2< self.difference_in_hours <=4:
                aftershock_magnitude = mainshock_magnitude - 0.3
                return aftershock_magnitude

        if 2.5<= mainshock_magnitude < 4:
            if self.difference_in_hours <=2:
                aftershock_magnitude = mainshock_magnitude 
                return aftershock_magnitude
            if 2< self.difference_in_hours <=4:
                aftershock_magnitude = mainshock_magnitude - 0.2
                return aftershock_magnitude
            if 4< self.difference_in_hours <=8:
                aftershock_magnitude = mainshock_magnitude - 1.2
                return aftershock_magnitude
            
        if 4<= mainshock_magnitude < 6:
            if self.difference_in_hours <=2:
                aftershock_magnitude = mainshock_magnitude 
                return aftershock_magnitude
            if 2< self.difference_in_hours <=4:
                aftershock_magnitude = mainshock_magnitude - 0.2
                return aftershock_magnitude
            if 4< self.difference_in_hours <=8:
                aftershock_magnitude = mainshock_magnitude - 1.2
                return aftershock_magnitude
            if 8< self.difference_in_hours <=12:
                aftershock_magnitude = mainshock_magnitude - 1.5
                return aftershock_magnitude
            if 12< self.difference_in_hours <=24:
                aftershock_magnitude = mainshock_magnitude - 1.8
                return aftershock_magnitude
            if 24< self.difference_in_hours <=48:
                aftershock_magnitude = mainshock_magnitude - 2.4
                return aftershock_magnitude
            if 48< self.difference_in_hours <=96:
                aftershock_magnitude = mainshock_magnitude - 4.0
                return aftershock_magnitude
            
        if 6 <= mainshock_magnitude <7 :
            if self.difference_in_hours <=2:
                aftershock_magnitude = mainshock_magnitude 
                return aftershock_magnitude
            if 2< self.difference_in_hours <=4:
                aftershock_magnitude = mainshock_magnitude - 0.2
                return aftershock_magnitude
            if 4< self.difference_in_hours <=8:
                aftershock_magnitude = mainshock_magnitude - 1.2
                return aftershock_magnitude
            if 8< self.difference_in_hours <=12:
                aftershock_magnitude = mainshock_magnitude - 1.5
                return aftershock_magnitude
            if 12< self.difference_in_hours <=24:
                aftershock_magnitude = mainshock_magnitude - 1.8
                return aftershock_magnitude
            if 24< self.difference_in_hours <=48:
                aftershock_magnitude = mainshock_magnitude - 2.4
                return aftershock_magnitude
            if 48< self.difference_in_hours <=96:
                aftershock_magnitude = mainshock_magnitude - 4.0
                return aftershock_magnitude
            if 96< self.difference_in_hours <=168:
                aftershock_magnitude = mainshock_magnitude - 5.0
                return aftershock_magnitude
            
        if 7 <= mainshock_magnitude :
            if self.difference_in_hours <=2:
                aftershock_magnitude = mainshock_magnitude 
                return aftershock_magnitude
            if 2< self.difference_in_hours <=4:
                aftershock_magnitude = mainshock_magnitude - 0.2
                return aftershock_magnitude
            if 4< self.difference_in_hours <=8:
                aftershock_magnitude = mainshock_magnitude - 1.2
                return aftershock_magnitude
            if 8< self.difference_in_hours <=12:
                aftershock_magnitude = mainshock_magnitude - 1.5
                return aftershock_magnitude
            if 12< self.difference_in_hours <=24:
                aftershock_magnitude = mainshock_magnitude - 1.8
                return aftershock_magnitude
            if 24< self.difference_in_hours <=48:
                aftershock_magnitude = mainshock_magnitude - 2.0
                return aftershock_magnitude
            if 48< self.difference_in_hours <=96:
                aftershock_magnitude = mainshock_magnitude - 2.8
                return aftershock_magnitude
            if 96< self.difference_in_hours <=168:
                aftershock_magnitude = mainshock_magnitude - 3.2
                return aftershock_magnitude
            if 7<self.days_gap <=14:
                aftershock_magnitude = mainshock_magnitude - 4.0
                return aftershock_magnitude
            if 14<self.days_gap <= 30:
                aftershock_magnitude = mainshock_magnitude - 4.8
                return aftershock_magnitude
            if 30<self.days_gap :
                aftershock_magnitude = mainshock_magnitude - 6.0
                return aftershock_magnitude
        
    #========================================================================================================
    def correlationScore(self , L, alpha):
        """
        t_i = Time of earthquakes in list1 
        t_j = Time of earthquakes in list2
        N = total no.of earthquakes
        L = is a characteristic length scale
        alpha = scaling factor
        """
       
        # # Cross-correlation component
        # C_ij = (1 / N) * np.sum((t_i - mu_i) * (t_j - mu_j))        #NEED TO CHANGE THIS!!!!!!!
        
        # Spatial correlation component
        S_ij = np.exp(-self.distance_between_stations / L)
        
        # Magnitude dependency component
        W_ij = np.exp(alpha * (self.mag1 - self.mag2))
        
        # Combined correlation
        correlation = S_ij * W_ij
        # # Combined correlation
        # correlation = C_ij * S_ij * W_ij
        return correlation
#---------------------------------------------------------------------------------------------------------------------------------------
######################___________________________________________MAIN_____________________________________________________############################################################################

## READING Trace-category,Magnitudes,p_sample_arrival and s_sample_arrival

with open('predicted_trace_categories.txt', 'r') as file:
    # Read all lines in the file
    lines = file.readlines()

# Extract only the labels (earthquake/noise)
labels = [line.split(': ')[1].strip() for line in lines]
is_earthquake1 = labels[0]
is_earthquake2 = labels[1]
print(is_earthquake1)
print(is_earthquake2)

# Initialize an empty list to store the predicted magnitudes
predicted_magnitudes = []
# Open the file and read its contents
with open('predicted_magnitudes.txt', 'r') as file:
    for line in file:
        # Use regular expression to find the number in the line
        match = re.search(r'Predicted Magnitude: ([0-9.]+)', line)
        if match:
            # Extract the number and convert it to float
            magnitude = float(match.group(1))
            # Append the magnitude to the list
            predicted_magnitudes.append(magnitude)
    
mag1 = predicted_magnitudes[0]
mag2 = predicted_magnitudes[1]
print(f'magnitude1: {mag1}')
print(f'magnitude2: {mag2}')

# Read the file and extract the pred[0] values
pred_values = []
with open('p_arrival_sample.txt', 'r') as file:
    for line in file:
        # Split the line to extract the predicted value
        parts = line.split(': Predicted p_arrival_sample: ')
        if len(parts) == 2:
            # Extract the value and convert it to a float
            pred_value = float(parts[1].strip())
            pred_values.append(pred_value)

p_sample_arrival1 = pred_values[0]
p_sample_arrival2 = pred_values[1]
# print(f'p_sample1:{p_sample_arrival1}')
# print(f'p_sample2:{p_sample_arrival2}')

# Read the file and extract the pred[0] values
pred_values = []
with open('s_arrival_sample.txt', 'r') as file:
    for line in file:
        # Split the line to extract the predicted value
        parts = line.split(': Predicted s_arrival_sample: ')
        if len(parts) == 2:
            # Extract the value and convert it to a float
            pred_value = float(parts[1].strip())
            pred_values.append(pred_value)

s_sample_arrival1 = pred_values[0]
s_sample_arrival2 = pred_values[1]
# print(f's_sample1:{s_sample_arrival1}')
# print(f's_sample2:{s_sample_arrival2}')
#-----------------------------------------------------------------------------------------------------
#Finding trace start time1

trace_start_times1 = full_csv.loc[full_csv['trace_name'] == test_trace_names[0], 'trace_start_time']
trace_start_times2 = full_csv.loc[full_csv['trace_name'] == test_trace_names[1], 'trace_start_time']
datetime_value1 = trace_start_times1.values[0]
datetime_value2 = trace_start_times2.values[0]
date_time_parts_str1 = datetime_value1.split()
date_time_parts_str2 = datetime_value2.split()

if len(date_time_parts_str1) > 1:
    date_str1 = date_time_parts_str1[0]

    time_str1 = date_time_parts_str1[1]
    time_str_split1 = time_str1.split('.')
    time_str_rounded_down1 = time_str_split1[0]

    trace_start_date1 = date_str1
    trace_start_time1 = time_str_rounded_down1
    print(f'start date of eq1: {trace_start_date1}')
    print(f'start time of eq1: {trace_start_time1}')

#Finding trace start time2
if len(date_time_parts_str2) > 1:
    date_str2 = date_time_parts_str2[0]

    time_str2 = date_time_parts_str2[1]
    time_str_split2 = time_str2.split('.')
    time_str_rounded_down2 = time_str_split2[0]

    trace_start_date2 = date_str2
    trace_start_time2 = time_str_rounded_down2
    print(f'start date of eq2: {trace_start_date2}')
    print(f'start time of eq2: {trace_start_time2}')

#-----------------------------------------------------------------------------------------------------
#Finding lat and long of receiver station
lat = full_csv.loc[full_csv['trace_name'] == test_trace_names[0], 'receiver_latitude']
receiver_latitude1 = lat.values[0]
long = full_csv.loc[full_csv['trace_name'] == test_trace_names[0], 'receiver_longitude']
receiver_longitude1 = long.values[0]
lat = full_csv.loc[full_csv['trace_name'] == test_trace_names[1], 'receiver_latitude']
receiver_latitude2 = lat.values[0]
long = full_csv.loc[full_csv['trace_name'] == test_trace_names[1], 'receiver_longitude']
receiver_longitude2 = long.values[0]

#Finding lat and long of source
lat = full_csv.loc[full_csv['trace_name'] == test_trace_names[0], 'source_latitude']
source_latitude1 = lat.values[0]
long = full_csv.loc[full_csv['trace_name'] == test_trace_names[0], 'source_longitude']
source_longitude1 = long.values[0]
lat = full_csv.loc[full_csv['trace_name'] == test_trace_names[1], 'source_latitude']
source_latitude2 = lat.values[0]
long = full_csv.loc[full_csv['trace_name'] == test_trace_names[1], 'source_longitude']
source_longitude2 = long.values[0]

print(f'lat1 is {receiver_latitude1} ,lon1 is {receiver_longitude1}, lat2 is {receiver_latitude2}, lon2 is {receiver_longitude2}')
print(f'sourcelat1 is {source_latitude1} ,sourcelon1 is {source_longitude1}, sourcelat2 is {source_latitude2}, sourcelon2 is {source_longitude2}')


################################################################################################################################################################################################################
if is_earthquake1 == "earthquake" and is_earthquake2 == 'earthquake':
    Correlation_effects = Correlate(mag1,p_sample_arrival1,s_sample_arrival1,mag2,p_sample_arrival2,s_sample_arrival2,trace_start_time1,trace_start_time2,api_key,trace_start_date1,trace_start_date2)
    Correlation_effects.haversine_distance(receiver_latitude1,receiver_longitude1,receiver_latitude2,receiver_longitude2,source_latitude1,source_longitude1,source_latitude2,source_longitude2)  # calculating distance between both reciever stations

    #step1:
    are_earthquakes_unrelated_by_distance = Correlation_effects.are_earthquakes_related() # checks if two earthquakes are unrelated based on their magnitudes and the distance between stations.
    #print(f'the variable are_earthquakes_unrelated_by_distance is :{are_earthquakes_unrelated_by_distance}')
    if(are_earthquakes_unrelated_by_distance == False):
        print("The two earthquakes are independent and dont effect each other since they are too far from each other")
        exit(0)
    else:
        are_earthquakes_unrelated_by_time = Correlation_effects.earthquakes_days_gap() # checks if two earthquakes are unrelated based on the time gap between the first occurence of them.
        #print(f'the variable are_earthquakes_unrelated_by_time is :{are_earthquakes_unrelated_by_time}')
        if(are_earthquakes_unrelated_by_time == False):
            print("The two earthquakes are independent and dont directly effect each other ,by days,")

    #Step2:
    are_earthquakes_related_by_same_epicenter = Correlation_effects.are_earthquakes_related_by_epicenter() #checks if two earthquakes(that have occured around the same time) have the same epicenter
    if(are_earthquakes_related_by_same_epicenter == True):
        print("The two earthquake signals are from the same epicenter")
        exit(0)
    else:
        print("The two earthquake signals are not from the same epicenter")

    #step3: first we check if the stronger earthquake has in fact triggered the weaker earthquake, then we check if the MAIN or AFTER shockwaves are related to the weaker earthquake
    higher_magnitude = max(mag1, mag2)
    lower_magnitude = min(mag1,mag2)

    region1 = Correlation_effects.get_region_from_coordinates(receiver_latitude1,receiver_longitude1)
    print(f'the region of {test_trace_names[0]} is {region1}')
    region2 = Correlation_effects.get_region_from_coordinates(receiver_latitude2,receiver_longitude2)
    print(f'the region of {test_trace_names[1]} is {region2}')

    if(lower_magnitude == mag1):
        alpha = Correlation_effects.get_attenuation_coefficient(region1)
    else:
        alpha = Correlation_effects.get_attenuation_coefficient(region2)

    do_they_interact = Correlation_effects.check_direct_interaction()
    print(f'the value of do_they_interact is: {do_they_interact}')
    if(do_they_interact == True):
        M_r = Correlation_effects.magnitude_at_distance(higher_magnitude, alpha )  # Calculate the Mainshock magnitude at the given distance
        if ( M_r < lower_magnitude ):
            print("Even though the two earthquakes occur in reasonable distances they are independent and dont effect each other")
        else:
            print(f"The two earthquakes are related and the MAIN shockwaves of {higher_magnitude} effects {lower_magnitude}")
            correlation = Correlation_effects.correlationScore( L = 50, alpha = 0.1)
            print("Correlation Score is", correlation)
            if(correlation > 0.8):
                print("the two earthquakes are well-related")
            if(correlation > 1):
                print("the two earthquakes are extremely_closely-related")
    #Then maybe Mainshcok or aftershock triggering
    if(do_they_interact == False):  
        #first check for main shock triggering
        is_MainTriggering_possible = Correlation_effects.check_Mainshock_triggering()
        print(f'value of is_MainTriggering_possible : {is_MainTriggering_possible}')
        if (is_MainTriggering_possible == True):
            M_r = Correlation_effects.magnitude_at_distance(higher_magnitude, alpha )
            if (lower_magnitude == mag1):
                critical_threshold = Correlation_effects.find_critical_stress_threshold(region1)
            else:
                critical_threshold = Correlation_effects.find_critical_stress_threshold(region2)
            impact, delta_sigma = Correlation_effects.evaluate_triggering(M_r, critical_threshold)
        else:
            #then check for aftershock effects.
            aftershock  = Correlation_effects.calculate_largest_aftershock(higher_magnitude)
            M_aftershock_r = Correlation_effects.magnitude_at_distance(aftershock,alpha)  # Calculate the aftershock magnitude at the given distance
            print(f'the magnitude of the aftershock of the stronger eq ,AFTER travelling is: {M_aftershock_r}')
            if ( M_aftershock_r < lower_magnitude ):
                print("Even though the two earthquakes occur in reasonable distances they are independent and dont effect each other because the aftershock isnt strong enough to trigger a new earthquake")
            else:
                print(f"The two earthquakes are related and the AFTER shockwaves of {higher_magnitude} triggers {lower_magnitude}")
                correlation = Correlation_effects.correlationScore( L = 50, alpha = 0.1)
                print("Correlation Score:", correlation)
else:
    print("Noise cant be related")
