import numpy as np
import math
import random
from geopy.geocoders import Nominatim
import opencage
from opencage.geocoder import OpenCageGeocode
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import DBSCAN
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import GridSearchCV

####################################################################################################################################
            ##################            INPUT            ########################
#choose the values of the below for forecasting

#5)Wangjing earthquake
source_magnitude = 5.0
source_latitude = 24.622
source_longitude =  95.093

api_key = 'afe9311b9a5d4b96877fb8a92eb048e3'
####################################################################################################################################
class EQprediction:
    def __init__(self, source_magnitude,source_latitude, source_longitude,eq_prone_latitudes, eq_prone_longitudes,api_key):
        self.mag = source_magnitude
        self.lat = source_latitude
        self.long = source_longitude
        self.api_key = api_key
        self.eq_prone_latitudes = eq_prone_latitudes
        self.eq_prone_longitudes = eq_prone_longitudes

        self.predicted_latsNlongs = []

        self.distances = []
        self.higher_than_threshold = []
        self.magnitudes_at_distances = []  # List to store magnitudes after attenuation
        self.list_of_alphas = []  # List to store attenuation coefficients for each region
        self.sorted_regions_of_prone_eqs = []
        self.difference_in_pressure = []

        
        self.X_test = []

        
        
    #==========================================================================================================================
    def get_wave_velocities(self,region_name):
        # Define a dictionary with P-wave and S-wave velocities for different regions
        wave_velocities = {
            'California': {'Vp': 6.0, 'Vs': 3.5},
            'Nevada': {'Vp': 6.4, 'Vs': 4.0},
            'Oregon': {'Vp': 5.8, 'Vs': 3.3},
            'Washington': {'Vp': 6.1, 'Vs': 3.4},
            'Texas': {'Vp': 6.3, 'Vs': 3.7},
            'Utah': {'Vp': 6.0, 'Vs': 3.5},
            'Hawaii': {'Vp': 6.0, 'Vs': 3.4},
            'Florida': {'Vp': 5.5, 'Vs': 3.2},
            'Missouri': {'Vp': 6.2, 'Vs': 3.6},
            'Montana': {'Vp': 6.1, 'Vs': 3.5},
            'Tennessee': {'Vp': 6.0, 'Vs': 3.5},
            'Virginia': {'Vp': 6.0, 'Vs': 3.5},
            'Kansas': {'Vp': 6.0, 'Vs': 3.5},
            'Oklahoma': {'Vp': 6.0, 'Vs': 3.5},
            'South Carolina': {'Vp': 5.8, 'Vs': 3.3},
            'New York': {'Vp': 5.7, 'Vs': 3.3},
            'Massachusetts': {'Vp': 5.6, 'Vs': 3.2},
            'Idaho': {'Vp': 6.2, 'Vs': 3.6},
            'Wyoming': {'Vp': 6.2, 'Vs': 3.6},
            'Colorado': {'Vp': 6.1, 'Vs': 3.5},
            'New Mexico': {'Vp': 6.0, 'Vs': 3.5},
            'Arizona': {'Vp': 6.1, 'Vs': 3.4},
            'Puerto Rico': {'Vp': 5.9, 'Vs': 3.3},
            'Alaska': {'Vp': 6.4, 'Vs': 3.9},
            'Piedmont': {'Vp': 5.9, 'Vs': 3.4},
            'Catalonia': {'Vp': 5.9, 'Vs': 3.2},
            'North Rhine-Westphalia': {'Vp': 5.9, 'Vs': 3.3},
            'Calabria': {'Vp': 5.9, 'Vs': 3.4},
            'British Columbia': {'Vp': 6.2, 'Vs': 3.5},
            'Autonomous Republic of Adjara': {'Vp': 5.9, 'Vs': 3.4},
            'La Unión' : {'Vp': 5.7, 'Vs':3.3},
            'Nuble Region': {'Vp': 6.2, 'Vs': 3.8},
            "O'Higgins Region": {'Vp': 6.5, 'Vs': 3.7},
            'Marche': {'Vp': 6.0, 'Vs': 3.5},
            'Lazio': {'Vp': 5.8, 'Vs': 3.4},
            'Tuscany': {'Vp': 5.9, 'Vs': 3.4}
            # Add more regions as needed
        }

        # Look up the velocities for the given region name
        wave_velocities = wave_velocities.get(region_name, None)
        #wave_velocities2 = wave_velocities.get(region_name2, None)
        
        Vp  = wave_velocities['Vp']
        Vs = wave_velocities['Vs']
        return Vp,Vs
        # self.Vp2  = wave_velocities2['Vp']
        # self.Vs2 = wave_velocities2['Vs']

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
            'Puerto Rico': 0.02,
            'Piedmont': 0.0025,
            'Catalonia': 0.005,
            'North Rhine-Westphalia': 0.0035,
            'Calabria': 0.0035,
            'British Columbia': 0.0035,
            'Autonomous Republic of Adjara': 0.0035,
            'La Unión' : 0.0025,
            'Nuble Region': 0.0025,
            "O'Higgins Region": 0.0035,
            'Marche': 0.0035 ,
            'Lazio':0.0035,
            'Tuscany': 0.0025
            # Add more regions and their coefficients as needed
        }
        return attenuation_coefficients.get(region,0.003) #default value
    def get_region_from_coordinates(self ):
        """
        Fetch the region for each earthquake-prone location and store it in regions_of_prone_eqs[].
        """
        # Initialize the geocoder with the API key
        geocoder = OpenCageGeocode(self.api_key)

        # Clear the list before adding new regions
        regions_of_prone_eqs = []

        for lat, lon in zip(self.eq_prone_latitudes, self.eq_prone_longitudes):
            result = geocoder.reverse_geocode(lat, lon)

            # Extract region from the geocoding result
            if result and 'components' in result[0] and 'state' in result[0]['components']:
                region = result[0]['components']['state']
            else:
                region = "Unknown"  # Fallback if region is not found

            # Store the region in the list
            regions_of_prone_eqs.append(region)
        return regions_of_prone_eqs

    #==========================================================================================================================

    def calculate_distances(self):
        """
        Calculate and sort the distances from the source location to earthquake-prone regions,
        and maintain the order of regions and distances.
        """
        self.distances = []  # Clear previous results
        region_distance_pairs = []  # To hold (region, distance) tuples

        # Convert source latitude and longitude from degrees to radians
        source_lat_rad = math.radians(self.lat)
        source_lon_rad = math.radians(self.long)

        R = 6371.0  # Radius of the Earth in kilometers

        for lat2, lon2 in zip(self.eq_prone_latitudes, self.eq_prone_longitudes):
            lat2_rad = math.radians(lat2)
            lon2_rad = math.radians(lon2)

            # Haversine formula
            dlat = lat2_rad - source_lat_rad
            dlon = lon2_rad - source_lon_rad

            a = math.sin(dlat / 2)**2 + math.cos(source_lat_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            distance = R * c  # Distance in kilometers
            self.distances.append(distance)

        # Get the regions in the same order as latitudes and longitudes
        regions_of_prone_eqs = self.get_region_from_coordinates()

        # Pair the regions with their respective distances
        distance_region_lat_long_tuple = list(zip( self.distances,regions_of_prone_eqs,self.eq_prone_latitudes,self.eq_prone_longitudes))

        # Sort by distance (ascending)
        distance_region_lat_long_tuple.sort(key=lambda x: x[0])

        # Unpack sorted pairs into regions and distances
        self.distances ,self.sorted_regions_of_prone_eqs,self.eq_prone_latitudes,self.eq_prone_longitudes = zip(*distance_region_lat_long_tuple)

        # Get the attenuation coefficients for the sorted regions and store in the same order
        self.list_of_alphas = [self.get_attenuation_coefficient(region) for region in self.sorted_regions_of_prone_eqs]

        # # Write the regions to the test_eq_region.txt file in the sorted order
        # with open('test_eq_region.txt', 'w') as file:
        #     file.write(' '.join(regions_of_prone_eqs))

    #=========================================================================================================================  
     
    def calc_magnitude_at_distance(self): #change it to a list_of_alphas 
        """
        Calculate the magnitude of an earthquake at given distances considering attenuation.

        Parameters:
        alpha: Attenuation coefficient (e.g., 0.05)

        Returns:
        List of magnitudes at the given distances (self.distances).
        """
        # Clear previous magnitudes list
        self.magnitudes_at_distances = []

        # Initial amplitude (A0) corresponding to magnitude M0
        A0 = 10**self.mag

        # Loop through distances and corresponding alphas simultaneously
        for distance, alpha in zip(self.distances, self.list_of_alphas):
            # Amplitude at distance considering attenuation
            A_r = A0 * np.exp(-alpha * distance)

            # Convert amplitude back to magnitude
            M_r = np.log10(A_r)

            # Append the result to the magnitudes list
            self.magnitudes_at_distances.append(M_r)

        return 
############################################################################################################################################
    #########################               MAINSHOCK PREDICTION                     #########################
    def seismic_moment(self, magnitude):
        """Calculate the seismic moment from magnitude."""
        return 10**(1.5 * magnitude + 9.1)

    def static_stress_change(self, magnitude, distance_km):
        """Calculate the static stress change in MPa."""
        M0 = self.seismic_moment(magnitude)
        distance_m = distance_km * 1000  # Convert km to meters
        delta_sigma = M0 / (distance_m**3)  # Static stress change
        return delta_sigma/1e6  # Convert to MPa
    #==========================================================================================================================
    def evaluate_Mainshock_triggering(self, eq_prone_crits):
        """Evaluate the triggering potential for each earthquake-prone region."""
        self.higher_than_threshold = []  # Clear previous results
        count =1
        
        # Iterate through distances, magnitudes, and critical thresholds simultaneously
        for distance, critical_threshold ,lat,long in zip(self.distances, eq_prone_crits,self.eq_prone_latitudes ,self.eq_prone_longitudes):
            # Calculate static stress change using the distance and magnitude at that distance
            delta_sigma = self.static_stress_change(self.mag, distance)
            
            
            # Evaluate if the static stress change exceeds the critical threshold
            if delta_sigma >= critical_threshold:
                print(f'the value of Mainshock delta_sigma of {count} with distance: {distance} is {delta_sigma}')
                self.higher_than_threshold.append(True)
                pressure_diff = delta_sigma-critical_threshold
                self.difference_in_pressure.append(pressure_diff)

                self.X_test.append([distance, self.mag, pressure_diff])
                self.predicted_latsNlongs.append([lat,long])
            else:
                self.higher_than_threshold.append(False)
            count+=1
        #return self.higher_than_threshold
    #==========================================================================================================================
    def train_model(self):
        data_array = []
                
        with open("inputsForEq_predictions/mainshock_trainingdata.txt", "r") as file:
            for line in file:
                # Split the line into three numbers and convert them to integers
                subarray = list(map(float, line.split()))
                data_array.append(subarray)
        
        # Convert to numpy array for ease of indexing and operations
        data = np.array(data_array)
        X = data[:, [0, 1, 2]]
        Y = data[:, [3]]
        Y = data[:, [3]].ravel()

        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
        }


        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42), param_grid, cv=3, scoring="neg_mean_squared_error"
        )
        grid_search.fit(X_train, Y_train)
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, Y_train)
        
        # Predict on the validation set and print the RMSE
        Y_pred = best_model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(Y_val, Y_pred))
        print(f"Validation RMSE: {rmse}")
        return best_model
    
    def predictMainshock_earthquake(self, Mainshock_model):
        # for is_higher,region in zip(self.higher_than_threshold,self.sorted_regions_of_prone_eqs):
        #     if is_higher == True:
        #         # Get wave velocities for the region
        #         velocities = self.get_wave_velocities(region)
        #         Vp = velocities[0]  # P-wave velocity (km/s)
        #         Vs = velocities[1]  # S-wave velocity (km/s)
        #         # Calculate the time taken for P-wave and S-wave to travel the distance
        #         tp = self.distances / Vp  # Time for P-wave
        #         ts = self.distances / Vs  # Time for S-wave
        #         trigger_time = ts  # Assume triggering happens after S-wave arrival
        #         print(f'time taken to trigger earthquake by Mainshock: {trigger_time}')
                #use a model to predict the new earthquake at the prone location - i need distance,pressure_diff,eq at dist and eq at the source
                #write a model to predict the magnitude of the triggered earthquake given txt file containing distances(distance between the sources) *space* 
                    #magnitudes_at_distances(magnitude of the original earthquake after travelling) *space* pressure_diff *space* weaker earthquake for training
                    #and self.distances(distance between the sources), self.magnitudes_at_distances(magnitude of the original earthquake after travelling),pressure_diff for predicting.
        try:
            if len(self.X_test) == 0:
                raise ValueError("X_test is empty. Please provide valid test data.")
            else:
                # Perform the prediction
                raw_predictions = Mainshock_model.predict(self.X_test)
                
                # Add variability to predictions based on distance from the source earthquake
                # Scale predictions by proximity
                distances = np.array(self.distances)  # Distances of points from the source
                #########################################################
                if raw_predictions.shape != distances.shape:
                    min_len = min(len(raw_predictions), len(distances))
                    raw_predictions = raw_predictions[:min_len]
                    distances = distances[:min_len]
                #########################################################
                # Scale predictions inversely with distance using a nonlinear function
                max_distance = 100.0  # Maximum distance to normalize (e.g., 100 km)
                normalized_distances = distances / max_distance  # Normalize distances to [0, 1]
                scaled_predictions = raw_predictions * (1 - np.tanh(normalized_distances))  # Use a tanh-based scaling

                # Add proportional Gaussian noise for realistic variation
                noise = np.random.normal(0, 0.15 * raw_predictions.std(), size=scaled_predictions.shape)  # StdDev is 15% of prediction std dev
                varied_predictions = scaled_predictions + noise

                # Dynamically calculate realistic bounds based on prediction characteristics
                min_magnitude = max(2.5, np.percentile(varied_predictions, 5))  # Ensure minimum bound is at least 2.5
                max_magnitude = min(8.0, np.percentile(varied_predictions, 95))  # Ensure maximum bound is at most 8.0

                # Ensure predictions are within dynamic bounds
                varied_predictions = np.clip(varied_predictions, min_magnitude, max_magnitude)

                # Debugging Step: Print prediction statistics
                print("Prediction Stats - Min:", np.min(varied_predictions), "Max:", np.max(varied_predictions),
                    "Mean:", np.mean(varied_predictions), "StdDev:", np.std(varied_predictions))

                # Extract latitudes and longitudes from predicted points
                latitudes = [entry[0] for entry in self.predicted_latsNlongs]
                latitudes = np.array(latitudes)
                longitudes = [entry[1] for entry in self.predicted_latsNlongs]
                longitudes = np.array(longitudes)

                # Combine predictions into one array for clustering
                earthquake_data = np.column_stack((latitudes, longitudes, varied_predictions))

                # Normalize latitude and longitude relative to the source earthquake for clustering
                earthquake_data[:, 0] -= self.lat
                earthquake_data[:, 1] -= self.long

                # Apply DBSCAN
                clustering = DBSCAN(eps=0.5, min_samples=2).fit(earthquake_data[:, :2])
                labels = clustering.labels_  # -1 indicates outliers

                # Filter predictions to exclude outliers
                filtered_predictions = earthquake_data[labels != -1]

                # Convert back to actual latitude and longitude
                filtered_predictions[:, 0] += self.lat
                filtered_predictions[:, 1] += self.long


                # Define the output file path
                output_file = "predicted_Mainshockmagnitudes_wangjing.txt"

                # Write predictions to the text file
                with open(output_file, "w") as file:
                    for prediction, distance in zip(filtered_predictions, distances[labels != -1]):
                        file.write(f"{prediction} {distance}km\n")

                print(f"Predictions have been saved to {output_file}")
        except NotFittedError:
            print("The model is not fitted yet. Make sure to call 'train_model()' before making predictions.")



    # def predictMainshock_earthquake(self,Mainshock_model):
    #     # for is_higher,region in zip(self.higher_than_threshold,self.sorted_regions_of_prone_eqs):
    #     #     if is_higher == True:
    #     #         # Get wave velocities for the region
    #     #         velocities = self.get_wave_velocities(region)
    #     #         Vp = velocities[0]  # P-wave velocity (km/s)
    #     #         Vs = velocities[1]  # S-wave velocity (km/s)
    #     #         # Calculate the time taken for P-wave and S-wave to travel the distance
    #     #         tp = self.distances / Vp  # Time for P-wave
    #     #         ts = self.distances / Vs  # Time for S-wave
    #     #         trigger_time = ts  # Assume triggering happens after S-wave arrival
    #     #         print(f'time taken to trigger earthquake by Mainshock: {trigger_time}')
    #             #use a model to predict the new earthquake at the prone location - i need distance,pressure_diff,eq at dist and eq at the source
    #             #write a model to predict the magnitude of the triggered earthquake given txt file containing distances(distance between the sources) *space* 
    #                 #magnitudes_at_distances(magnitude of the original earthquake after travelling) *space* pressure_diff *space* weaker earthquake for training
    #                 #and self.distances(distance between the sources), self.magnitudes_at_distances(magnitude of the original earthquake after travelling),pressure_diff for predicting.  
    #     try:
    #         if len(self.X_test) == 0:
    #             raise ValueError("X_test is empty. Please provide valid test data.")
    #         else:
    #             # Perform the prediction
    #             predictions = Mainshock_model.predict(self.X_test)

    #             # max_distance_km = 500
    #             # dist_filtered_predictions = predictions[
    #             #     predictions["distance"] <= max_distance_km
    #             # ]
    #             latitudes = [entry[0] for entry in self.predicted_latsNlongs]
    #             latitudes = np.array(latitudes)  # Corresponding latitudes
    #             longitudes = [entry[1] for entry in self.predicted_latsNlongs]
    #             longitudes = np.array(longitudes)


    #         ###################################     FILTERING         #################################
    #             # Combine predictions into one array for clustering
    #             earthquake_data = np.column_stack((latitudes, longitudes, predictions))

    #             # Normalize latitude and longitude relative to the source earthquake for clustering
    #             earthquake_data[:, 0] -= self.lat
    #             earthquake_data[:, 1] -= self.long

    #             # Apply DBSCAN
    #             clustering = DBSCAN(eps=0.5, min_samples=2).fit(earthquake_data[:, :2])  # Clustering on normalized lat/lon
    #             labels = clustering.labels_  # -1 indicates outliers

    #             # Filter predictions to exclude outliers
    #             filtered_predictions = earthquake_data[labels != -1]  # Keep only clustered points (label != -1)

    #             # Convert back to actual latitude and longitude
    #             filtered_predictions[:, 0] += self.lat
    #             filtered_predictions[:, 1] += self.long
    #         ###################################################################################################
    #             # Define the output file path
    #             output_file = "predicted_Mainshockmagnitudes.txt"

    #             # Write predictions to the text file
                
    #             with open(output_file, "w") as file:
    #                 for prediction,distance in zip(filtered_predictions,self.distances):
    #                     file.write(f"{prediction} {distance}km\n")

    #             print(f"Predictions have been saved to {output_file}")
    #     except NotFittedError:
    #         print("The model is not fitted yet. Make sure to call 'train_model()' before making predictions.")

    
############################################################################################################################################
    #########################               AFTERSHOCK PREDICTION                     #########################
    def aftershock_magnitudes(self, alpha = 1.2): #changed alpha from 1.5 to 1.2
        """
        Calculate the magnitude of the aftershock at a given time t.

        Parameters:
        mainshock_magnitude (float): The magnitude of the mainshock.
        alpha (float): The decay constant for the region.
        time (float or numpy array): The time after the mainshock (in hours, days, etc.).

        Returns:
        float or numpy array: The calculated aftershock magnitude.
        """
        if self.mag < 2.5:
            # add cases for values of difference_in_minutes
            aftershock1_mag =  self.mag - alpha * np.log(random.randint(10, 30)) #after 10 mins
        else:
            aftershock1_mag =  self.mag - 1.2 # immediately (1hr)
            aftershock2_mag =  self.mag - (1.2+alpha * np.log(random.randint(6, 10))) # between 6-10 hours
            aftershock3_mag =  self.mag - (1.2+alpha * np.log(random.randint(24, 48))) #between 1-2 days
            aftershock4_mag =  self.mag - (1.2+alpha * np.log(random.randint(96, 120))) #after 4 days
            aftershock5_mag = self.mag - (1.2+alpha * np.log(random.randint(720, 1000))) #after 1 month

        if (aftershock1_mag > 0):
            #fininsh this
            if self.mag <= 2.5:
                return aftershock1_mag
            if 2.5 < self.mag and self.mag <= 4 :
                return aftershock1_mag,aftershock2_mag
            if 4 < self.mag and self.mag <= 5 :
                return aftershock1_mag,aftershock2_mag,aftershock3_mag
            if 5 < self.mag and self.mag <= 7 :
                return aftershock1_mag,aftershock2_mag,aftershock3_mag,aftershock4_mag
            if 7 < self.mag :
                return aftershock1_mag,aftershock2_mag,aftershock3_mag,aftershock4_mag,aftershock5_mag
        else:
            print("no aftershock caused")

    #write a model to predict the magnitude of the triggered earthquake and after how much time does it trigger (take care of the scenario if it gets triggered multiple times)
    #given txt file containing distances(distance between the sources) *space* magnitude of stronger eq *space* magnitude of the weaker earthquake *space*
    # magnitude of the aftershock *space* time gap between the original eq and the eq triggered by aftershock for training and self.distances(distance between the sources),
    # magnitude of stronger eq(at source only) for testing and magnitude of aftershocks (figure it out here itself) to predict time gap between the original eq and the eq triggered by aftershock
    # and magnitude of the triggered earthquake.
    #plan A - use VAE or GAN for generating distributions of aftershocks/time gap and then use a RandomForest(RF) regressor to get time gap and mag.
    #plan B -for prediction, to get aftershock magnitude we do (mainshock mag - rand()) for each scenario

    #_____create a function in correlation.py to calculate the magnitude of aftershock using the logarithmic decay________
        
        
    def load_data(self):
        data_array = []
        with open("inputsForEq_predictions/aftershock_trainingdata.txt", "r") as file:
            for line in file:
                subarray = list(map(float, line.split()))
                data_array.append(subarray)
        data = np.array(data_array)
        x = data[:, [0, 1 ,2,3]]  
        #y = data[:, [4, 5]]
        # Separate targets
        y_hours = data[:, 4]       # Column 4: Difference in hours
        y_aftershock = data[:, 5]  # Column 5: Magnitude of the triggered earthquake
        return x, y_hours ,y_aftershock
    
    def split_data(self,x, y):
        return train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Regressor
    def train_random_forest(self,X_train, y_train):
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        return rf_model
    ########################################################
    # Train Gradient Boosting Regressor
    def train_gradient_boosting(self, X_train, y_train_lat, y_train_lon):
        """
        Train separate Gradient Boosting models for latitude and longitude.
        """
        # Train model for latitude
        gb_model_lat = GradientBoostingRegressor(random_state=42)
        gb_model_lat.fit(X_train, y_train_lat)

        # Train model for longitude
        gb_model_lon = GradientBoostingRegressor(random_state=42)
        gb_model_lon.fit(X_train, y_train_lon)

        return gb_model_lat, gb_model_lon

    
    ########################################################

    # Train Neural Network
    def train_neural_networks(self, X_train, y_train_hours, y_train_aftershock):
        input_dim = X_train.shape[1]  # Number of features in X_train
        
        # Model for predicting 'y_hours'
        model_hours = Sequential()
        model_hours.add(Dense(64, input_dim=input_dim, activation='relu'))
        model_hours.add(Dense(32, activation='relu'))
        model_hours.add(Dense(1, activation='linear'))  # Output layer for a single value
        model_hours.compile(optimizer='adam', loss='mean_squared_error')
        print("Training model for 'y_hours'...")
        model_hours.fit(X_train, y_train_hours, epochs=50, batch_size=32, verbose=1)
        
        # Model for predicting 'y_aftershock'
        model_aftershock = Sequential()
        model_aftershock.add(Dense(64, input_dim=input_dim, activation='relu'))
        model_aftershock.add(Dense(32, activation='relu'))
        model_aftershock.add(Dense(1, activation='linear'))  # Output layer for a single value
        model_aftershock.compile(optimizer='adam', loss='mean_squared_error')
        print("Training model for 'y_aftershock'...")
        model_aftershock.fit(X_train, y_train_aftershock, epochs=50, batch_size=32, verbose=1)
        
        return model_hours, model_aftershock

        #######################################################

    
    # Evaluate the model based on training
    def evaluate_trained_model(self,model, X_test, y_test, is_nn=False):
        if is_nn:
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")

        return y_pred
    
    def predict_triggered_due_to_aftershock(self,nn_model_hours,nn_model_aftershock,predicted_aftershocks):
        for distance, critical_threshold in zip(self.distances, eq_prone_crits):
            for mag in predicted_aftershocks:
                # Calculate static stress change using the distance and magnitude at that distance
                delta_sigma = self.static_stress_change(mag, distance)
                # Evaluate if the static stress change exceeds the critical threshold
                print(f'delta sigma is{delta_sigma}')
                if np.all(delta_sigma >= critical_threshold):
                    pressure_diff = delta_sigma-critical_threshold
                    self.difference_in_pressure.append(pressure_diff)

        outputs = []
        #print(self.difference_in_pressure)
        for distance ,pressure_diff in zip(self.distances,self.difference_in_pressure):
            for aftershock in predicted_aftershocks:
                inputs = [distance,self.mag,pressure_diff,aftershock]
                # If the input array is 1D and you want it to work with 2D input:
                
                print(f'The input before converting is::{inputs}')
                
                # Ensure all elements in `inputs` are lists
                inputs_fixed = []
                for element in inputs:
                    if not isinstance(element, (list, np.ndarray)):
                        inputs_fixed.append([element])  # Wrap scalar in a list
                    else:
                        inputs_fixed.append(element)  # Already a list/array
                #Pad sequences if lengths are inconsistent
                inputs_fixed = pad_sequences(inputs_fixed, padding='post', dtype='float32')
                # Debug: Print inputs after padding
                print("Inputs after padding:", inputs_fixed)


                # outputs.append(model.predict(inputs)) 
                with open(f'predicted_Aftershocks_wangjing.txt', 'w') as file:
                    # Predict and write to the file
                    prediction = nn_model_hours.predict(inputs_fixed)  #predict(inputs_reshaped)
                    file.write(f"{prediction} hours")
                    file.write(" \n ")
                    prediction = nn_model_aftershock.predict(inputs_fixed)  #predict(inputs_reshaped)
                    file.write(f"{prediction} ")
        
    
        

############################################################################################################################################

            #########################               MAIN FUNCTION                     #########################


# Initialize lists to store latitude, longitude, and critical stress thresholds
eq_prone_lats = []
eq_prone_longs = []
eq_prone_crits = []

# Open and read the EQ_prone_regions.txt file
with open('inputsForEq_predictions/EQ_prone_regions.txt', 'r') as file:
    for line in file:
        # Split each line by tab or spaces to extract values
        values = line.strip().split()
        
        # Store latitude, longitude, and critical stress threshold in respective lists
        eq_prone_lats.append(float(values[0]))
        eq_prone_longs.append(float(values[1]))
        eq_prone_crits.append(float(values[2]))

eq_prediction = EQprediction(source_magnitude,source_latitude, source_longitude,eq_prone_lats, eq_prone_longs,api_key)
eq_prediction.calculate_distances()
eq_prediction.calc_magnitude_at_distance()
eq_prediction.evaluate_Mainshock_triggering(eq_prone_crits)
Mainshock_model =  eq_prediction.train_model()

eq_prediction.predictMainshock_earthquake(Mainshock_model)

###############################################################################################################################################################################
####################################################################################################################################################################################
###################################################################################################################################################################################


predicted_aftershocks = eq_prediction.aftershock_magnitudes()
##############################################################################################
predicted_aftershocks = np.array(predicted_aftershocks)
if len(predicted_aftershocks.shape) == 1:  # If 1D, reshape to (1, -1)
    predicted_aftershocks = predicted_aftershocks.reshape(1, -1)
else:
    predicted_aftershocks = predicted_aftershocks  # Already in the correct shape
##############################################################################################


# x, y_hours,y_aftershock = eq_prediction.load_data()
# x_train, x_test, y_train_hours, y_test_hours = eq_prediction.split_data(x, y_hours)
# x_train, x_test, y_train_aftershock, y_test_aftershock = eq_prediction.split_data(x, y_aftershock)

# # Train and evaluate Random Forest
# print("Random Forest Regressor:")
# rf_model = eq_prediction.train_random_forest(x_train, y_train)
# eq_prediction.evaluate_trained_model(rf_model, x_test, y_test)

# Train and evaluate Gradient Boosting Regressor
# print("\nGradient Boosting Regressor:")
# gb_model = eq_prediction.train_gradient_boosting(x_train, y_train)
# eq_prediction.evaluate_trained_model(gb_model, x_test, y_test)

# # Train and evaluate Neural Network
# print("\nNeural Network:")
# nn_model_hours, nn_model_aftershock = eq_prediction.train_neural_networks(x_train, y_train_hours , y_train_aftershock)
# eq_prediction.evaluate_trained_model(nn_model_hours, x_test, y_test_hours, is_nn=True)
# eq_prediction.evaluate_trained_model(nn_model_aftershock, x_test, y_test_aftershock, is_nn=True)


# # aftershock_forecast = eq_prediction.predict_triggered_due_to_aftershock(rf_model,predicted_aftershocks)
# # print(aftershock_forecast)
# # aftershock_forecast = eq_prediction.predict_triggered_due_to_aftershock(gb_model,predicted_aftershocks)
# # print(aftershock_forecast)
# aftershock_forecast = eq_prediction.predict_triggered_due_to_aftershock(nn_model_hours,nn_model_aftershock,predicted_aftershocks)
# print(aftershock_forecast)
