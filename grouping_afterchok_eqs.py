import pandas as pd
import numpy as np
import math
from datetime import timedelta,datetime

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('STEAD csv/chunk6.csv',low_memory=False)

# Step 2: Convert 'trace_start_time' to datetime
#df['trace_start_time'] = pd.to_datetime(df['trace_start_time'], format='%Y-%m-%d %H:%M:%S')

def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Radius of Earth in kilometers
    R = 6371.0

    # Difference in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c
    return distance


# Function to handle parsing and rounding
def parse_and_round_datetime(datetime_str):
    # Check if the string contains fractional seconds (".%f")
    if '.' in datetime_str:
        try:
            # Attempt to parse with fractional seconds
            dt = pd.to_datetime(datetime_str, format='%Y-%m-%d %H:%M:%S.%f')
            # Round to nearest second
            dt = dt.round('S')
        except ValueError:
            # If it fails to parse (e.g., format mismatch), return NaT
            return pd.NaT
    else:
        try:
            # Parse without fractional seconds
            dt = pd.to_datetime(datetime_str, format='%Y-%m-%d %H:%M:%S')
        except ValueError:
            # If parsing fails, return NaT
            return pd.NaT
    return dt

# Apply the function to the 'trace_start_time' column in the DataFrame
df['trace_start_time'] = df['trace_start_time'].apply(parse_and_round_datetime)
df['trace_start_time'] = pd.to_datetime(df['trace_start_time'])

# Check the results
print(df['trace_start_time'].head())

# Step 3: Sort the DataFrame by 'trace_start_time'
df = df.sort_values(by='trace_start_time')


# Pre-extract columns for better performance
latitudes = df['source_latitude'].values
longitudes = df['source_longitude'].values
trace_names = df['trace_name'].values
magnitudes = df['source_magnitude'].values
times = df['trace_start_time'].values


# Step 4: Initialize an empty list for groups
groups = []
current_group = [df.iloc[0]]

# Step 5: Loop through the DataFrame and group based on time difference
for i in range(1, len(df)):
    current_time, previous_time = times[i], times[i-1]
    current_eq, previous_eq = trace_names[i], trace_names[i-1]
    current_magnitude, previous_magnitude = magnitudes[i], magnitudes[i-1]

    
    # Calculate time difference in minutes and hours and seconds
    time_diff_in_seconds = (current_time - previous_time)/np.timedelta64(1, 's')
    # time_diff_minutes = (current_time - previous_time)/ np.timedelta64(1, 'm')
    # time_diff_hrs = (current_time - previous_time)/ np.timedelta64(1, 'h')
    mag_diff = current_magnitude - previous_magnitude

    distance_km = haversine_distance(latitudes[i-1], longitudes[i-1], latitudes[i], longitudes[i])

    # Convert numpy.datetime64 to Python datetime for date comparison
    current_date = pd.Timestamp(current_time).date()
    previous_date = pd.Timestamp(previous_time).date()
     # Apply the grouping rules based on magnitude and time difference
    if current_date == previous_date:
        higher_magnitude = max(current_magnitude, previous_magnitude)


        ##########################  change values of time_diff and add constraint of mag_diff(in extraction) ##########################

        if 0.5<current_magnitude <= 2.5 and 0.5<previous_magnitude <= 2.5 and time_diff_in_seconds > 300  and distance_km >= 1 :
            current_group.append(df.iloc[i])
        elif 2.5 < higher_magnitude <= 4.0 and time_diff_in_seconds >600 and distance_km > 5 :
            # Group if the higher magnitude is between 2.5 and 4.0, and time difference is between 2 and 20 minutes
            current_group.append(df.iloc[i])
        elif 4.0<higher_magnitude <= 6.0 and  time_diff_in_seconds >1200 and distance_km > 10 :
            # If both magnitudes are > 4.0 and time difference is between 2 and 30 minutes
            current_group.append(df.iloc[i])
    else:
        # Only save the group if it contains more than one earthquake
        if len(current_group) > 1:
            groups.append(pd.DataFrame(current_group))
        current_group = [df.iloc[i]]

# Add the last group if it contains more than one earthquake
if len(current_group) > 1:
    groups.append(pd.DataFrame(current_group))

# Step 6: Concatenate all grouped DataFrames into a single DataFrame
if groups:
    grouped_df = pd.concat(groups)
    grouped_df.to_csv(f'aftershock_grouped_chunks/aftershock_grouped_chunk6.csv', index=False)
    print('Grouped earthquakes have been saved to "aftershock_grouped_chunk6.csv".')
else:
    print('No groups found with earthquakes less than 30 minutes apart.')
