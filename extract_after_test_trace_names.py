import pandas as pd
import math
from datetime import timedelta


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


# Step 1: Read the grouped CSV file into a DataFrame
grouped_df = pd.read_csv('aftershock_grouped_chunks/aftershock_grouped_chunk6.csv')

# Step 2: Convert 'trace_start_time' to datetime
grouped_df['trace_start_time'] = pd.to_datetime(grouped_df['trace_start_time'])

# Step 3: Initialize an empty list to store trace names
test_trace_names = []

# Step 4: Sort the DataFrame by 'trace_start_time'
grouped_df = grouped_df.sort_values(by='trace_start_time')

latitudes =grouped_df['source_latitude'].values
longitudes = grouped_df['source_longitude'].values

# Step 5: Loop through the DataFrame and form pairs based on your conditions
for i in range(1, len(grouped_df)):
    current_row = grouped_df.iloc[i]
    previous_row = grouped_df.iloc[i - 1]
    
    current_time = current_row['trace_start_time']
    previous_time = previous_row['trace_start_time']
    
    current_magnitude = current_row['source_magnitude']
    previous_magnitude = previous_row['source_magnitude']
    
    distance_km = haversine_distance(latitudes[i-1], longitudes[i-1], latitudes[i], longitudes[i])

    # Check if the earthquakes occurred on the same date
    same_date = current_time.date() == previous_time.date()
    
    if same_date:
        # Calculate time difference in minutes
        time_diff = (current_time - previous_time).total_seconds() / 60.0
        higher_magnitude = max(current_magnitude, previous_magnitude)

        # Apply your conditions
        if 1.5<current_magnitude <= 2.5 and 1.5< previous_magnitude <= 2.5 and time_diff>10 and time_diff < 30 and distance_km > 5 and distance_km < 15:
            print (f'distance1 is: {distance_km}' )
            # Both magnitudes <= 2.5 and time_diff < 10 minutes
            test_trace_names.append((previous_row['trace_name'], current_row['trace_name']))
        elif 2.5<higher_magnitude <= 3.2 and time_diff > 15 and distance_km > 10 and distance_km < 25:
            print (f'distance2 is: {distance_km}' )
            test_trace_names.append((previous_row['trace_name'], current_row['trace_name']))
        elif 3.2<higher_magnitude <= 4.0  and time_diff > 20 and distance_km > 15 and distance_km < 50:
            print (f'distance3 is: {distance_km}' )
            test_trace_names.append((previous_row['trace_name'], current_row['trace_name']))
        elif 4.0<higher_magnitude <= 5.0 and time_diff > 30 and distance_km > 20 and distance_km < 75:
            print (f'distance4 is: {distance_km}' )
            test_trace_names.append((previous_row['trace_name'], current_row['trace_name']))
        elif 5.0<higher_magnitude <= 6.0 and time_diff>40  and distance_km > 25 and distance_km < 100:
            print (f'distance5 is: {distance_km}' )
            test_trace_names.append((previous_row['trace_name'], current_row['trace_name']))
          
print(f"Collected {len(test_trace_names)} pairs of trace names.")

# Step 7: Write the collected pairs to a text file
with open('after_test_pairs.txt', 'a') as f:
    for pair in test_trace_names:
        f.write(f"{pair[0]} {pair[1]}\n")

print("Test pairs have been saved to 'after_test_pairs.txt'.")

# Step 8: Print the first few pairs to verify
print(test_trace_names[:5])
