import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib



#####################################################################################
# test_trace_names = ['PAH.NN_20151216135336_EV','PAH.NN_20151216142205_EV']
#####################################################################################

#======================================================================================================

# 1. Load the training dataset
data = {  # first 11 entries are from california,next 16 are from portland ,remaining from nevada
    'Magnitude': [3.7, 3.5 ,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.2,3.2,3.2,3.2,3.2,4,4.9,5.7,3,3,3,3,3,4.5,4.5, 3.1,3.1 ,3.1,3.1,3.1,3.1,3.1,3.1,3.1,3.8,3.8,3.6,3.6,3.7],
    'HypocentralDistance': [82.92,142.21,92.84, 111.81,105.64,285.28,277.44,630.07,599.56,47.33,30.01,58.19,67.76,56.27,61.3,59.94,95.61,129.71,138.62,47.24,43.63,39.04,48.53,41.02,103.57,122.3,
                 47.86,35.79,50.14,43.33,45.36,43.37,54.73,47.13,120.65,30.21,38.06,99.12,111.16,88.09 ],  # Hypocentral Distance = sqrt[(epicentralDistance)^2 + (depth)^2]
    'SiteFactor': [1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.3,1.3,1.3,1.3,1.3,1.3,1.3,1.3,1.3,1.3,1.3,1.3,1.3,1.3,1.3,1.3,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,],  # Site amplification factor
    'AttenuationCoeff': [0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.004,
                         0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.004],  #if causing problem just remove it.
    'PGA': [0.01492229,0.0089248434,0.038614187,0.0014429359,0.0027908882,0.0000023504294,0.00039158582,0.000067854289,0.000071975781,0.022247764,0.030828327,0.0046098165,0.012103008,0.017184066,
                     0.0048861366,0.011030815,0.0084368595,0.0664393,0.14820794,0.047813394,0.0030063416,0.01195665,0.011946911,0.022751248,0.038323078,0.0071293817,0.0406,0.0242,0.0114,0.00924,0.0263,
                      0.0197,0.0156,0.0226,0.00076 ,1.02,0.154,0.00226,0.00242,0.0135]  
}

df = pd.DataFrame(data)

#======================================================================================================
  
# 3. Prepare the feature matrix (X) and target vector (y)
features = ['Magnitude', 'HypocentralDistance','SiteFactor' ,'AttenuationCoeff']
X = df[features]
y = df['PGA']

#======================================================================================================

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#======================================================================================================

# 5. Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#======================================================================================================

# 6. Model Selection
# Using RandomForestRegressor for better handling of non-linearities
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
# Save the trained model to a file
joblib.dump(model, "random_forest_model.pkl")
#======================================================================================================

# 7. Model Evaluation
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

#======================================================================================================
SiteFactors = {
    'California': 1.2,
    'Nevada': 1.1,
    'Oregon': 1.3,
    'Washington': 1.4
}

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
###########################################################################################

def classify_earthquake_damage(pga):
    """
    Classifies earthquake damage based on Peak Ground Acceleration (PGA) in m/s².
    :param pga: float, PGA value in m/s²
    :return: str, damage category
    """
    if pga < 0.2:
        return "No Damage"
    elif 0.2 <= pga < 0.8:
        return "Minor"
    elif 0.8 <= pga <= 2.0:
        return "Moderate "
    elif 2.0 < pga < 5.0:
        return "Severe "
    elif 5.0 <= pga < 10.0:
        return "Heavy "
    else:
        return "Catastrophic"

###########################################################################################
# 8. Predict PGA for a new earthquake event
    # Combining the values into a single array
mag = 5.477439
dist = 6.390942892318233
sf = 2.4
att = attenuation_coefficients['California']
new_event = np.array([[mag, dist, sf, att ]])  #mag,distance, site factor ,att coeff.

# model = joblib.load("random_forest_model.pkl")

# scaler = StandardScaler()
# scaler.fit_transform(new_event)
# new_event_scaled = scaler.transform(new_event)
predicted_pga = model.predict(new_event)
# print(f"Predicted PGA value: {predicted_pga[0]}")
# print(f' Magnitude of the triggered earthquake is {mag} \n')  
# print(f'with distance {dist} km  from the source earthquake \n')  
# print(f'the location has a site factor of approxiamtely {sf} \n')
# print(f'The attenuation coefficient of the region is {att}')       
print(f"Predicted PGA in m/s²: {predicted_pga[0]*9.8} m/s² -> {classify_earthquake_damage(predicted_pga[0]*9.8)}")

