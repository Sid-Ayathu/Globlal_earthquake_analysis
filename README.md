# Global_earthquake_analysis
This project helps us to find correlations between quakes &amp; tremors in one part of the globe to another, and between events of different magnitudes, and in some cases - correlations between events occurring over time.

TO RUN THIS PROJECT , PLEASE FOLLOW THE FOLLOWING STEPS:
Step1: install the dataset which has been used to train the CNN model from here - https://drive.google.com/file/d/1oiuS7ByCyE2-7rARs6jXWN34Amf-Vrbg/view .
Step2: run create_images.py and create 75,000 images of earthquake signals.
Step3: run seismic_cnn.py to train the classification_cnn model for predicting trace_category and regression_snn for predicting source_magnitude,p_sample_arrival and s_sample_arrival.
Step4: Put the trace_name of the images of 2 earthquakes of which you want to find their correlations in testing.py and execute it.
Step5: Put the trace_name of the images in correlations.py and run it.
