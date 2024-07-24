import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import time
from joblib import Parallel, delayed

##############################################################################################################################################
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

eqpath = eq5_sig_path # select path to data chunk

img_save_path = '/Users/Siddharth/Desktop/IIITB/bashok_srip/STEAD_dataset/images/testing_spectrograms'

##############################################################################################################################################
# Function to create images for a specific trace
def make_image_for_trace(trace_name, eqpath):
    try:
        dtfl = h5py.File(eqpath, 'r')
        dataset = dtfl.get('data/' + str(trace_name))
        data = np.array(dataset)
        print('working on waveform ' + str(trace_name))
        
        # Plot waveform
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot(np.linspace(0, 60, 6000), data[:, 2], color='k', linewidth=1)
        ax.set_xlim([0, 60])
        ax.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(trace_name + '_waveform.png', bbox_inches='tight', dpi=50)
        plt.close()
        
        # Plot spectrogram
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.specgram(data[:, 2], Fs=100, NFFT=256, cmap='gray', vmin=-10, vmax=25)
        ax.set_xlim([0, 60])
        ax.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(trace_name + '.png', bbox_inches='tight', transparent=True, pad_inches=0, dpi=50)
        plt.close()
        
    except Exception as e:
        print(f'Error processing {trace_name}: {e}')
##############################################################################################################################################

# ENTER the trace_name of the earthquake whose seismogram you want to create
trace_name_to_process = 'PWL.AK_20170225113534_EV'

# Check if the trace name exists in the CSV
if trace_name_to_process in full_csv['trace_name'].values:
    eqpath = eq5_sig_path
    start = time.time()
    make_image_for_trace(trace_name_to_process, eqpath)
    end = time.time()
    print(f'Took {end - start} s')
else:
    print(f'Trace name {trace_name_to_process} not found in the CSV file.')
