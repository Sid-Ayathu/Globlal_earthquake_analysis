'''

**This script requires that you have downloaded all data chunks from the STanford EArthquake Dataset (STEAD). The data can be downloaded here: https://drive.google.com/file/d/1oiuS7ByCyE2-7rARs6jXWN34Amf-Vrbg/view.

This script reads in the metadata csv files for each data chunk, and allows you to create images from selected waveform signals by pulling the signal data from the hdf5 files. Running the make_images function creates two images:
        1. Waveform plot of signal
        2. Spectrogram plot of signal
        
This script runs in parallel using joblib. Set n_jobs to choose number of cores (-1 will use all cores, -2 will use all but one core, etc.)

'''

import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
import timeit
from joblib import Parallel,delayed

############################# USER INPUT #############################
#please enter the path to the stead dataset in your local system
data_dir = 'path_to_dataset'
#############################################################################

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

############################# USER INPUT #############################
chunk_name = earthquakes_5 # select chunk of data from which you want to create the images
data_start = 0 # select start of data rows you want to pull from that chunk
data_end = 199000 # select end of data rows you want to pull from that chunk
data_interval = 1000 # select interval you'd like to pull (smaller interval with more loops may run faster)
eqpath = eq5_sig_path # select path to data chunk
no_of_imgs = 12000 #no.of images you want to create from that chunk
#############################################################################

## Make images
eqlist = chunk_name['trace_name'].to_list()
eqlist = np.random.choice(eqlist,no_of_imgs,replace=False) # turn on to get random sample of signals

starts = list(np.linspace(data_start,data_end-data_interval,int((data_end-data_start)/data_interval)))
ends = list(np.linspace(data_interval,data_end,int((data_end-data_start)/data_interval)))
set = str(chunk_name)

count = 0
for n in range(0,len(starts)):
    traces = eqlist[int(starts[n]):int(ends[n])]
    path = eqpath
    count += 1
    
    def make_images(i):
        # retrieving selected waveforms from the hdf5 file:
        try:
            dtfl = h5py.File(path, 'r')
            dataset = dtfl.get('data/'+str(traces[i]))
            # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
            data = np.array(dataset)
            print('working on ' + set + ' waveform ' +str(traces[i]) + ' chunk '+str(count) + ' number ' +str(i))
            
            fig, ax = plt.subplots(figsize=(3,2))
            ax.plot(np.linspace(0,60,6000),data[:,2],color='k',linewidth=1)
            ax.set_xlim([0,60])
            ax.axis('off')
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.savefig(traces[i]+'.png',bbox_inches='tight',dpi=50)
            plt.close()
            
            fig, ax = plt.subplots(figsize=(3,2))
            ax.specgram(data[:,2],Fs=100,NFFT=256,cmap='gray',vmin=-10,vmax=25)
            ax.set_xlim([0,60])
            ax.axis('off')
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.savefig(traces[i]+'.png',bbox_inches='tight',transparent = True,pad_inches=0,dpi=50)
            plt.close()
            
        except:
            print('String index out of range')


    # create images for selected data (runs in parallel using joblib)
    start = time.time()
    print(start)
    Parallel(n_jobs=-2)(delayed(make_images)(i) for i in range(0,len(traces))) # run make_images loop in parallel on all but 2 cores for each value of i
    end = time.time()
    print(f'Took {end-start} s')
