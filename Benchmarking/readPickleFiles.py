# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 17:57:45 2023

@author: hafsa
"""


import pickle
import numpy as np



file = 'set1_Music_results.pickle'
# Open the pickle file for reading in binary mode
with open(file, 'rb') as file:
    # Load the data from the pickle file
    result = pickle.load(file)

# Now 'data' contains the contents of the pickle file



Model = ['JAECBF', 'WaveUnet', 'IC-ConvTas', 'Mix', 'Truth']
Seat =  ['Seat 0', 'Seat 1', 'Seat 2', 'Seat 3'  ]
Metric = ['sdr', 'sir', 'sar', 'pesq_nb', 'wer', 'si-snr']



for model in Model:
    print(f'  *********************  {model} ***********************')
    for metric in Metric:
        print(f'  **************** {metric} *************')
        for seat in Seat: 
            #print(seat)
            print(np.median(result[metric][model][seat]))
