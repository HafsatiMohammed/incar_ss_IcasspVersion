# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 13:10:40 2022

@author: hafsati
"""


import numpy as np 
import os 
from tqdm import tqdm 
import scipy.io
import scipy
import pickle
import matplotlib.pyplot as plt
import random
import librosa



from itertools import zip_longest

import os
import random

# Set the path to your folder containing the 40 subfolders
folder_path = r'./LibriSpeech/test-clean'

# Initialize an empty list to store the selected file paths
selected_files = []

# Iterate through the subfolders
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Check if the file has a .flac extension
        if file.endswith('.flac'):
            # Add the full file path to the selected_files list
            file_path = os.path.join(root, file)
            selected_files.append(file_path)




def read_rir(path):
     with open(path, "rb") as f:
         len = np.fromfile(f,dtype= int,count= 1)
         h = np.fromfile(f, '<f4')
     return len,h



def Load_SRIR():  
    SRIR_Dir = 'C:/Users/hafsa/Desktop/SoundSceneWithMcroomsim/Generation_Scenes_Sonores/Simulated_Car_ImpulseResponses/' 
    SRIR_Dir_save = 'C:/Users/hafsa/Desktop/SoundSceneWithMcroomsim/Generation_Scenes_Sonores/Simulated_Car_ImpulseResponses_Real_PklFiles/' 
    os.makedirs(SRIR_Dir_save)
    #SRIR = np.zeros((629,4,4,32682)) # Receiver, Source, impulse
    #print(os.listdir(SRIR_Dir))
    cpt=0
    for eachfile in tqdm(os.listdir(SRIR_Dir)):   
        Name = eachfile.split('_')
        file = os.path.join(SRIR_Dir, eachfile)
        matlabfile = scipy.io.loadmat(file)
        ImpResp_Rev = matlabfile['ImpResp_Rev'];
        SRIR = np.zeros((4,8,32682))
        for i in range(ImpResp_Rev.shape[0]):
            for j in range(ImpResp_Rev.shape[1]):
                ImpResp_Rev_ = ImpResp_Rev[i,j]
                secs = ImpResp_Rev[0,0].shape[0]/48000
                samps = secs*16000
                SRIR[i,j,:] =  np.squeeze(ImpResp_Rev_[:32682,:],1) #np.squeeze(scipy.signal.resample(ImpResp_Rev_,int(samps))[:], 1)
              
                #plt.plot(np.squeeze(scipy.signal.resample(ImpResp_Rev_,int(samps))[:], 1))
                #plt.show()
        #os.mkdir(SRIR_Dir)
        with open(SRIR_Dir_save+str(cpt)+'.pkl','wb') as f: 
            pickle.dump([SRIR],f)
            cpt=cpt+1
    return


def olafilt(b, x, zi=None):
    L_I = b.shape[0]
    # Find power of 2 larger that 2*L_I (from abarnert on Stackoverflow)
    L_F = 2<<(L_I-1).bit_length()
    L_S = L_F - L_I + 1
    L_sig = x.shape[0]
    offsets = range(0, L_sig, L_S)

    # handle complex or real input
    if np.iscomplexobj(b) or np.iscomplexobj(x):
        fft_func = np.fft.fft
        ifft_func = np.fft.ifft
        res = np.zeros(L_sig+L_F, dtype=np.complex128)
    else:
        fft_func = np.fft.rfft
        ifft_func = np.fft.irfft
        res = np.zeros(L_sig+L_F)

    FDir = fft_func(b, n=L_F)

    # overlap and add
    for n in offsets:
        res[n:n+L_F] += ifft_func(fft_func(x[n:n+L_S], n=L_F)*FDir)

    if zi is not None:
        res[:zi.shape[0]] = res[:zi.shape[0]] + zi
        return res[:L_sig], res[L_sig:]
    else:
        return res[:L_sig]
    


def Add_noise(s,n,SNR):
    #SNR = 10**(SNR/20)
    Es = np.sqrt(np.sum(s[:]**2))
    En = np.sqrt(np.sum(n[:]**2))
    iSNR = 10*np.log10(Es**2/(En**2+1e-6)) 
    alpha = 10**((iSNR-SNR)/20)
    
    #alpha = Es/(SNR*(En+1e-8))
    #•Mix = s+alpha*n[0:160000]
    
    return  alpha

### to load SRIR

def load_pickle(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)








def DataGenration_filePerfile(SRIR,Lsp_SRIR ,Speakers_Dir, Music_Dir, NumberOfSpeakers, Mic_conf, SNR):
    ## Get the speakers 
        Source_path = []
        len_mix = 1
        lenghthOfMixture = 10 #up to 10 seconds
        sampling_rate = 16000
        #print(Speakers_Dir)
        List_Speech = os.listdir(Speakers_Dir)
        List_Music  = os.listdir(Music_Dir)
        Chosen_Speech = random.choices(List_Speech,k=NumberOfSpeakers)
        #print(len(Chosen_Speech))
        Chosen_Music = random.choices(List_Music,k=1)
        Speech_before = []
        #SRIR = load_pickle(SRIR_file)[0] # load the car impulse response
        #SRIR_Dir_Noise_Impulse =r'C:/Users/hafsa/Desktop/SoundSceneWithMcroomsim/Generation_Scenes_Sonores/Simulated_DiffuseNoise_ImpulseResponses_PklFiles/'  #'./Simulated_DiffuseNoise_ImpulseResponses_PklFiles/' 
        spk_number = ['12','13','14','15']
        if Mic_conf == 'set0':
            mic_number = ['02','03','04','05']
        elif Mic_conf == 'set1':
            mic_number = ['01','06','07','12']
        elif Mic_conf == 'set2':
            mic_number = ['03','04','09','10']
            
        mic_number = ['02','03','04','05']
        lsp_number = ['18','19','20','21']



        
        random_selection = random.sample(selected_files, 4)
    

     
                  



        # Handle the sources 
        for il in range(len(random_selection)):
             y,fs = librosa.load(random_selection[il],sr=sampling_rate)
             y =  y/(np.max(np.abs(y))+1e-6)
             if len(y)>len_mix:
                len_mix = len(y)
                lenghthOfMixture = len_mix 


             Source_path.append(random_selection[il])
             if len(y)<lenghthOfMixture:
                 y = np.append(np.zeros(lenghthOfMixture-len(y)), y)
             else:
                 y = y[:lenghthOfMixture] 
            
            
             Speech_before.append(y)
             max_length = max(len(signal) for signal in Speech_before)
             Speech = [np.array(list(signal) + [0] * (max_length - len(signal))) for signal in Speech_before]


        
                 
        Contrib_Rev = np.zeros((4,11,lenghthOfMixture)).astype(np.float32) # mic, source, lenght of mixture
        Contrib_Anec = np.zeros((4,5,lenghthOfMixture)) 
        Y = np.zeros((4,5,lenghthOfMixture))                  
        for il in range(len(random_selection)): 
             y = Speech[il] 
             #print(len(y))
             for mic in range(4):
               SRIR_Anec= np.zeros((1919))
               I = np.argmax(SRIR[mic,il,:])
               SRIR_Anec[I:I+8] = SRIR[mic,il,I:I+8]
               #plt.plot(SRIR[mic,il,:])
               Contrib_Rev[mic,il,:] = olafilt(SRIR[mic,il,:],y)[:lenghthOfMixture]
               #plt.plot(Contrib_Rev[mic,il,:])
               #plt.show()
               Contrib_Anec[mic,il,:] = olafilt(SRIR_Anec,y)[:lenghthOfMixture]
               Y[mic,il,:] = y
               #sd.play(SRIR[mic,il,:], 16000)
               #sd.wait()

        #Handle the noise in a diffuse manner 
        Music = np.zeros(lenghthOfMixture)
        
        for il in range(len(Chosen_Music)):
             y,fs = librosa.load(os.path.join(Music_Dir, Chosen_Music[il]),sr=sampling_rate)
             y = y/(np.max(np.abs(y))+1e-6)
             
             if len(y)<lenghthOfMixture:
                y = np.append(np.zeros(lenghthOfMixture-len(y)), y)
             else:
                y = y[:lenghthOfMixture] 
             Music = y + Music
             
             
             
        Music = Music/(np.max(np.abs(Music))+1e-6)

        #Noise = Noise/(np.max(np.abs(Noise))+1e-6)
        for il in range(4):                 
            for mic in range(SRIR.shape[0]):
                #SRIR_Anec = np.zeros()
                Contrib_Rev[mic,il+4+1,:] = olafilt(Lsp_SRIR[mic,il,:],Music)[:lenghthOfMixture]
                Contrib_Anec[mic,4,:] =  Music
        DesiredSNR = np.random.uniform(low=SNR, high=SNR, size=(1,))
        
        alpha_Music = Add_noise(Contrib_Rev[0,0,:] , Contrib_Rev[0,5,:],DesiredSNR) # sum(Contrib_Rev[0,5:,:],0)
        alpha_Music = alpha_Music.astype(np.float32)
        Contrib_Rev[:,5:5+4,:] = alpha_Music*Contrib_Rev[:,5:5+4,:]        
        #plt.plot(Contrib_Rev[0,5:5+4,:])
        #plt.show()




        folder = './v_class/background_noise'
        folders_noise = os.listdir(folder)
        random_folder = random.choice(folders_noise)
        chosenPathFolder = os.path.join(folder,random_folder)
        name_noise = chosenPathFolder.split('/')[-1].split('\\')[-1] 
        #▲print(name_noise )
        Noise = np.zeros(lenghthOfMixture) 
        
        """
        for il in range(len(Chosen_Noise)):
             y,fs = librosa.load(os.path.join(Noise_Dir, Chosen_Noise[il]),sr=sampling_rate)

             if len(y)<sampling_rate*lenghthOfMixture:
                y = np.append(np.zeros(sampling_rate*lenghthOfMixture-len(y)), y)
             else:
                y = y[:sampling_rate*lenghthOfMixture] 
             y = y/(np.max(np.abs(y))+1e-6)             
             Noise = y + Noise
        #Noise = Noise/np.max(np.abs(Noise)+1e-6)     
        #Noise = np.reshape(Noise, (5,160000))
        """
        start_index = random.randint(0, 30*16000 - lenghthOfMixture)
        
        # Calculate the end index based on the start index and the number of samples (160,000)
        end_index = start_index + lenghthOfMixture        

            
        for mic in range(SRIR.shape[0]):
                Noise,_ = librosa.load(os.path.join( chosenPathFolder ,'con_' + name_noise+'_mic_'+mic_number[mic]+'.wav'), sr=16000)   
                Contrib_Rev[mic,9,:] = Noise[start_index:end_index]   
        del Noise
                 
                    

        DesiredSNR = np.random.uniform(low=5, high=15, size=(1,))
        alpha = Add_noise(Contrib_Rev[0,0,:], Contrib_Rev[0,9,:] ,DesiredSNR)
        #print(f'herrre     {alpha}')    
        Contrib_Rev[:,9,:] = alpha*Contrib_Rev[:,9,:]        
        #plt.plot(Contrib_Rev[0,9,:])        
        
        #print(alpha_Music)
        Mix = np.sum(Contrib_Rev[:,:9,:],1)  +  np.sum(Contrib_Rev[:,9:,:],1) 
        
        Mix = Mix/np.max(np.abs(Mix)+1e-6)
        
        return Mix, Contrib_Anec, Source_path, lenghthOfMixture, Y # Mixture (channels, samples) , Conrib_Rev (Receiver, source, sample)  

    
        
    
        
        
        
        

   
if __name__ == '__main__':
    target_sampling_rate = 16000
    
    # Calculate the ratio of original to target sampling rate
    resampling_ratio = target_sampling_rate / 48000
    
    
    SRIR = np.zeros((4, 4, 1919))
    spk_number = ['12','13','14','15']
    mic_number = ['02','03','04','05']
    lsp_number = ['18','19','20','21']
        
    from scipy.signal import resample
    
    for mic in range(4):
        for speaker in range(4):
            file = './v_class/impulse_responses/lsp_'+ spk_number[speaker]+ '_mic_' + mic_number[mic]+ '_enh.fir'

            _ , original_signal =  read_rir(file)  
            new_length = int(len(original_signal) * resampling_ratio)
            SRIR[mic,speaker,:]  = resample(original_signal, new_length)   
            
            
            #plt.plot( SRIR[mic,speaker,:] )
            
    Lsp_SRIR = np.zeros((4, 4, 1919))
    
    for mic in range(4):
        for speaker in range(4):
            file = './v_class/impulse_responses/lsp_'+ lsp_number[speaker]+ '_mic_' + mic_number[mic]+ '_enh.fir'
            _ , original_signal =  read_rir(file)  
            new_length = int(len(original_signal) * resampling_ratio)
            Lsp_SRIR[mic,speaker,:] =  resample(original_signal, new_length)                 
   
    #Speakers_Dir =  'C:/Users/hafsa/Documents/InteractiveVoiceCommand/Src_code/Speech_Separation_Enhancement/BMW/LibriSpeech/train-clean-360/clean_speech' 
    
    Maximum = np.max( [np.max(np.abs(Lsp_SRIR)), np.max(np.abs(SRIR))] )
    
    SRIR = SRIR/Maximum
    Lsp_SRIR = Lsp_SRIR/Maximum 
    
 
    Speakers_Dir =  './LibriSpeech/train-clean-360/clean_speech' 

    
    #SRIR_file = os.path.join(SRIR_Dir_Car_Impulse,ChosenSRIR[0])
    Music_Dir =  './MusicFiles' 
    Mix, Contrib_Anec, Source_path, lenghthOfMixture = DataGenration_filePerfile(SRIR,Lsp_SRIR ,Speakers_Dir, Music_Dir, 4) #DataGenration(SRIR_file ,Speakers_Dir,CarEngineAndUrbanNoise_Dir,Noise_Dir) 
   
    
