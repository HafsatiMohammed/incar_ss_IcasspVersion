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
    #print(np.min(n), np.max(n))
    #n = n.astype(np.float64)
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




def load_all_dataInMemory(Dir_SRIR_file_car, Dir_SRIR_file_noise, Speakers_Dir, CarEngineAndUrbanNoise_Dir, Noise_Dir, Music_Dir, Wind_Dir):
### Dir_SRIR_file_car
    ListofSRIR = os.listdir(Dir_SRIR_file_car)
    SRIR_car = np.zeros((len(ListofSRIR),4,8,32682))
    for cpt, each_file in tqdm(enumerate(ListofSRIR)):
        SRIR_file = os.path.join(Dir_SRIR_file_car,each_file)
        SRIR_car[cpt,:,:,:] = load_pickle(SRIR_file)[0] 
### Dir_SRIR_file_noise
    ListofSRIR = os.listdir(Dir_SRIR_file_noise)
    SRIR_noise = np.zeros((len(ListofSRIR),2500))
    for cpt, each_file in tqdm(enumerate(ListofSRIR)):
        SRIR_file = os.path.join(Dir_SRIR_file_noise, each_file)
        SRIR_noise[cpt,:] = load_pickle(SRIR_file)[0] 
### Speakers_Dir
    ListofSpeakers = os.listdir(Speakers_Dir)
    Speakers = np.zeros((len(ListofSpeakers),160000))
    for cpt, each_file in tqdm(enumerate(ListofSpeakers)):
        file = os.path.join(Speakers_Dir, each_file)
        y, _ = librosa.load(file, sr=16000)
        if len(y)<160000:
             y = np.append(np.zeros(160000-len(y)), y)
        else:
             y = y[:160000]   

        Speakers[cpt,:] =   y/(np.max(np.abs(y))+1e-6)
### CarEngineAndUrbanNoise_Dir
    ListofEngines = os.listdir(CarEngineAndUrbanNoise_Dir)
    Engines = np.zeros((len(ListofEngines),160000))
    for cpt, each_file in tqdm(enumerate(ListofEngines)):
        file = os.path.join(CarEngineAndUrbanNoise_Dir, each_file)
        y, _ = librosa.load(file,sr= 16000)
        if len(y)<160000:
             y = np.append(np.zeros(160000-len(y)), y)
        else:
             y = y[:160000]         
        Engines[cpt,:] =   y/(np.max(np.abs(y))+1e-6)
        

### Noise_Dir
    ListofNoise = os.listdir(Noise_Dir)
    Noises = np.zeros((len(ListofEngines),160000))
    for cpt, each_file in tqdm(enumerate(ListofNoise)):
        file = os.path.join(Noise_Dir, each_file)
        y, _ = librosa.load(file, sr=16000)
        if len(y)<160000:
             y = np.append(np.zeros(160000-len(y)), y)
        else:
             y = y[:160000]         
        Noises[cpt,:]  =   y/(np.max(np.abs(y))+1e-6)
        

###  Music_Dir
    ListofMusic = os.listdir(Music_Dir)
    Music = np.zeros((len(ListofMusic),160000))
    for cpt, each_file in tqdm(enumerate(ListofMusic[:2500])):
        file = os.path.join(Music_Dir, each_file)
        y, _ = librosa.load(file, sr = 16000)
        if len(y)<160000:
             y = np.append(np.zeros(160000-len(y)), y)
        else:
             y = y[:160000]         
        Music[cpt,:]  =   y/(np.max(np.abs(y))+1e-6)
        


### Wind_Dir
    ListofWind = os.listdir(Wind_Dir)
    Wind = np.zeros((len(ListofWind),160000))
    for cpt, each_file in tqdm(enumerate(ListofWind)):
        file = os.path.join(Wind_Dir, each_file)
        y, _ = librosa.load(file, sr = 16000)
        y = np.repeat(y, 10, axis = 0).flatten()
        if len(y)<160000:
             y = np.append(np.zeros(160000-len(y)), y)
        else:
             y = y[:160000]         
        Wind[cpt,:] =   y/(np.max(np.abs(y))+1e-6)
        
    
    
    return SRIR_car, SRIR_noise, Speakers, Noises ,Engines, Music, Wind

    


def read_rir(path):
     with open(path, "rb") as f:
         len = np.fromfile(f,dtype= int,count= 1)
         h = np.fromfile(f, '<f4')
     return len,h





def DataGenration_filePerfile(SRIR,Lsp_SRIR ,Speakers_Dir,Music_Dir, Mic_conf):
    ## Get the speakers 
        lenghthOfMixture = 10 #up to 10 seconds
        sampling_rate = 16000
        #print(Speakers_Dir)
        List_Speech = os.listdir(Speakers_Dir)
        List_Music  = os.listdir(Music_Dir)


        NumberOfSpeakers = 4 # evetually less sometimes 
        Chosen_Speech = random.choices(List_Speech,k=NumberOfSpeakers)
        #print(len(Chosen_Speech))
        Chosen_Music = random.choices(List_Music,k=1)

        #SRIR = load_pickle(SRIR_file)[0] # load the car impulse response
        #SRIR_Dir_Noise_Impulse =r'C:/Users/hafsa/Desktop/SoundSceneWithMcroomsim/Generation_Scenes_Sonores/Simulated_DiffuseNoise_ImpulseResponses_PklFiles/'  #'./Simulated_DiffuseNoise_ImpulseResponses_PklFiles/' 
        spk_number = ['12','13','14','15']
        if Mic_conf == 'set0':
            mic_number = ['02','03','04','05']
        elif Mic_conf == 'set1':
            mic_number = ['01','06','07','12']
        elif Mic_conf == 'set2':
            mic_number = ['03','04','09','10']
        lsp_number = ['18','19','20','21']
        """
        for mic in range(4):
            for speaker in range(4):
                file = 'C:/Users/hafsa/Downloads/v_class/impulse_responses/lsp_'+ spk_number(speaker)+ '_mic_' + mic_number(mic)+ '_enh.fir'
                _ , SRIR[mic,speaker,:] =  read_rir(file)  
        """
        

                  
     
                  
                  
                  
     
        

                  
        Contrib_Rev = np.zeros((4,11,sampling_rate*lenghthOfMixture)).astype(np.float32) # mic, source, lenght of mixture
        Contrib_Anec = np.zeros((4,5,sampling_rate*lenghthOfMixture))
        # Handle the sources 
        r = 0
        for il in range(len(Chosen_Speech)):
             #print(os.path.join(Speakers_Dir, Chosen_Speech[il]))
             y,fs = librosa.load(os.path.join(Speakers_Dir, Chosen_Speech[il]),sr=sampling_rate)
             y =  y/(np.max(np.abs(y))+1e-6)

             if len(y)<sampling_rate*lenghthOfMixture:
                 y = np.append(np.zeros(sampling_rate*lenghthOfMixture-len(y)), y)
             else:
                 y = y[:sampling_rate*lenghthOfMixture] 
                 
             
             #if il != 0:
                 
                 #if random.randint(0,10) < 4: # copilote, and other passenger might not be present 
                     #y = np.zeros((sampling_rate*lenghthOfMixture))
             """
              if random.randint(0,10) >= 7 and r == 0:
                     number1 = random.randint(0,int(160000/2))
                     number2 = random.randint(int(160000/2),int(160000))
                     y[number1:number2] = 10**-5*y[number1:number2]
                     r = 1
             """
             #print(len(y))
             for mic in range(4):
                SRIR_Anec= np.zeros((5760))
                I = np.argmax(SRIR[mic,il,:])
                SRIR_Anec[I:I+8] = SRIR[mic,il,I:I+8]
                #plt.plot(SRIR[mic,il,:])
                Contrib_Rev[mic,il,:] = olafilt(SRIR[mic,il,:],y)[:sampling_rate*lenghthOfMixture]
                #plt.plot(Contrib_Rev[mic,il,:])
                #plt.show()
                Contrib_Anec[mic,il,:] = olafilt(SRIR_Anec,y)[:sampling_rate*lenghthOfMixture]
                #sd.play(SRIR[mic,il,:], 16000)
                #sd.wait()

        #Handle the noise in a diffuse manner 
        Music = np.zeros(sampling_rate*lenghthOfMixture) 
        for il in range(len(Chosen_Music)):
             y,fs = librosa.load(os.path.join(Music_Dir, Chosen_Music[il]),sr=sampling_rate)
             y = y/(np.max(np.abs(y))+1e-6)
             
             if len(y)<sampling_rate*lenghthOfMixture:
                y = np.append(np.zeros(sampling_rate*lenghthOfMixture-len(y)), y)
             else:
                y = y[:sampling_rate*lenghthOfMixture] 
             Music = y + Music
             
             
             
        Music = Music/(np.max(np.abs(Music))+1e-6)

        #Noise = Noise/(np.max(np.abs(Noise))+1e-6)
        for il in range(4):                 
            for mic in range(SRIR.shape[0]):
                #SRIR_Anec = np.zeros()
                Contrib_Rev[mic,il+4+1,:] = olafilt(Lsp_SRIR[mic,il,:],Music)[:sampling_rate*lenghthOfMixture]
                Contrib_Anec[mic,4,:] =  Music
        DesiredSNR = np.random.uniform(low=5, high=20, size=(1,))
        
        alpha_Music = Add_noise(Contrib_Rev[0,0,:] , Contrib_Rev[0,5,:],DesiredSNR) # sum(Contrib_Rev[0,5:,:],0)
        alpha_Music = alpha_Music.astype(np.float32)
        Contrib_Rev[:,5:5+4,:] = alpha_Music*Contrib_Rev[:,5:5+4,:]        
        




        folder = './v_class/background_noise'
        folders_noise = os.listdir(folder)
        random_folder = random.choice(folders_noise)
        chosenPathFolder = os.path.join(folder,random_folder)
        name_noise = chosenPathFolder.split('/')[-1].split('\\')[-1] 
        #▲print(name_noise )
        Noise = np.zeros(sampling_rate*lenghthOfMixture) 
        
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
        start_index = random.randint(0, 30*16000 - 160000)
        
        # Calculate the end index based on the start index and the number of samples (160,000)
        end_index = start_index + 160000        

            
        for mic in range(SRIR.shape[0]):
                Noise,_ = librosa.load(os.path.join( chosenPathFolder ,'con_' + name_noise+'_mic_'+mic_number[mic]+'.wav'), sr=16000)   
                Contrib_Rev[mic,9,:] = Noise[start_index:end_index]   
        del Noise
                 
                    

        DesiredSNR = np.random.uniform(low=0, high=30, size=(1,))
        alpha = Add_noise(Contrib_Rev[0,0,:], Contrib_Rev[0,9,:] ,DesiredSNR)
        #print(f'herrre     {alpha}')    
        Contrib_Rev[:,9,:] = alpha*Contrib_Rev[:,9,:]        
        #plt.plot(Contrib_Rev[0,9,:])        
        
        #print(alpha_Music)
        Mix = np.sum(Contrib_Rev[:,:9,:],1)  +  np.sum(Contrib_Rev[:,9:,:],1) 
        
        Mix = Mix/np.max(np.abs(Mix)+1e-6)
        #print(np.isnan(Mix).any().item(), np.isnan(Contrib_Anec).any().item())
        
        return Mix, Contrib_Anec # Mixture (channels, samples) , Conrib_Rev (Receiver, source, sample)  

    
        
   
if __name__ == '__main__':

    SRIR = np.zeros((4, 4, 5760))
    spk_number = ['12','13','14','15']
    mic_number = ['02','03','04','05']
    lsp_number = ['18','19','20','21']
    for mic in range(4):
        for speaker in range(4):
            file = 'C:/Users/hafsa/Downloads/v_class/impulse_responses/lsp_'+ spk_number[speaker]+ '_mic_' + mic_number[mic]+ '_enh.fir'
            _ , SRIR[mic,speaker,:] =  read_rir(file)  
            plt.plot( SRIR[mic,speaker,:] )
            
    Lsp_SRIR = np.zeros((4, 4, 5760))
    
    for mic in range(4):
        for speaker in range(4):
            file = 'C:/Users/hafsa/Downloads/v_class/impulse_responses/lsp_'+ lsp_number[speaker]+ '_mic_' + mic_number[mic]+ '_enh.fir'
            _ , Lsp_SRIR[mic,speaker,:] =  read_rir(file)                 
   
    Speakers_Dir =  r'C:\Users\hafsa\Documents\InteractiveVoiceCommand\Src_code\Speech_Separation_Enhancement\BMW\LibriSpeech\train-clean-360\clean_speech' 
    
    Maximum = np.max( [np.max(np.abs(Lsp_SRIR)), np.max(np.abs(SRIR))] )
    
    SRIR = SRIR/Maximum
    Lsp_SRIR = Lsp_SRIR/Maximum
    
    
 

    
    #SRIR_file = os.path.join(SRIR_Dir_Car_Impulse,ChosenSRIR[0])
    Music_Dir =  './MusicFiles' 
    Mix, Cotrib = DataGenration_filePerfile(SRIR,Lsp_SRIR ,Speakers_Dir, Music_Dir) #DataGenration(SRIR_file ,Speakers_Dir,CarEngineAndUrbanNoise_Dir,Noise_Dir) 
    print(Mix.shape)
    print(Cotrib.shape)











