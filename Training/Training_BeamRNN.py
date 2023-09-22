# This code is a part of Tencent AI's Intellectual property
#----------------------------------------------------------------------------------------------------------
# Title: JOINT AEC AND BEAMFORMING with Double-Talk Detection using RNN-Transformer
# Authors: Vinay Kothapally, Yong Xu, Meng Yu, Shi-Xiong Zhang, Dong Yu
# Submitted to ICASPP 2022
#----------------------------------------------------------------------------------------------------------
# This Script is currently private and only meant to serve as a reference to the reviewers of ICASPP 2022
# This work has been conducted by Vinay Kothapally during his Internship at Tencent AI Lab
#----------------------------------------------------------------------------------------------------------




import sys 
sys.path.append('./JAECBF/src')
from JAECBF.src.model import *

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:16:34 2022

@author: hafsa
"""


import torch
from pit_criterion import *

#from torch.nn.modules.conv import LazyConvTranspose2d
import torchaudio
import torch.nn as nn

import soundfile as sf
#import stft
import os 
import fnmatch 
import random 
import numpy as np
import glob 
from tqdm import tqdm
import pickle

from numpy.random import RandomState
import matplotlib.pyplot as plt
import scipy.io
from random import randint
import scipy
import librosa
from multiprocessing import Process, Value, Array, Lock
import heapq
#from DataGeneration_server_CollectedData import *
from DataGeneration_server_vclass import *

from asteroid.losses import pairwise_neg_sisdr
from asteroid.losses import PITLossWrapper
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse







def main(args):



    Mic_conf=args.arg1 
    cuda_visible =  args.arg2
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible



    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
    SRIR_Dir_Car_Impulse = './SRIR_Car' #r'C:\Users\hafsa\Documents\InteractiveVoiceCommand\Src_code\NVISO\RecordImpulseResponse\SRIR_Car' 
    Speakers_Dir =  './LibriSpeech/train-clean-360/clean_speech' 
    Noise_Dir =  './NoiseSegmented' #r'C:\Users\hafsa\Documents\InteractiveVoiceCommand\Src_code\NVISO\RecordImpulseResponse\NoiseSegmented'
    Music_Dir =  './MusicFiles'
    Wind_Dir  = './wind'
    CarEngineAndUrbanNoise_Dir = './CarEngineFiles'
    Dir_SRIR_file_noise = './Simulated_DiffuseNoise_ImpulseResponses_PklFiles' 
    #, SRIR_noise, Speakers, Noises ,Engines, Music, Wind = load_all_dataInMemory(SRIR_Dir_Car_Impulse, Dir_SRIR_file_noise, Speakers_Dir, CarEngineAndUrbanNoise_Dir, Noise_Dir, Music_Dir, Wind_Dir)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print(device)

    #model = ConvTasNet(num_sources=4 ,enc_kernel_size = 256, enc_num_feats = 512, msk_kernel_size = 3, msk_num_hidden_feats = 512, msk_num_layers = 5, msk_num_stacks = 3,NumberOfChannels = 4, MtoN_channels=  32, ico_channels = 32,  hiddenc_channels= 32).cuda()
    model = RNNBF(L=512, N=256, X=4, R=4, B=256, H=512, P=3, F=256, cos=True, ipd="").to('cuda')
    
    if Mic_conf == 'set0':
        checkpoint = torch.load('./Models/'+str(32000)+'BestModel_BeamRNN.pth.tar' , map_location=torch.device(device)   )
        model.load_state_dict(checkpoint)

    elif Mic_conf == 'set1':
        checkpoint = torch.load('./Models/'+str(32000)+'BestModel_BeamRNN_set1.pth.tar' , map_location=torch.device(device)   )
        model.load_state_dict(checkpoint)

    elif Mic_conf == 'set2':
        checkpoint = torch.load('./Models/'+str(32000)+'BestModel_BeamRNN_set2.pth.tar' , map_location=torch.device(device)   )
        model.load_state_dict(checkpoint)
    
    """
    checkpoint = torch.load('../BestModel_TASNET_ICASSP_MVA.pth.tar', map_location=torch.device('cuda'))
    checkpoint = checkpoint.state_dict()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    model.load_state_dict(checkpoint)
    
    #model = torch.nn.DataParallel(model)
    model.cuda()
    """
    
    optimizer = torch.optim.Adam(model.parameters(),
                                      lr=0.0001,
                                      weight_decay=0.0)
    #model = load_model("best_model_custom_IRM_mobile.h5")
    
   
    loss_glob =1000000    
    loss =1000000    
    List_RT60 = []
    NumberOfIteration = 500

    save_dir = "./ValidpklFiles_BeamRNN_"+Mic_conf+"/"
    best_loss = 1000
    n_epochs = 100
    loss_train = np.zeros(shape = (n_epochs,), dtype=np.float32)
    loss_valid = np.zeros(shape = (n_epochs,), dtype=np.float32)
    loss_val = np.zeros(shape = (n_epochs,))
    Batch_size = 10
    training = True
    
    SRIR = np.zeros((4, 4, 1919))
    spk_number = ['12','13','14','15']
    if Mic_conf == 'set0':
        mic_number = ['02','03','04','05']
    elif Mic_conf == 'set1':
        mic_number = ['01','06','07','12']
    elif Mic_conf == 'set2':
        mic_number = ['03','04','09','10']
    lsp_number = ['18','19','20','21']
    
    target_sampling_rate = 16000
    
    # Calculate the ratio of original to target sampling rate
    resampling_ratio = target_sampling_rate / 48000

# Calculate the new length of the resampled signal



        
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
    
    if training== True: 
        for epoch in range(100):
            print(f'*************************************** Epoch : {epoch} *********************************** ')
            print(f'****************** Training phase ************************* ')
            """
            List_RT60 = os.listdir(SRIR_Dir_Car_Impulse)
            List_RT60*3
            random.shuffle(List_RT60)
            """
            step_batch = int(1)
            cpt = 1
            loss_glob = 0
            model.train() 
            batch_Sequence = 10
            cpt_SRIR = 0
            NumberOfExamples = 250
            with tqdm(range(0,NumberOfExamples, step_batch), unit="batch") as pbar:
                for ibatch in pbar:
                    """
                    SRIR_file  = []
                    for il in range(4):
                        Seat = 'Seat'+str(il+1)
                        listOfDir = os.listdir(os.path.join(SRIR_Dir_Car_Impulse, Seat))
                        ChosenSRIR = random.choices(listOfDir,k=1)
                        SRIR_file.append(os.path.join(SRIR_Dir_Car_Impulse,Seat,ChosenSRIR[0]))
                    """
                    #Mix, Cotrib = DataGenration(SRIR_car, SRIR_noise, Speakers, Noises ,Engines, Music, Wind, cpt_SRIR)#DataGenration(SRIR_file ,Speakers_Dir,CarEngineAndUrbanNoise_Dir,Noise_Dir, Music_Dir)
                    #Mix, Cotrib = DataGenration_filePerfile(SRIR_file ,Speakers_Dir, Music_Dir, Noise_Dir)
                    
                    Mix, Cotrib =DataGenration_filePerfile(SRIR,Lsp_SRIR ,Speakers_Dir, Music_Dir, Mic_conf)
                    Contrib = np.empty((4, 160000))
                    for il in range(Contrib.shape[0]):
                            Contrib[il,:]=  Cotrib[0,il,:]                    
                    
                    
                    """
                    Contrib = np.empty((4, 160000))
                    for il in range(Contrib.shape[0]):
                        if il ==0 or il == 1:
                            Contrib[il,:]=  Cotrib[il,il,:]
                        else:
                            Contrib[il,:]=  Cotrib[2,il,:]
                    """

                    

                    Input =  torch.from_numpy(    np.concatenate((np.expand_dims(Mix[0:4,:],0), np.expand_dims(np.expand_dims(Cotrib[0,4,:],0), 1) ), 1)         ).float().to(device)   
                    


                    Output_SS =  torch.from_numpy(np.expand_dims(Contrib,0)).to(device)                     
                    
                    for bil in range(5):
                        Input_ = Input[:,:, bil*16000*2  :bil*16000*2 + 16000*2]    
                        Output_SS_  = Output_SS [:,:,bil*16000*2  :bil*16000*2 + 16000*2]



                        """
                        Input_ = Input_.permute(1,0,2)
                        numberOfSamples = int(2*16000)
                        batch_size = int(2*16000/numberOfSamples)
                        Input_ = Input_.reshape(4,batch_size,numberOfSamples)
                        Input_ = Input_.permute(1,0,2)
                        
                        Output_SS_ = Output_SS_.permute(1,0,2)
                        Output_SS_ = Output_SS_.reshape(4,batch_size,numberOfSamples)
                        Output_SS_ = Output_SS_.permute(1,0,2)
                        """
                        
                        
                        if not torch.isnan(Input_).any().item() and  not torch.isnan(Output_SS_).any().item() :   
                            
                            if epoch == -1:
                                pass


                            if epoch == 0:
                                    if not(os.path.exists(save_dir)):
                                        os.mkdir(save_dir)
                                    with open(save_dir+str(cpt)+'.pkl','wb') as f: 
                                        pickle.dump([Input_,Output_SS_],f)
                                    cpt_SRIR = cpt_SRIR+1
                            if epoch >=1:
                                
                                #print(Input_[:,:2,:].shape)
                                #print(Input_[:,3,:].shape)
                                SS , bf_enhanced_mag = model(Input_[:,:4,:], Input_[:,4,:], verbose=False) 
                                mixture_lengths = torch.from_numpy(np.array([16000*2*1]))   
                                loss1 = torch.zeros(4)
                                loss2 = torch.zeros(4)
                                for il in range(4):
                                    loss1[il], max_snr, estimate_source, reorder_estimate_source =  cal_loss(Output_SS_[:,il,:].unsqueeze(1), SS[:,il,:].unsqueeze(1), mixture_lengths.to(device) )
                                loss2 =  loss_func(Output_SS_[:,:,:], SS[:,:,:])
                                loss1 = torch.mean(loss1)
                                #loss2 = torch.mean(loss2)
                                #loss2, max_snr, estimate_source, reorder_estimate_source = cal_loss(Output_SS[:,:,:], SS, mixture_lengths.to(device)) 
                                """
                                if cpt%50 == 0:
                                    plt.plot(Input[0,0,:].cpu().detach().numpy())
                                    plt.savefig('Input.png')
                                    plt.show()                        
                                    for il in range(4):
                                        plt.plot(SS[0,il,:].cpu().detach().numpy())
                                        plt.savefig(str(il)+'.png')
                                        plt.show()
                                """
            
                                loss =(loss1+ loss2)/2
                        
                                optimizer.zero_grad()
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                               5)
                                optimizer.step()
                                cpt_SRIR = cpt_SRIR+1
                                    
                            loss_glob =loss+loss_glob
                            cpt = cpt+1

                        else:
                            print(torch.isnan(Input_).any().item() , torch.isnan(Output_SS_).any().item() )  

                    pbar.set_postfix(loss = loss_glob/cpt)
                    
            loss_train[epoch] = loss_glob/cpt
            #epoch_loss_avg.reset_states()   
            model.eval()
            DirVal = os.listdir(save_dir)
            cpt = 0
            loss_glob = 0
            
            print(f'****************** Validation phase ************************* ')
            with torch.no_grad():
                with tqdm(range(0,int(len(DirVal))), unit="batch") as pbar: #
                    for il in pbar:
                        file = DirVal[il]
                        Input, Output_SS  =  load_pickle(os.path.join(save_dir, file))

                        SS , bf_enhanced_mag = model(Input[:,:4,:], Input[:,4,:], verbose=False)                         
                        mixture_lengths = torch.from_numpy(np.array([16000*2*1]))
                        loss1 = torch.zeros(4)
                        loss2 = torch.zeros(4)
                        for il in range(4):
                            loss1[il], max_snr, estimate_source, reorder_estimate_source =  cal_loss(Output_SS[:,il,:].unsqueeze(1), SS[:,il,:].unsqueeze(1), mixture_lengths.to(device) )
                        loss2 =  loss_func(Output_SS[:,:,:], SS[:,:,:])
                        loss1 = torch.mean(loss1)
                        #loss2 = torch.mean(loss2)
                        loss = (loss1.to(device)+loss2.to(device))/2
                        #loss2 = torch.mean(loss2)
                        #loss = (loss1+loss2)/2
                        loss_glob =loss+loss_glob
                        cpt = cpt+1
                        pbar.set_postfix( loss_valid = loss_glob/cpt)
                loss_valid[epoch] = loss_glob/cpt
                #epoch_loss_avg.reset_states()  
                if loss_valid[epoch] < best_loss:
                    #print(f'*************The model has been saved, the loss decreased from best_loss  to loss_valid[epoch] ')
                     #model.save("best_model_custom_IRM_mobile_SDR.h5")
                    #Wmodel.compute_output_shape(input_shape=(1,16000,1))
                    file_path = os.path.join('./'+str(16000*2)+'BestModel_BeamRNN_'+Mic_conf+'.pth.tar' )
                    torch.save(model.state_dict(), file_path)
                    print(f"\n[INFO]\tNew best loss (from {best_loss:0.2f} to {loss_valid[epoch]:0.2f}).. Model saved.")
                    best_loss = loss_valid[epoch]
            print(f"\nEpoch {epoch+1}/{n_epochs}:\n\tTrain:\t\tloss:\t\t{loss_train[epoch]:0.3f}.\n")





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="A script that demonstrates argparse with main()")
    
    # Define arguments
    parser.add_argument('--arg1', default = 'set0' ,type=str, help='The first argument')
    parser.add_argument('--arg2', default = '0' ,type=str, help='The first argument')
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    print('hello first')
    main(args)    


