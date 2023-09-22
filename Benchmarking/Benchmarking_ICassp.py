import sys 
sys.path.append('./JAECBF/src')
from JAECBF.src.model import *
#import sounddevice as sd
import sys 
sys.path.append('./Wave_U_Net_Pytorch/')
from Wave_U_Net_Pytorch.model.waveunet import *
import torch
from pit_criterion import *
import  InterConvTasnet_Icassp  
import  InterConvTasnet_Icassp_v2
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
from DataGeneration_server_CollectedData_forBenchmarking import *
#from asteroid.losses import pairwise_neg_sisdr
#from asteroid.losses import PITLossWrapper
import soundfile as sf
import fast_bss_eval
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import argparse
import Levenshtein
import os
from pesq import pesq 

from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
si_snr = ScaleInvariantSignalNoiseRatio()
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
si_sdr = ScaleInvariantSignalDistortionRatio()
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio



os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'




def main(args):



    import numpy as np
    print('hello')
    Mic_conf=args.arg1 
    ContentType=args.arg2
    gpu=args.arg3
    SNR=args.arg4
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model_wave2ec = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    models = ['JAECBF', 'WaveUnet', 'IC-ConvTas', 'Mix', 'Truth']
    metrics = ['si-sdr', 'si-snr', 'wer', 'pesq_nb',  'pesq_wb', 'sdr', 'sar', 'sir']
    seats = ['Seat 0', 'Seat 1', 'Seat 2', 'Seat 3']
    
    results = {metric: {model: {seat: [] for seat in seats} for model in models} for metric in metrics}
    
    
    
    
    
    
    
    
    target_sampling_rate = 16000
    
    # Calculate the ratio of original to target sampling rate
    resampling_ratio = target_sampling_rate / 48000
    
    
    SRIR = np.zeros((4, 4, 1919))
    spk_number = ['12','13','14','15']
    if Mic_conf == 'set0':
        mic_number = ['02','03','04','05']
    elif Mic_conf == 'set1':
        mic_number = ['01','06','07','12']
    elif Mic_conf == 'set2':
        mic_number = ['03','04','09','10']
        
        
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
  
    #loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
    SRIR_Dir_Car_Impulse = './SRIR_Car' #r'C:\Users\hafsa\Documents\InteractiveVoiceCommand\Src_code\NVISO\RecordImpulseResponse\SRIR_Car' 
    Noise_Dir =  './NoiseSegmented' #r'C:\Users\hafsa\Documents\InteractiveVoiceCommand\Src_code\NVISO\RecordImpulseResponse\NoiseSegmented'
    if ContentType == 'Music':
        Music_Dir =  './MusicFiles'
    else:
        Music_Dir =  '././LibriSpeech/train-clean-360/clean_speech'
    Wind_Dir  = './wind'
    CarEngineAndUrbanNoise_Dir = './CarEngineFiles'
    Dir_SRIR_file_noise = './Simulated_DiffuseNoise_ImpulseResponses_PklFiles' 
    #, SRIR_noise, Speakers, Noises ,Engines, Music, Wind = load_all_dataInMemory(SRIR_Dir_Car_Impulse, Dir_SRIR_file_noise, Speakers_Dir, CarEngineAndUrbanNoise_Dir, Noise_Dir, Music_Dir, Wind_Dir)

    ToTest = 'True'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print(device)

    print(Mic_conf)
    numberOfSamples = 32000

    if ToTest == 'True':
        
        model_RNN = RNNBF(L=512, N=256, X=4, R=4, B=256, H=512, P=3, F=256, cos=True, ipd="").to(device)
        if Mic_conf == 'set0':
            checkpoint = torch.load('../Training/Models/'+str(numberOfSamples)+'BestModel_BeamRNN_set0.pth.tar' , map_location=torch.device(device)   )
        elif Mic_conf == 'set1':
            checkpoint = torch.load('../Training/Models/'+str(numberOfSamples)+'BestModel_BeamRNN_set1.pth.tar' , map_location=torch.device(device)   )
        elif Mic_conf == 'set2':
            checkpoint = torch.load('../Training/Models/'+str(numberOfSamples)+'BestModel_BeamRNN_set2.pth.tar' , map_location=torch.device(device)   )        
        
        
        


        
        
        #checkpoint = torch.load('./'+str(numberOfSamples)+'BestModel_BeamRNN.pth.tar' , map_location=torch.device(device))
        model_RNN.load_state_dict(checkpoint)  
        model_RNN.eval()
    #elif ToTest == 'True':

        feature_growth = 'double'
        levels = 6
        features = 32
        output_size = 1
        sr = 32633
        channels = 5
        instruments = ["bass", "drums", "other", "vocals"]
        kernel_size = 5
        depth = 1
        strides = 2
        conv_type = 'gn'
        res = 'fixed'
        separate = 0
        
        num_features = [ features*i for i in range(1,  levels+1)] if  feature_growth == "add" else \
                       [ features*2**i for i in range(0,  levels)]
        target_outputs = int( output_size *  sr)
        
        model_WAveUnet = Waveunet( channels, num_features,  channels,  instruments, kernel_size= kernel_size,target_output_size=target_outputs, depth= depth, strides= strides,conv_type= conv_type, res= res, separate= separate).to(device)  
        numberOfSamples = 32633     
        if Mic_conf == 'set0':
            checkpoint = torch.load('../Training/Models/'+str(numberOfSamples)+'BestModel_WaveUnet_set0.pth.tar' , map_location=torch.device(device)   )
        elif Mic_conf == 'set1':
            checkpoint = torch.load('../Training/Models/'+str(numberOfSamples)+'BestModel_WaveUnet_set1.pth.tar' , map_location=torch.device(device)   )
        elif Mic_conf == 'set2':
            checkpoint = torch.load('../Training/Models/'+str(numberOfSamples)+'BestModel_WaveUnet_set2.pth.tar' , map_location=torch.device(device)   )




        #checkpoint = torch.load('./'+str(numberOfSamples)+'BestModel_WaveUnet.pth.tar' , map_location=torch.device(device)   )
        model_WAveUnet.load_state_dict(checkpoint)           
        model_WAveUnet.eval()
    #elif  ToTest == 'True':
        numberOfSamples = 32000
        
        if Mic_conf == 'set0':
            model_InterChannel = InterConvTasnet_Icassp.ConvTasNet(num_sources=4 ,enc_kernel_size = 256, enc_num_feats = 512, msk_kernel_size = 3, msk_num_hidden_feats = 512, msk_num_layers = 5, msk_num_stacks = 3,NumberOfChannels = 5, MtoN_channels=  32, ico_channels = 32,  hiddenc_channels= 32).to(device)
            checkpoint = torch.load('../Training/Models/'+str(numberOfSamples)+'BestModel_InterChannel_set0.pth.tar' , map_location=torch.device(device)   )
        elif Mic_conf == 'set1':
            model_InterChannel = InterConvTasnet_Icassp.ConvTasNet(num_sources=4 ,enc_kernel_size = 256, enc_num_feats = 512, msk_kernel_size = 3, msk_num_hidden_feats = 512, msk_num_layers = 5, msk_num_stacks = 3,NumberOfChannels = 5, MtoN_channels=  32, ico_channels = 32,  hiddenc_channels= 32).to(device)
            
            checkpoint = torch.load('../Training/Models/'+str(numberOfSamples)+'BestModel_InterChannel_set1.pth.tar' , map_location=torch.device(device)   )
        elif Mic_conf == 'set2':
            model_InterChannel = InterConvTasnet_Icassp.ConvTasNet(num_sources=4 ,enc_kernel_size = 256, enc_num_feats = 512, msk_kernel_size = 3, msk_num_hidden_feats = 512, msk_num_layers = 5, msk_num_stacks = 3,NumberOfChannels = 5, MtoN_channels=  32, ico_channels = 32,  hiddenc_channels= 32).to(device)

            checkpoint = torch.load('../Training/Models/'+str(numberOfSamples)+'BestModel_InterChannel_set2.pth.tar' , map_location=torch.device(device)   )

        #checkpoint = torch.load('./'+str(numberOfSamples)+'BestModel_InterChannel.pth.tar' , map_location=torch.device(device))
        model_InterChannel.load_state_dict(checkpoint)   
        model_InterChannel.eval()

    NumberOfMixtures = 1
    for ispeech in tqdm(range (NumberOfMixtures)): 


        try: 
               # print(f'****************iteration number {ispeech}**************.')
                SRIR_file  = []
                '''
                for il in range(4):
                    Seat = 'Seat'+str(il+1)
                    listOfDir = os.listdir(os.path.join(SRIR_Dir_Car_Impulse, Seat))
                    ChosenSRIR = random.choices(listOfDir,k=1)
                    SRIR_file.append(os.path.join(SRIR_Dir_Car_Impulse,Seat,ChosenSRIR[0]))
                '''
                #Mix, Cotrib = DataGenration(SRIR_car, SRIR_noise, Speakers, Noises ,Engines, Music, Wind, cpt_SRIR)#DataGenration(SRIR_file ,Speakers_Dir,CarEngineAndUrbanNoise_Dir,Noise_Dir, Music_Dir)
                Mix, Cotrib, Source_path, lenghthOfMixture, Y = DataGenration_filePerfile(SRIR,Lsp_SRIR ,Speakers_Dir, Music_Dir, 4, Mic_conf, SNR) #DataGenration(SRIR_file ,Speakers_Dir,CarEngineAndUrbanNoise_Dir,Noise_Dir) 
                Contrib = np.empty((4, lenghthOfMixture))
                for il in range(Contrib.shape[0]):
                      if Mic_conf == "set0" or Mic_conf == "set1"or Mic_conf == "set2":  
                            Contrib[il,:]=  Cotrib[0,il,:]
                            sf.write(ContentType+'Truth_'+str(il)+'_'+Mic_conf+'_'+str(SNR)+'.wav',Contrib[il,:],16000)
                            sf.write(ContentType+'Mix_'+str(il)+'_'+Mic_conf+'_'+str(SNR)+'.wav',Mix[il,:],16000)
                      else: 
                            Contrib[il,:]=  Y[0,il,:]
                sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources( Contrib, Mix[:4,:])
                
                for seat in range(4):
                    
                    results['sdr']['Mix'][seats[seat]].append(sdr[seat])
                    results['sir']['Mix'][seats[seat]].append(sir[seat])
                    results['sar']['Mix'][seats[seat]].append(sar[seat])
                    results['pesq_nb']['Mix'][seats[seat]].append(pesq(16000, Contrib[seat,:], Mix[seat,:], 'nb')   )
                    results['si-snr']['Mix'][seats[seat]].append(scale_invariant_signal_noise_ratio( torch.from_numpy(Mix[seat,:]), torch.from_numpy(Contrib[seat,:]) ) )


                print(pesq(16000, Contrib[seat,:], Mix[seat,:], 'wb'))
                
                sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources( Contrib,Contrib)
                
                for seat in range(4):
                    
                    results['sdr']['Truth'][seats[seat]].append(sdr[seat])
                    results['sir']['Truth'][seats[seat]].append(sir[seat])
                    results['sar']['Mix'][seats[seat]].append(sar[seat])
                    results['pesq_nb']['Truth'][seats[seat]].append(pesq(16000, Contrib[seat,:], Contrib[seat,:], 'nb')   )
                    results['si-snr']['Truth'][seats[seat]].append(scale_invariant_signal_noise_ratio( torch.from_numpy(Contrib[seat,:]), torch.from_numpy(Contrib[seat,:]) ) )


                for il in range(4):
                    input_values = processor(Mix[il,:], sampling_rate=16000, return_tensors="pt").input_values
                    logits = model_wave2ec(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription_estim = processor.decode(predicted_ids[0])                    
                    
                    audio_path = Source_path[il]
                    audio_filename = os.path.splitext(os.path.basename(audio_path))[0].rsplit('-', 1)[0]

                    transcription_dir = os.path.dirname(audio_path)
                    transcription_file_path = os.path.join(transcription_dir, f'{audio_filename}.trans.txt')
                    with open(transcription_file_path, 'r') as trans_file:
                        for line in trans_file:
                            parts = line.strip().split()
                            if parts[0] == os.path.basename(audio_path).split('.')[0]:
                                transcription_truth = ' '.join(parts[1:])
                    wer_score = wer(transcription_truth, transcription_estim)
                    print(wer_score)
                    if wer_score > 100:
                        wer_score  = 100
                    results['wer']['Mix'][seats[il]].append(wer_score)        
                

                for il in range(4):
                    input_values = processor(Contrib[il,:], sampling_rate=16000, return_tensors="pt").input_values
                    logits = model_wave2ec(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription_estim = processor.decode(predicted_ids[0])                    
                    
                    audio_path = Source_path[il]
                    audio_filename = os.path.splitext(os.path.basename(audio_path))[0].rsplit('-', 1)[0]

                    transcription_dir = os.path.dirname(audio_path)
                    transcription_file_path = os.path.join(transcription_dir, f'{audio_filename}.trans.txt')
                    with open(transcription_file_path, 'r') as trans_file:
                        for line in trans_file:
                            parts = line.strip().split()
                            if parts[0] == os.path.basename(audio_path).split('.')[0]:
                                transcription_truth = ' '.join(parts[1:])
                    wer_score = wer(transcription_truth, transcription_estim)
                    print(wer_score)
                    if wer_score > 100:
                        wer_score  = 100
                    results['wer']['Truth'][seats[il]].append(wer_score)      


                #Input =  torch.from_numpy(    np.concatenate((np.expand_dims(Mix[0:2,:],0), np.expand_dims(np.expand_dims(Mix[4,:],0), 1), np.expand_dims(np.expand_dims(Cotrib[4,4,:],0), 1) ), 1)         ).float().to(device)   



                Output_SS =  torch.from_numpy(np.expand_dims(Contrib,0)).to(device)                     




                SS = np.zeros((4,lenghthOfMixture))

                with torch.no_grad(): 
                    
                    if ToTest =='True':
                        model_results = {}
                        Input =  torch.from_numpy(    np.concatenate((np.expand_dims(Mix[0:4,:],0), np.expand_dims(np.expand_dims(Cotrib[0,4,:],0), 1) ), 1)         ).float().to(device)   
                        for bil in range(int(lenghthOfMixture/(16000*2))):
                            Input_ = Input[:,:, bil*16000*2  :bil*16000*2 + 16000*2]    
                            #Input_ = Input_.permute(1,0,2)
                            #numberOfSamples = int(16000)
                            #batch_size = int(16000/numberOfSamples)
                            
                            #Input_ = Input_.reshape(4,batch_size,numberOfSamples)
                            #Input_ = Input_.permute(1,0,2)
                            SS_ , bf_enhanced_mag = model_RNN(Input_[:,:4,:], Input_[:,4,:], verbose=False)
                            SS[:, bil*16000*2  :bil*16000*2 + 16000*2] = SS_.cpu().detach().numpy() 
                        if   lenghthOfMixture % 16000*2 > 0:
                            
                            Input_ = Input[:,:, lenghthOfMixture -16000*2  :]    
                            #Input_ = Input_.permute(1,0,2)
                            #numberOfSamples = int(16000)
                            #batch_size = int(16000/numberOfSamples)
                            
                            #Input_ = Input_.reshape(4,batch_size,numberOfSamples)
                            #Input_ = Input_.permute(1,0,2)
                            SS_ , bf_enhanced_mag = model_RNN(Input_[:,:4,:], Input_[:,4,:], verbose=False)
                            SS[:, lenghthOfMixture -16000*2  :] = SS_.cpu().detach().numpy()                     
                                            
                        sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(Contrib, SS)
                        print(sdr)
                        for seat in range(4):
                            sf.write(ContentType+'BeamRNN_'+str(seat)+'_'+Mic_conf+'_'+str(SNR)+'.wav', SS[seat,:]/np.max(np.abs(SS[seat,:])), 16000)
                            results['pesq_nb']['JAECBF'][seats[seat]].append(pesq(16000, Contrib[seat,:], SS[seat,:], 'nb')   )
                            results['sdr']['JAECBF'][seats[seat]].append(sdr[seat])
                            results['sir']['JAECBF'][seats[seat]].append(sir[seat])
                            results['sar']['JAECBF'][seats[seat]].append(sar[seat])
                            results['si-snr']['JAECBF'][seats[seat]].append(scale_invariant_signal_noise_ratio( torch.from_numpy(SS[seat,:]), torch.from_numpy(Contrib[seat,:]) ) )
                            print(f' si-snr: {scale_invariant_signal_noise_ratio( torch.from_numpy(SS[seat,:]), torch.from_numpy(Contrib[seat,:]) )}')
                        print(pesq(16000, Contrib[seat,:], SS[seat,:], 'wb'))                    






         
                        for il in range(4):
                            input_values = processor(SS[il,:], sampling_rate=16000, return_tensors="pt").input_values
                            logits = model_wave2ec(input_values).logits
                            predicted_ids = torch.argmax(logits, dim=-1)
                            transcription_estim = processor.decode(predicted_ids[0])                    
                            
                            audio_path = Source_path[il]
                            audio_filename = os.path.splitext(os.path.basename(audio_path))[0].rsplit('-', 1)[0]

                            transcription_dir = os.path.dirname(audio_path)
                            transcription_file_path = os.path.join(transcription_dir, f'{audio_filename}.trans.txt')
                            with open(transcription_file_path, 'r') as trans_file:
                                for line in trans_file:
                                    parts = line.strip().split()
                                    if parts[0] == os.path.basename(audio_path).split('.')[0]:
                                        transcription_truth = ' '.join(parts[1:])
                            wer_score = wer(transcription_truth, transcription_estim)
                            if wer_score > 100:
                                wer_score  = 100
                            results['wer']['JAECBF'][seats[il]].append(wer_score)

            
                    #elif ToTest == 'True':
                        Input =  torch.from_numpy(    np.concatenate((np.expand_dims(Mix[0:4,:],0), np.expand_dims(np.expand_dims(Cotrib[0,4,:],0), 1) ), 1)         ).float()

                        for start_idx in range(0, Input.shape[2] - 32009 + 1, int(32009/2)):
                            
                            end_idx = start_idx+32009+312
                            if start_idx == 0:
                                input_segment = Input[:, :, start_idx:end_idx]
                                Input_ = np.concatenate((np.zeros((1, 5, 312)), input_segment), axis=2)
                            else:
                                input_segment = Input[:, :, start_idx:end_idx]
                                Input_ = np.concatenate((Input[:,:,start_idx-312:start_idx], input_segment), axis=2)
                            
                            predicted_output_segment = model_WAveUnet(torch.from_numpy(Input_).float().to(device)   )

                            #predicted_output_segment = model(Input_[:,:3,:], Input_[:,3,:], verbose=False)
                            SS[:, start_idx:start_idx+32009] = predicted_output_segment.cpu()


                            
                        input_segment = Input[:, :, Input.shape[2]-32009-312:Input.shape[2]]
                        Input_ = np.concatenate((input_segment, np.ones((1, 5, 312))), axis=2)
                        predicted_output_segment = model_WAveUnet(torch.from_numpy(Input_[:,:,:]).float().to(device)   ) 
                        SS[:, Input.shape[2]-32009:Input.shape[2]] = predicted_output_segment.cpu()
                        #SS = SS/np.max(np.abs(SS),1)[:,np.newaxis]
                        #Contrib = Contrib/np.max(np.abs(Contrib),1)[:,np.newaxis]

                        sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources( Contrib, SS)
                        print(sdr)
                        print(perm)    

                        for seat in range(4):
                            sf.write(ContentType+'WaveUnet_'+str(seat)+'_'+Mic_conf+'_'+str(SNR)+'.wav',SS[seat,:]/np.max(np.abs(SS[seat,:])),16000)
                            results['pesq_nb']['WaveUnet'][seats[seat]].append(pesq(16000, Contrib[seat,:], SS[seat,:], 'nb')   )
                            results['sdr']['WaveUnet'][seats[seat]].append(sdr[seat])
                            results['sir']['WaveUnet'][seats[seat]].append(sir[seat])
                            results['sar']['WaveUnet'][seats[seat]].append(sar[seat])
                            results['si-snr']['WaveUnet'][seats[seat]].append(scale_invariant_signal_noise_ratio( torch.from_numpy(SS[seat,:]), torch.from_numpy(Contrib[seat,:]) ) )
                            print(f' si-snr: {scale_invariant_signal_noise_ratio( torch.from_numpy(SS[seat,:]), torch.from_numpy(Contrib[seat,:]) )}')
                        print(pesq(16000, Contrib[seat,:], SS[seat,:], 'wb'))      
                        """
                        for seat in range(4):

                        """
                        for il in range(4):
                            input_values = processor(SS[il,:], sampling_rate=16000, return_tensors="pt").input_values
                            logits = model_wave2ec(input_values).logits
                            predicted_ids = torch.argmax(logits, dim=-1)
                            transcription_estim = processor.decode(predicted_ids[0])                    
                            
                            audio_path = Source_path[il]
                            audio_filename = os.path.splitext(os.path.basename(audio_path))[0].rsplit('-', 1)[0]

                            transcription_dir = os.path.dirname(audio_path)
                            transcription_file_path = os.path.join(transcription_dir, f'{audio_filename}.trans.txt')
                            with open(transcription_file_path, 'r') as trans_file:
                                for line in trans_file:
                                    parts = line.strip().split()
                                    if parts[0] == os.path.basename(audio_path).split('.')[0]:
                                        transcription_truth = ' '.join(parts[1:])
                            wer_score = wer(transcription_truth, transcription_estim)
                            print(wer_score)
                            if wer_score > 100:
                                wer_score  = 100
                            results['wer']['WaveUnet'][seats[il]].append(wer_score)
                            #sd.play(SS[0,:],16000)
                            #sd.wait()
                    
                    #elif ToTest == 'True':
            

                        Input =  torch.from_numpy(    np.concatenate((np.expand_dims(Mix[0:4,:],0), np.expand_dims(np.expand_dims(Cotrib[0,4,:],0), 1) ), 1)         ).float().to(device)   
                        for bil in range(int(lenghthOfMixture/(16000*2))):
                            Input_ = Input[:,:, bil*16000*2  :bil*16000*2 + 16000*2]    
                            #Input_ = Input_.permute(1,0,2)
                            #numberOfSamples = int(16000)
                            #batch_size = int(16000/numberOfSamples)
                            
                            #Input_ = Input_.reshape(4,batch_size,numberOfSamples)
                            #Input_ = Input_.permute(1,0,2)
                            SS_ = model_InterChannel(Input_[:,:,:].to(device))[0,:,:]
                            SS[:, bil*16000*2  :bil*16000*2 + 16000*2] = SS_.cpu().detach().numpy() 
                        if   lenghthOfMixture % 16000*2 > 0:
                            
                            Input_ = Input[:,:, lenghthOfMixture -16000*2  :]    
                            #Input_ = Input_.permute(1,0,2)
                            #numberOfSamples = int(16000)
                            #batch_size = int(16000/numberOfSamples)
                            
                            #Input_ = Input_.reshape(4,batch_size,numberOfSamples)
                            #Input_ = Input_.permute(1,0,2)
                            SS_ = model_InterChannel(Input_[:,:,:])[0,:,:]
                            SS[:, lenghthOfMixture -16000*2  :] = SS_.cpu().detach().numpy()                            
                            
                        sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(Contrib, SS)
                        print(sdr)
                        """
                        for seat in  range(4):

                        """


                        for seat in range(4):
                            sf.write(ContentType+'IC-ConvTas_'+str(seat)+'_'+Mic_conf+'_'+str(SNR)+'.wav',SS[seat,:]/np.max(np.abs(SS[seat,:])),16000)
                            
                            results['pesq_nb']['IC-ConvTas'][seats[seat]].append(pesq(16000, Contrib[seat,:], SS[seat,:], 'nb')   )
                            results['sdr']['IC-ConvTas'][seats[seat]].append(sdr[seat])
                            results['sir']['IC-ConvTas'][seats[seat]].append(sir[seat])
                            results['sar']['IC-ConvTas'][seats[seat]].append(sar[seat])
                            results['si-snr']['IC-ConvTas'][seats[seat]].append(scale_invariant_signal_noise_ratio( torch.from_numpy(SS[seat,:]), torch.from_numpy(Contrib[seat,:]) ) )
                            print(f' si-snr: {scale_invariant_signal_noise_ratio( torch.from_numpy(SS[seat,:]), torch.from_numpy(Contrib[seat,:]) )}')
                        print(pesq(16000, Contrib[seat,:], SS[seat,:], 'wb'))      
                        

                        for il in range(4):
                            input_values = processor(SS[il,:], sampling_rate=16000, return_tensors="pt").input_values
                            logits = model_wave2ec(input_values).logits
                            predicted_ids = torch.argmax(logits, dim=-1)
                            transcription_estim = processor.decode(predicted_ids[0])                    
                            
                            audio_path = Source_path[il]
                            audio_filename = os.path.splitext(os.path.basename(audio_path))[0].rsplit('-', 1)[0]

                            transcription_dir = os.path.dirname(audio_path)
                            transcription_file_path = os.path.join(transcription_dir, f'{audio_filename}.trans.txt')
                            with open(transcription_file_path, 'r') as trans_file:
                                for line in trans_file:
                                    parts = line.strip().split()
                                    if parts[0] == os.path.basename(audio_path).split('.')[0]:
                                        transcription_truth = ' '.join(parts[1:])
                            wer_score = wer(transcription_truth, transcription_estim)
                            
                            if wer_score > 100:
                                wer_score  = 100
                            results['wer']['IC-ConvTas'][seats[il]].append(wer_score)                

        except:
            print('Problem occured !')



    import pickle
    #file_name = Mic_conf +'_'+ ContentType +'_results.pickle'
    
    # Open the file in binary write mode ('wb')
    with open(file_name, 'wb') as file:
        pickle.dump(results, file)    
    import matplotlib.pyplot as plt
    import numpy as np
    metricsi = ['sdr', 'wer']    
    # Set figure size and create a subplots grid
    fig, axs = plt.subplots(len(metricsi), len(seats), figsize=(15, 15), sharey=True)

    # Iterate over metrics
    for i, metric in enumerate(metricsi):
        # Set a title for the metric
        axs[i, 0].set_ylabel(metric,  fontsize=20)

        # Iterate over seats
        for j, seat in enumerate(seats):
            # Data for the boxplot
            data = [results[metric][model][seat] for model in models]

            # Create boxplot
            box = axs[i, j].boxplot(data, patch_artist=True)

            # Customize box colors
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
            for median in box['medians']:
                median.set(color='black')
            # Add title for each subplot
            axs[i, j].set_title(seat, fontsize=20)

            # Set x-axis ticks
            axs[i, j].set_xticklabels(models, fontsize=10)
            axs[i, j].tick_params(axis='y', labelsize=20)  # Adjust fontsize as needed    



    # Adjust the layout
    plt.tight_layout()
    
    # Show the plot
    plt.savefig(Mic_conf+'_'+ ContentType + '.png')
    plt.show()
        
        
    metricsi =metrics   
    # Set figure size and create a subplots grid
    fig, axs = plt.subplots(len(metricsi), len(seats), figsize=(15, 15), sharey=True)

    # Iterate over metrics
    for i, metric in enumerate(metricsi):
        # Set a title for the metric
        axs[i, 0].set_ylabel(metric,  fontsize=20)

        # Iterate over seats
        for j, seat in enumerate(seats):
            # Data for the boxplot
            data = [results[metric][model][seat] for model in models]

            # Create boxplot
            box = axs[i, j].boxplot(data, patch_artist=True)

            # Customize box colors
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
            for median in box['medians']:
                median.set(color='black')
            # Add title for each subplot
            axs[i, j].set_title(seat, fontsize=20)

            # Set x-axis ticks
            axs[i, j].set_xticklabels(models, fontsize=10)
            axs[i, j].tick_params(axis='y', labelsize=20)  # Adjust fontsize as needed    



    # Adjust the layout
    plt.tight_layout()
    
    # Show the plot
    plt.savefig(Mic_conf + '_'+ ContentType+'all.png')
    plt.show()






















if __name__ == '__main__':
    

    

    
    def wer(reference, hypothesis):
        # Split the reference and hypothesis into words
        ref_words = reference.split()
        hyp_words = hypothesis.split()
    
        # Create a matrix to store edit distances
        dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    
        # Initialize the first row and first column
        for i in range(len(ref_words) + 1):
            dp[i][0] = i
        for j in range(len(hyp_words) + 1):
            dp[0][j] = j
    
        # Compute edit distances using dynamic programming
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    
        # The last cell of the matrix contains the edit distance
        edit_distance = dp[len(ref_words)][len(hyp_words)]
    
        # Calculate WER as a percentage
        wer = (float(edit_distance) / len(ref_words)) * 100
    
        return wer
    
    def SDR(references, estimates):
        # compute SDR for one song
        delta = 1e-7  # avoid numerical errors
        num = np.sum(np.square(references), axis=(1, 2))
        den = np.sum(np.square(references - estimates), axis=(1, 2))
        num += delta
        den += delta
        return 10 * np.log10(num / den)




    def __TransformIntobatches__(Input, Output_size, input_size ,pad_left, pad_right):
        
        Input_batch = np.zeros(int(Input.shape[2]/Output_size) +1 , 4 , input_size)  
    
        for il in range(   int(Input.shape[2]/Output_size)):
            if il == 0:
                Input_batch [il,:,:] = torch.cat( (torch.zeros(4,pad_left) ,Input_batch[0,:,:input_size+pad_right]),1)
            #else:
                 #Input_batch [il,:,:] = Input_batch[0,:,il*Output_size-pad_left:il*Output_size+Output_size+pad_right])
            
        Input_batch [il+1,:,:] = torch.cat( (Input_batch[0,:,:(il+1)*Output_size-pad_left:Input.shape[2]], torch.zeros(4,input_size - (Input.shape[2]-(il+1)*Output_size+1)    )  ),1)  
        return Input_batch
    
    
    


    
    
    
    parser = argparse.ArgumentParser(description="A script that demonstrates argparse with main()")
    
    # Define arguments
    parser.add_argument('--arg1', default = 'set2' ,type=str, help='The first argument')
    # Parse the command-line arguments
    parser.add_argument('--arg2', default = 'Music' ,type=str, help='The first argument')


    parser.add_argument('--arg3', default = '0' ,type=str, help='The first argument')

    parser.add_argument('--arg4', default = 5 ,type=int, help='The first argument')

    args = parser.parse_args()

    # Call the main function with the parsed arguments
    print('hello first')
    main(args)
    
    
    












