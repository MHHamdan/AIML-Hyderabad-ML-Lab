# -*- coding: utf-8 -*-


#import pyaudio
#import wave

import pyaudio
import wave
import keyboard as kb
import librosa
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import torch
import os
from torch.autograd import Variable

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_network():

    net = torch.nn.Sequential()

    saved_net = torch.load(BASE_DIR + "/Hackathon-setup/net_speech_89.pt").cpu()

    for index, module in enumerate(saved_net):
        net.add_module("layer"+str(index),module)
        if (index+1)%17 == 0 :
            break
    return net

def wait_for_key() :
	while True:
	    try:
	        if kb.is_pressed('s'):
 	             return
	        else:
	            pass
	    except:
	        continue 


#Use this function to return the deep learning audio features by providing the audio file path
#filepath for path of the audio file
#sr(samplingrate = 8000) for all the recordings and newly recorded audio files use the same sampling rate
#n_mfcc =30
#n_mels = 128
#frames = 15
def get_features(filepath, sr=8000, n_mfcc=30, n_mels=128, frames = 15):
    
    
    y, sr = librosa.load(filepath, sr=sr)
    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_S = librosa.power_to_db(S,ref=np.max)
    features = librosa.feature.mfcc(S=log_S, n_mfcc=n_mfcc)
    if features.shape[1] < frames :
        features = np.hstack((features, np.zeros((n_mfcc, frames - features.shape[1]))))
    elif features.shape[1] > frames:
        features = features[:, :frames]
    # Find 1st order delta_mfcc
    delta1_mfcc = librosa.feature.delta(features, order=1)

    # Find 2nd order delta_mfcc
    delta2_mfcc = librosa.feature.delta(features, order=2)
    features = np.hstack((delta1_mfcc.flatten(), delta2_mfcc.flatten()))
    features = features.flatten()[np.newaxis, :]
    features = Variable(torch.from_numpy(features)).float()
    deep_net = get_network()
    deep_features = deep_net(features)
    #print(features.shape)
    #print(audio_file)
    #features.flatten()[np.newaxis, :]
    return deep_features.data.numpy().flatten()


#Function to record the voice sample, total recording time is 1 sec
#Username is the identifier for the person recording the voice
#j is the label for the sample For Example : if you recording the sample for "one" label is 1, for "yes" it is 11 etc.
#v is the unique identifier for each sample recorded by a person 
#Example username is r1 , j is 1 (label), v is 10 (10th sample recorded by that person) audio file will be saved with the name 1_r1_10.wav
#returns the filepath after recording
def record_voice(Username, j, v, dir ):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 8000
    CHUNK = 1024
    RECORD_SECONDS = 1
    WAVE_OUTPUT_FILENAME = "file.wav"
    audio = pyaudio.PyAudio()
 
# start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
    print("recording...")
    frames = []
 
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")
 
 
# stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    WAVE_OUTPUT_FILENAME = str(j)+"_"+Username+"_"+str(v)+".wav"
    #print(WAVE_OUTPUT_FILENAME)
    waveFile = wave.open(dir+WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    return dir+WAVE_OUTPUT_FILENAME


##Given audio file path, this plays that wav file
def play_audio(path) :

	CHUNK = 1024

	wf = wave.open(path, 'rb')

	# instantiate PyAudio (1)
	p = pyaudio.PyAudio()

	# open stream (2)
	stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
	                channels=wf.getnchannels(),
	                rate=wf.getframerate(),
	                output=True)

	# read data
	data = wf.readframes(CHUNK)

	# play stream (3)
	while len(data) > 0:
	    stream.write(data)
	    data = wf.readframes(CHUNK)

	# stop stream (4)
	stream.stop_stream()
	stream.close()

	# close PyAudio (5)
	p.terminate()


#play_audio("/Users/raghavamodhugu/Documents/GitHub/Voice_commands_based_ordering_system/speech_data/0_b5_21.wav")
    
    
def plotchart(objects, confidence):
    y_pos = np.arange(len(objects))
     
    plt.bar(y_pos, confidence, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('confidence')
    plt.title('latest confidence chart wise')
     
    plt.show()