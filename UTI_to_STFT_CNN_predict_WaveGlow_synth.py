'''
Written by Tamas Gabor Csapo <csapot@tmit.bme.hu>
First version Nov 9, 2016
Restructured Feb 4, 2018 - get data
Restructured Sep 19, 2018 - DNN training
Restructured Oct 13, 2018 - DNN training
Restructured Feb 18, 2020 - UTI to STFT
Restructured March 2, 2020 - improvements by Laszlo Toth <tothl@inf.u-szeged.hu>
 - swish, ultrasound scaling to [-1,1]
Documentation May 5, 2020 - more comments added

Keras implementation of the UTI-to-STFT model of
Tamas Gabor Csapo, Csaba Zainko, Laszlo Toth, Gabor Gosztolya, Alexandra Marko,
,,Ultrasound-based Articulatory-to-Acosutic Mapping with WaveGlow Speech Synthesis'', submitted to Interspeech 2020.
-> this script is for predicting the STFT (Mel-Spectrogram) parameters from UTI input,
-> and for synthesizing speech with the WaveGlow neural vocoder
'''

import matplotlib
matplotlib.use('agg')

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as io_wav
from detect_peaks import detect_peaks
import os
import os.path
import glob
import tgt
import pickle
import skimage

from scipy.signal import savgol_filter

# sample from Csaba
import WaveGlow_functions

from keras.models import model_from_json
from keras.layers import Activation

# do not use all GPU memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True 
set_session(tf.Session(config=config))


# defining the swish activation function
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Swish(swish)})


# read_ult reads in *.ult file from AAA
def read_ult(filename, NumVectors, PixPerVector):
    # read binary file
    ult_data = np.fromfile(filename, dtype='uint8')
    ult_data = np.reshape(ult_data, (-1, NumVectors, PixPerVector))
    return ult_data


# read_psync_and_correct_ult reads *_sync.wav and finds the rising edge of the pulses
# if there was a '3 pulses bug' during the recording,
# it removes the first three frames from the ultrasound data
def read_psync_and_correct_ult(filename, ult_data):
    (Fs, sync_data_orig) = io_wav.read(filename)
    sync_data = sync_data_orig.copy()

    # clip
    sync_threshold = np.max(sync_data) * 0.6
    for s in range(len(sync_data)):
        if sync_data[s] > sync_threshold:
            sync_data[s] = sync_threshold

    # find peeks
    peakind1 = detect_peaks(sync_data, mph=0.9*sync_threshold, mpd=10, threshold=0, edge='rising')
    
    '''
    # figure for debugging
    plt.figure(figsize=(18,4))
    plt.plot(sync_data)
    plt.plot(np.gradient(sync_data), 'r')
    for i in range(len(peakind1)):
        plt.plot(peakind1[i], sync_data[peakind1[i]], 'gx')
        # plt.plot(peakind2[i], sync_data[peakind2[i]], 'r*')
    plt.xlim(2000, 6000)
    plt.show()    
    '''
    
    # this is a know bug: there are three pulses, after which there is a 2-300 ms silence, 
    # and the pulses continue again
    if (np.abs( (peakind1[3] - peakind1[2]) - (peakind1[2] - peakind1[1]) ) / Fs) > 0.2:
        bug_log = 'first 3 pulses omitted from sync and ultrasound data: ' + \
            str(peakind1[0] / Fs) + 's, ' + str(peakind1[1] / Fs) + 's, ' + str(peakind1[2] / Fs) + 's'
        print(bug_log)
        
        peakind1 = peakind1[3:]
        ult_data = ult_data[3:]
    
    for i in range(1, len(peakind1) - 2):
        # if there is a significant difference between peak distances, raise error
        if np.abs( (peakind1[i + 2] - peakind1[i + 1]) - (peakind1[i + 1] - peakind1[i]) ) > 1:
            bug_log = 'pulse locations: ' + str(peakind1[i]) + ', ' + str(peakind1[i + 1]) + ', ' +  str(peakind1[i + 2])
            print(bug_log)
            bug_log = 'distances: ' + str(peakind1[i + 1] - peakind1[i]) + ', ' + str(peakind1[i + 2] - peakind1[i + 1])
            print(bug_log)
            
            raise ValueError('pulse sync data contains wrong pulses, check it manually!')
    return ([p for p in peakind1], ult_data)



def get_training_data(dir_file, filename_no_ext, NumVectors = 64, PixPerVector = 842):
    print('starting ' + dir_file + filename_no_ext)

    # read in raw ultrasound data
    ult_data = read_ult(dir_file + filename_no_ext + '.ult', NumVectors, PixPerVector)
    
    try:
        # read pulse sync data (and correct ult_data if necessary)
        (psync_data, ult_data) = read_psync_and_correct_ult(dir_file + filename_no_ext + '_sync.wav', ult_data)
    except ValueError as e:
        raise
    else:
        
        # works only with 22kHz sampled wav
        (Fs, speech_wav_data) = io_wav.read(dir_file + filename_no_ext + '_speech_volnorm.wav')
        assert Fs == 22050

        mgc_lsp_coeff = np.fromfile(dir_file + filename_no_ext + '_speech_volnorm_cut_ultrasound.mgclsp', dtype=np.float32).reshape(-1, order + 1)
        lf0 = np.fromfile(dir_file + filename_no_ext + '_speech_volnorm_cut_ultrasound.lf0', dtype=np.float32)

        (mgc_lsp_coeff_length, _) = mgc_lsp_coeff.shape
        (lf0_length, ) = lf0.shape
        assert mgc_lsp_coeff_length == lf0_length

        # cut from ultrasound the part where there are mgc/lf0 frames
        ult_data = ult_data[0 : mgc_lsp_coeff_length]

        # read phones from TextGrid
        tg = tgt.io.read_textgrid(dir_file + filename_no_ext + '_speech.TextGrid')
        tier = tg.get_tier_by_name(tg.get_tier_names()[0])

        tg_index = 0
        phone_text = []
        for i in range(len(psync_data)):
            # get times from pulse synchronization signal
            time = psync_data[i] / Fs

            # get current textgrid text
            if (tier[tg_index].end_time < time) and (tg_index < len(tier) - 1):
                tg_index = tg_index + 1
            phone_text += [tier[tg_index].text]

        # add last elements to phone list if necessary
        while len(phone_text) < lf0_length:
            phone_text += [phone_text[:-1]]

        print('finished ' + dir_file + filename_no_ext + ', altogether ' + str(lf0_length) + ' frames')

        return (ult_data, mgc_lsp_coeff, lf0, phone_text)


# Parameters of old vocoder
frameLength = 512 # 23 ms at 22050 Hz sampling
frameShift = 270 # 12 ms at 22050 Hz sampling, correspondong to 81.5 fps (ultrasound)
order = 24
n_mgc = order + 1

# STFT parameters
samplingFrequency = 22050
n_melspec = 80
hop_length_UTI = 270 # 12 ms
hop_length_WaveGlow = 256
stft = WaveGlow_functions.TacotronSTFT(filter_length=1024, hop_length=hop_length_UTI, \
    win_length=1024, n_mel_channels=n_melspec, sampling_rate=samplingFrequency, \
    mel_fmin=0, mel_fmax=8000)

# parameters of ultrasound images
framesPerSec = 81.67
# type = 'PPBA' # the 'PPBA' directory can be used for training data
type = 'EszakiSzel_1_normal' # the 'PPBA' directory can be used for test data
n_lines = 64
n_pixels = 842
n_pixels_reduced = 128


# TODO: modify this according to your data path
dir_base = "/shared/data_SSI2018/"
##### training data
# - 2 females: spkr048, spkr049
# - 5 males: spkr010, spkr102, spkr103, spkr104, spkr120
speakers = ['spkr048', 'spkr049', 'spkr102', 'spkr103']
# speakers = ['spkr102']

# download necessary WaveGlow sources
# !wget https://github.com/NVIDIA/waveglow/archive/a7168f398f8e2727291c2d46d0ef35ef30956de4.zip
# !unzip a7168f398f8e2727291c2d46d0ef35ef30956de4.zip -d /content/waveglow_orig_temp
# !mv waveglow_orig_temp/waveglow-a7168f398f8e2727291c2d46d0ef35ef30956de4 waveglow_o
# sys.path.append('/content/waveglow_o/')

# download pretrained WaveGlow model from:
# https://github.com/NVIDIA/waveglow/tree/a7168f398f8e2727291c2d46d0ef35ef30956de4
# https://drive.google.com/file/d/1cjKPHbtAMh_4HTHmuIGNkbOkPBD9qwhj/view?usp=sharing
#
# !wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1cjKPHbtAMh_4HTHmuIGNkbOkPBD9qwhj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1cjKPHbtAMh_4HTHmuIGNkbOkPBD9qwhj" -O waveglow_old.pt && rm -rf /tmp/cookies.txt
# -> result: 'waveglow_old.pt'

# load waveglow model
import soundfile as sf
import sys
sys.path.append('waveglow_o/')
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
waveglow_name = 'WaveGlow-EN'
waveglow_path = 'waveglow/waveglow_old.pt'
# waveglow_name = 'WaveGlow-HU'
# waveglow_path = 'waveglow/waveglow_635000'
print('loading WaveGlow model...')
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda()
print('loading WaveGlow model DONE')

for speaker in speakers:
    
    dir_synth = 'synthesized/' + speaker + '/'
    
    # collect all possible ult files
    ult_files_all = []
    dir_data = dir_base + speaker + "/" + type + "/"
    if os.path.isdir(dir_data):
        for file in sorted(os.listdir(dir_data)):
            if file.endswith('_speech_volnorm_cut_ultrasound.mgclsp'):
                ult_files_all += [dir_data + file[:-37]]
    
    csv_files = glob.glob('models/UTI_to_STFT_CNN-improved_' + speaker + '_*.csv')
    csv_files = sorted(csv_files)
    melspec_model_name = csv_files[-1][:-4]
    print(csv_files, melspec_model_name)
    
    # melspec network
    with open(melspec_model_name + '_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
    melspec_model = model_from_json(loaded_model_json)
    melspec_model.load_weights(melspec_model_name + '_weights_best.h5')
    # load scalers
    melspec_scalers = pickle.load(open(melspec_model_name + '_melspec_scalers.sav', 'rb'))
    
    for basefile in ult_files_all:
        try:
            (ult_data, mgc_lsp_coeff, lf0, phone_text) = get_training_data('', basefile)
            
            # load using mel_sample
            mel_data = WaveGlow_functions.get_mel(basefile + '_speech_volnorm_cut_ultrasound.wav', stft)
            mel_data = np.fliplr(np.rot90(mel_data.data.numpy(), axes=(1, 0)))
            
        except ValueError as e:
            print("wrong psync data, check manually!", e)
        else:
            ultmel_len = np.min((len(ult_data),len(mel_data)))
            ult_data = ult_data[0:ultmel_len]
            mel_data = mel_data[0:ultmel_len]
            
            print('predicting ', basefile, ult_data.shape, mel_data.shape)
            
            ult_test = np.empty((len(ult_data), n_lines, n_pixels_reduced))
            for i in range(ultmel_len):
                ult_test[i] = skimage.transform.resize(ult_data[i], (n_lines, n_pixels_reduced), preserve_range=True) / 255
            
            # input: already scaled to [0,1] range
            # rescale to [-1,1]
            ult_test -= 0.5
            ult_test *= 2
        
            # for CNN
            ult_test = np.reshape(ult_test, (-1, n_lines, n_pixels_reduced, 1))
            
            # input: already scaled to [0,1] range
            
            # predict with the trained DNN
            melspec_predicted = melspec_model.predict(ult_test)
            
            # we need to apply the inverse of the normalization
            for i in range(n_melspec):
                melspec_predicted[:, i] = melspec_scalers[i].inverse_transform(melspec_predicted[:, i].reshape(-1, 1)).ravel()
            
            # smooth signal
            melspec_predicted_smoothed = np.empty(melspec_predicted.shape)
            for i in range(n_melspec):
                melspec_predicted_smoothed[:, i] = savgol_filter(melspec_predicted[:, i], 5, 2) # window size (odd) / polynomial order
            
            # interpolate back to original frame shift: 256 samples (vs 270)
            print('interpolate / resize')
            interpolate_ratio = hop_length_UTI / hop_length_WaveGlow
            mel_data = skimage.transform.resize(mel_data, \
                (int(mel_data.shape[0] * interpolate_ratio), mel_data.shape[1]), preserve_range=True)
            melspec_predicted = skimage.transform.resize(melspec_predicted, \
                (int(melspec_predicted.shape[0] * interpolate_ratio), melspec_predicted.shape[1]), preserve_range=True)
            melspec_predicted_smoothed = skimage.transform.resize(melspec_predicted_smoothed, \
                (int(melspec_predicted_smoothed.shape[0] * interpolate_ratio), melspec_predicted_smoothed.shape[1]), preserve_range=True)
            
            # plot figures for debug
            plt.figure(figsize=(10,12))
            for i in range(10):
                plt.subplot(10,1,i+1)
                plt.plot(melspec_predicted[:, 8*i], 'r')
                plt.plot(melspec_predicted_smoothed[:, 8*i], 'b')
                plt.ylabel(str(8*i))
                plt.savefig(dir_synth + os.path.basename(basefile) + '_mel-predicted-smoothed-n' + str(n_melspec) + '.png')
            plt.close()
            
            plt.figure(figsize=(10,8))
            plt.subplot(311)
            plt.imshow(np.rot90(mel_data), cmap='gray')
            plt.title('original')
            plt.subplot(312)
            plt.imshow(np.rot90(melspec_predicted), cmap='gray')
            plt.title('predicted')
            plt.subplot(313)
            plt.imshow(np.rot90(melspec_predicted_smoothed), cmap='gray')
            plt.title('predicted & smoothed')
            plt.savefig(dir_synth + os.path.basename(basefile) + '_melspec-predicted-n' + str(n_melspec) + '.png')
            plt.close()
            
            # synthesis of original file // interpolated
            mel_data_for_synth = np.rot90(np.fliplr(mel_data), axes=(0, 1))
            mel_data_for_synth = torch.from_numpy(mel_data_for_synth.copy()).float().to(device)
            output_filename = dir_synth + os.path.basename(basefile) + '_' + waveglow_name + '_synth_orig.wav'
            print(waveglow_name + " orig... ", end="", flush=True)
            with torch.no_grad():
                audio = waveglow.infer(mel_data_for_synth.view([1,80,-1]).cuda(), sigma=0.666)
            sf.write(output_filename, audio[0].data.cpu().numpy(), 22050, subtype='PCM_16')
            print("end", flush=True)
            
            # synthesis of predicted file
            mel_data_for_synth = np.rot90(np.fliplr(melspec_predicted), axes=(0, 1))
            mel_data_for_synth = torch.from_numpy(mel_data_for_synth.copy()).float().to(device)
            output_filename = dir_synth + os.path.basename(basefile) + '_' + waveglow_name + '_synth_pred.wav'
            print(waveglow_name + " predicted... ", end="", flush=True)
            with torch.no_grad():
                audio = waveglow.infer(mel_data_for_synth.view([1,80,-1]).cuda(), sigma=0.666)
            sf.write(output_filename, audio[0].data.cpu().numpy(), 22050, subtype='PCM_16')
            print("end", flush=True)
            
            # synthesis of predicted & smoothed file
            mel_data_for_synth = np.rot90(np.fliplr(melspec_predicted_smoothed), axes=(0, 1))
            mel_data_for_synth = torch.from_numpy(mel_data_for_synth.copy()).float().to(device)
            output_filename = dir_synth + os.path.basename(basefile) + '_' + waveglow_name + '_synth_pred_smooth.wav'
            print(waveglow_name + " predicted & smoothed... ", end="", flush=True)
            with torch.no_grad():
                audio = waveglow.infer(mel_data_for_synth.view([1,80,-1]).cuda(), sigma=0.666)
            sf.write(output_filename, audio[0].data.cpu().numpy(), 22050, subtype='PCM_16')
            print("end", flush=True)
            
            if waveglow_name == 'WaveGlow-HU':
                # remove high freq noise
                # synthesis of predicted & smoothed file, with higher sigma
                output_filename = dir_synth + os.path.basename(basefile) + '_' + waveglow_name + '_synth_pred_smooth_sigma-0.9.wav'
                print(waveglow_name + " predicted & smoothed & sigma... ", end="", flush=True)
                with torch.no_grad():
                    audio = waveglow.infer(mel_data_for_synth.view([1,80,-1]).cuda(), sigma=0.9)
                sf.write(output_filename, audio[0].data.cpu().numpy(), 22050, subtype='PCM_16')
                print("end", flush=True)
            