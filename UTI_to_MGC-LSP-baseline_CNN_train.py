'''
Written by Tamas Gabor Csapo <csapot@tmit.bme.hu>
First version Nov 9, 2016
Restructured Feb 4, 2018 - get data
Restructured Sep 19, 2018 - DNN training
Restructured Oct 13, 2018 - DNN training
Restructured Feb 25, 2019 - CNN training for ContVoc parameters
Restructured Feb 20, 2020 - comparison with STST

Keras implementation of the UTI-to-ContF0 model of
Tamás Gábor Csapó, Mohammed Salah Al-Radhi, Géza Németh, Gábor Gosztolya, Tamás Grósz, László Tóth, Alexandra Markó, ,,Ultrasound-based Silent Speech Interface Built on a Continuous Vocoder'', Interspeech 2019, pp. 894-898.   http://arxiv.org/abs/1906.09885
-> this script is for training the MGC-LSP parameters
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as io_wav
from detect_peaks import detect_peaks
import os
import os.path
import gc
import re
import tgt
import csv
import datetime
import scipy
import pickle
import random
random.seed(17)
import skimage


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler



# do not use all GPU memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True 
set_session(tf.Session(config=config))




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


# Parameters of Continuous vocoder
frameLength = 512 # 23 ms at 22050 Hz sampling
frameShift = 270 # 12 ms at 22050 Hz sampling, correspondong to 81.5 fps (ultrasound)
order = 24
n_mgc = order + 1


# parameters of ultrasound images
framesPerSec = 81.67
type = 'PPBA' # the 'PPBA' directory can be used for training data
# type = 'EszakiSzel_1_normal' # the 'PPBA' directory can be used for training data
n_lines = 64
n_pixels = 842
n_pixels_reduced = 128


# TODO: modify this according to your data path
dir_base = "/shared/data_SSI2018/"
##### training data
# - 2 females: spkr048, spkr049
# - 5 males: spkr010, spkr102, spkr103, spkr104, spkr120
speakers = ['spkr048', 'spkr049', 'spkr102', 'spkr103']

for speaker in speakers:
    
    # collect all possible ult files
    ult_files_all = []
    dir_data = dir_base + speaker + "/" + type + "/"
    if os.path.isdir(dir_data):
        for file in sorted(os.listdir(dir_data)):
            if file.endswith('_speech_volnorm_cut_ultrasound.mgclsp'):
                ult_files_all += [dir_data + file[:-37]]
    
    # randomize the order of files
    random.shuffle(ult_files_all)
    
    # temp: only first 10 sentence
    # ult_files_all = ult_files_all[0:10]
    
    ult_files = dict()
    ult = dict()
    mgc = dict()
    ultmgc_size = dict()
    
    # train: first 90% of sentences
    ult_files['train'] = ult_files_all[0:int(0.9*len(ult_files_all))]
    # valid: last 10% of sentences
    ult_files['valid'] = ult_files_all[int(0.9*len(ult_files_all)):]
    
    for train_valid in ['train', 'valid']:
        n_max_ultrasound_frames = len(ult_files[train_valid]) * 500
        ult[train_valid] = np.empty((n_max_ultrasound_frames, n_lines, n_pixels_reduced))
        mgc[train_valid] = np.empty((n_max_ultrasound_frames, n_mgc))
        ultmgc_size[train_valid] = 0
        
        # load all training/validation data
        for basefile in ult_files[train_valid]:
            try:
                (ult_data, mgc_lsp_coeff, lf0, phone_text) = get_training_data('', basefile)
                
            except ValueError as e:
                print("wrong psync data, check manually!", e)
            else:
                ultmgc_len = np.min((len(ult_data),len(mgc_lsp_coeff)))
                ult_data = ult_data[0:ultmgc_len]
                mgc_lsp_coeff = mgc_lsp_coeff[0:ultmgc_len]
                
                print(basefile, ult_data.shape, mgc_lsp_coeff.shape)
                
                if ultmgc_size[train_valid] + ultmgc_len > n_max_ultrasound_frames:
                    print('data too large', n_max_ultrasound_frames, ultmgc_size[train_valid], ultmgc_len)
                    raise
                
                for i in range(ultmgc_len):
                    ult[train_valid][ultmgc_size[train_valid] + i] = skimage.transform.resize(ult_data[i], (n_lines, n_pixels_reduced), preserve_range=True) / 255
                
                mgc[train_valid][ultmgc_size[train_valid] : ultmgc_size[train_valid] + ultmgc_len] = mgc_lsp_coeff
                ultmgc_size[train_valid] += ultmgc_len
                
                print('n_frames_all: ', ultmgc_size[train_valid])


        ult[train_valid] = ult[train_valid][0 : ultmgc_size[train_valid]]
        mgc[train_valid] = mgc[train_valid][0 : ultmgc_size[train_valid]]

        # reshape ult for CNN
        ult[train_valid] = np.reshape(ult[train_valid], (-1, n_lines, n_pixels_reduced, 1))
        

    # input: already scaled to [0,1] range

    # target: normalization to zero mean, unit variance
    # feature by feature
    mgc_scalers = []
    for i in range(n_mgc):
        mgc_scaler = StandardScaler(with_mean=True, with_std=True)
        mgc_scalers.append(mgc_scaler)
        mgc['train'][:, i] = mgc_scalers[i].fit_transform(mgc['train'][:, i].reshape(-1, 1)).ravel()
        mgc['valid'][:, i] = mgc_scalers[i].transform(mgc['valid'][:, i].reshape(-1, 1)).ravel()

    ### single training without cross-validation
    # convolutional model, same as Interspeech2019
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', input_shape=(n_lines, n_pixels_reduced, 1), padding='same'))
    model.add(MaxPooling2D((3,3), padding='same'))
    model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((3,3), padding='same'))
    model.add(Flatten())
    model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
    model.add(Dense(n_mgc, kernel_initializer='normal', activation='linear'))

    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')

    print(model.summary())

    # early stopping to avoid over-training
    # csv logger
    current_date = '{date:%Y-%m-%d_%H-%M-%S}'.format( date=datetime.datetime.now() )
    print(current_date)
    model_name = 'models/UTI_to_MGC-LSP_CNN_' + speaker + '_' + current_date
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0), \
                 CSVLogger(model_name + '.csv', append=True, separator=';'), \
                 ModelCheckpoint(model_name + '_weights_best.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')]

    # save model
    model_json = model.to_json()
    with open(model_name + '_model.json', "w") as json_file:
        json_file.write(model_json)

    # serialize scalers to pickle
    pickle.dump(mgc_scalers, open(model_name + '_mgc_scalers.sav', 'wb'))

    # Run training
    history = model.fit(ult['train'], mgc['train'],
                            epochs = 100, batch_size = 128, shuffle = True, verbose = 1,
                            validation_data=(ult['valid'], mgc['valid']),
                            callbacks=callbacks)
    # here the training of the DNN is finished



