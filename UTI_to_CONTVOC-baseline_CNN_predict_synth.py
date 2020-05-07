'''
Written by Tamas Gabor Csapo <csapot@tmit.bme.hu>
First version Nov 9, 2016
Restructured Feb 4, 2018 - get data
Restructured Sep 19, 2018 - DNN training
Restructured Oct 13, 2018 - DNN training
Restructured Feb 25, 2019 - CNN training for ContVoc parameters
Restructured Feb 20, 2020 - comparison with STST
Restructured March 2, 2020 - improvements by Laszlo Toth <tothl@inf.u-szeged.hu>
 - swish, ultrasound scaling to [-1,1]
Documentation May 5, 2020 - more comments added

Keras implementation of the UTI-to-ContF0 model of
Tamás Gábor Csapó, Mohammed Salah Al-Radhi, Géza Németh, Gábor Gosztolya, Tamás Grósz, László Tóth, Alexandra Markó, ,,Ultrasound-based Silent Speech Interface Built on a Continuous Vocoder'', Interspeech 2019, pp. 894-898.   http://arxiv.org/abs/1906.09885
-> this script is for predicting the MGC-LSP & ContF0 & MVF parameters from UTI input,
-> and for synthesizing speech with the Continuous vocoder
'''

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

# additional requirement: SPTK 3.8 or above in PATH
import vocoder_ContF0


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


def get_contf0_mvf(dir_file, filename_no_ext):
    # works only with 22kHz sampled wav
    
    contf0 = np.fromfile(dir_file + filename_no_ext + '_speech_volnorm_cut_ultrasound.lf0cont', dtype=np.float32)
    mvf = np.fromfile(dir_file + filename_no_ext + '_speech_volnorm_cut_ultrasound.mvf', dtype=np.float32)
    
    (contf0_length) = contf0.shape
    (mvf_length) = mvf.shape
    assert contf0_length == mvf_length
    
    return (contf0, mvf)

# Parameters of Continuous vocoder
frameLength = 512 # 23 ms at 22050 Hz sampling
frameShift = 270 # 12 ms at 22050 Hz sampling, correspondong to 81.5 fps (ultrasound)
order = 24
n_contf0_mvf = 2
n_mgc = order + 1


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


for speaker in speakers:
    dir_synth = 'synthesized/' + speaker + '/'
    
    # residual excitation for ContVoc
    if speaker in ['spkr048', 'spkr049']: # females
        residual_codebook_filename = 'resid_cdbk_slt_0080_pca.bin'
    elif speaker in ['spkr102', 'spkr103']: # males
        residual_codebook_filename = 'resid_cdbk_awb_0080_pca.bin'
    resid_codebook_pca = vocoder_ContF0.read_residual_codebook(residual_codebook_filename)

    
    # collect all possible ult files
    ult_files_all = []
    dir_data = dir_base + speaker + "/" + type + "/"
    if os.path.isdir(dir_data):
        for file in sorted(os.listdir(dir_data)):
            if file.endswith('_speech_volnorm_cut_ultrasound.mgclsp'):
                ult_files_all += [dir_data + file[:-37]]
    
    # mgc network
    csv_files = sorted(glob.glob('models/UTI_to_MGC-LSP_CNN-improved_' + speaker + '_*.csv'))
    mgc_model_name = csv_files[-1][:-4]
    print(csv_files, mgc_model_name)
    with open(mgc_model_name + '_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
    mgc_model = model_from_json(loaded_model_json)
    mgc_model.load_weights(mgc_model_name + '_weights_best.h5')
    # load scalers
    mgc_scalers = pickle.load(open(mgc_model_name + '_mgc_scalers.sav', 'rb'))
    
    # contf0 & mvf network
    csv_files = sorted(glob.glob('models/UTI_to_CONTF0-MVF_CNN-improved_' + speaker + '_*.csv'))
    contf0_mvf_model_name = csv_files[-1][:-4]
    print(csv_files, contf0_mvf_model_name)
    with open(contf0_mvf_model_name + '_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
    contf0_mvf_model = model_from_json(loaded_model_json)
    contf0_mvf_model.load_weights(contf0_mvf_model_name + '_weights_best.h5')
    # load scalers
    contf0_mvf_scalers = pickle.load(open(contf0_mvf_model_name + '_contf0_mvf_scalers.sav', 'rb'))
    
    for basefile in ult_files_all:
        try:
            (ult_data, mgc_lsp_coeff, lf0, phone_text) = get_training_data('', basefile)
            
            # load cont vocoder features
            (contf0_original, mvf_original) = get_contf0_mvf('', basefile)
        except ValueError as e:
            print("wrong psync data, check manually!")
        else:
            print('testing on: ' + basefile)
            
            min_len = np.min((len(ult_data),len(mgc_lsp_coeff), len(contf0_original), len(mvf_original)))
            ult_data = ult_data[0:min_len]
            mgc_data = mgc_lsp_coeff[0:min_len]
            contf0_data = contf0_original[0:min_len]
            mvf_data = mvf_original[0:min_len]
            
            ult_test = np.empty((len(ult_data), n_lines, n_pixels_reduced))
            for i in range(min_len):
                ult_test[i] = skimage.transform.resize(ult_data[i], (n_lines, n_pixels_reduced), preserve_range=True) / 255
            
            # for CNN
            ult_test = np.reshape(ult_test, (-1, n_lines, n_pixels_reduced, 1))
            
            # input: already scaled to [0,1] range
            # rescale to [-1,1]
            ult_test -= 0.5
            ult_test *= 2
            
            # predict with the trained CNN
            mgc_predicted = mgc_model.predict(ult_test)
            
            # apply the inverse of the normalization
            for i in range(n_mgc):
                mgc_predicted[:, i] = mgc_scalers[i].inverse_transform(mgc_predicted[:, i].reshape(-1, 1)).ravel()
            
            # predict with the trained CNN
            contf0_mvf_predicted = contf0_mvf_model.predict(ult_test)
            
            # apply the inverse of the normalization
            for i in range(n_contf0_mvf):
                contf0_mvf_predicted[:, i] = contf0_mvf_scalers[i].inverse_transform(contf0_mvf_predicted[:, i].reshape(-1, 1)).ravel()
            
            # get back contf0 and mvf
            contf0_predicted = contf0_mvf_predicted[:, 0]
            mvf_predicted = contf0_mvf_predicted[:, 1]
            
            # synthesize using continuous vocoder // original
            vocoder_ContF0.mgc_decoder_residual(mgc_lsp_coeff, contf0_original, mvf_original, dir_synth + os.path.basename(basefile) + '_ContVoc_synth_orig', resid_codebook_pca, \
                            Fs = 22050, frlen = 512, frshft = 270, order = 24, alpha = 0.42, stage = 3, \
                            hpf_order = 11, noise_scaling = 0.04)
            
            # synthesize using continuous vocoder // predicted
            vocoder_ContF0.mgc_decoder_residual(mgc_predicted, contf0_predicted, mvf_predicted, dir_synth + os.path.basename(basefile) + '_ContVoc_synth_pred', resid_codebook_pca, \
                            Fs = 22050, frlen = 512, frshft = 270, order = 24, alpha = 0.42, stage = 3, \
                            hpf_order = 11, noise_scaling = 0.04)
            