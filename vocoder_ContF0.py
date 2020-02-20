#!/usr/bin/env python

'''
Skeleton for 'Continuous residual-based vocoder'

Written by
- Tamas Gabor CSAPO, csapot@tmit.bme.hu
- Mohammed Al-Radhi, malradhi@tmit.bme.hu

Nov 2016 - Jan 2017 - Feb 2018

requirement: SPTK 3.8 or above in PATH folder

References
* Mohammed Salah Al-Radhi, Tamás Gábor Csapó, Géza Németh ,,Time-domain envelope modulating the noise component of excitation in a continuous residual-based vocoder for statistical parametric speech synthesis'', Interspeech 2017, Stockholm, Sweden, pp. 434-438, 2017.
* Tamás Gábor Csapó, Géza Németh, Milos Cernak, Philip N. Garner, ,,Modeling Unvoiced Sounds In Statistical Parametric Speech Synthesis with a Continuous Vocoder'', EUSIPCO 2016 (24th European Signal Processing Conference), Budapest, Hungary, pp. 1338-1342, 2016.
* Bálint Pál Tóth, Tamás Gábor Csapó, ,,Continuous Fundamental Frequency Prediction with Deep Neural Networks'', EUSIPCO 2016 (24th European Signal Processing Conference), Budapest, Hungary, pp. 1348-1352, 2016.
* Tamás Gábor Csapó, Géza Németh, Milos Cernak, ,,Residual-based excitation with continuous F0 modeling in HMM-based speech synthesis'', SLSP 2015 (3rd International Conference on Statistical Language and Speech Processing), Budapest, Hungary, Lecture Notes in Artificial Intelligence 9449, pp. 27-38, 2015.
'''

'''

################################## Usage example ###################################################################

residual_codebook_filename = 'resid_cdbk_awb_0080_pca.bin'

# read in residual PCA codebook and filtered version
resid_codebook_pca = read_residual_codebook(residual_codebook_filename)

wav_path = ''
basefilename = '20180122_spkr102_012_speech_volnorm_cut_ultrasound'
basefilename_out = basefilename + '_vocoded_ContF0'

log_f0cont = np.fromfile(wav_path + basefilename + '.lf0cont', dtype=np.float32)
log_mvf = np.fromfile(wav_path + basefilename + '.mvf', dtype=np.float32)
order = 24
mgc_lsp_coeff = np.fromfile(wav_path + basefilename + '.mgclsp', dtype=np.float32).reshape(-1, order + 1)

mgc_decoder_residual(mgc_lsp_coeff, log_f0cont, log_mvf, basefilename_out, resid_codebook_pca,
    Fs = 22050, frlen = 512, frshft = 270, order = 24, alpha = 0.42, stage = 3,
    hpf_order = 11, noise_scaling = 0.04)
'''



import matplotlib.pyplot as plt
import numpy as np
import os
import pysptk
import scipy
import scipy.signal
import scipy.io.wavfile as io_wav
import struct
from subprocess import call, run




####################################  lowpass / highpass filters  ###############################

def cheby1_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # rp: The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number.
    b, a = scipy.signal.cheby1(order, 0.1, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data_l, cutoff, fs, order=5):
    b, a = cheby1_lowpass(cutoff, fs, order=order)
    y = scipy.signal.lfilter(b, a, data_l)
    return y

def cheby1_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # rp: The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number.
    c, d = scipy.signal.cheby1(order, 0.1, normal_cutoff, btype='high', analog=False)
    return c, d

def highpass_filter(data_h, cutoff, fs, order=5):
    c, d = cheby1_highpass(cutoff, fs, order=order)
    z = scipy.signal.lfilter(c, d, data_h)
    return z

###################################################################################################





##################################  Read Residual Codebook ########################################

def read_residual_codebook(codebook_filename):
    file_in = open(codebook_filename, 'rb')
    f = file_in.read(4) # 4 byte int
    cdbk_size, = struct.unpack('i', f)
    f = file_in.read(4) # 4 byte int
    resid_pca_length, = struct.unpack('i', f)
    resid_pca = np.zeros((cdbk_size, resid_pca_length))
    for i in range(cdbk_size):
        if i > 0:
            f = file_in.read(4) # 4 byte int
            resid_pca_length, = struct.unpack('i', f)
        f = file_in.read(8 * resid_pca_length) # 8 byte double * resid_pca_length
        resid_pca_current = struct.unpack('<%dd' % resid_pca_length, f)
        resid_pca[i] = resid_pca_current
    
    return resid_pca

###################################################################################################



    
###################### Synthesis using Continuous F0 + MVF + MGC + Residual #######################

def mgc_decoder_residual(mgc_lsp_coeff, log_f0cont, log_mvf, basefilename_out,
    resid_codebook_pca, Fs_codebook = 16000,
    Fs = 22050, frlen = 512, frshft = 200, order = 24, alpha = 0.42, stage = 3,
    hpf_order = 11, noise_scaling = 0.04):
    
    pitch = np.float64(np.exp(log_f0cont))
    mvf = np.exp(log_mvf)
    
    # create voiced source excitation using SPTK
    source_voiced = pysptk.excite(Fs / pitch, frshft)
    
    # create unvoiced source excitation using SPTK
    pitch_unvoiced = np.zeros(len(pitch))
    source_unvoiced = pysptk.excite(pitch_unvoiced, frshft)
    
    source = np.zeros(source_voiced.shape)
    
    # generate excitation frame by frame pitch synchronously
    
    # voiced component
    for i in range(len(source)):
        if source_voiced[i] > 2: # location of impulse in original impulse excitation
            mvf_index = int(i / frshft)
            mvf_curr = mvf[mvf_index]
            
            if mvf_curr > Fs_codebook / 2:
                mvf_curr = Fs_codebook / 2
            
            # voiced component from residual codebook
            voiced_frame_lpf = resid_codebook_pca[int((Fs_codebook / 2 - mvf_curr) / 100)]
            
            # put voiced and unvoiced component to pitch synchronous location
            j_start = np.max((round(len(voiced_frame_lpf) / 2) - i, 0))
            j_end   = np.min((len(voiced_frame_lpf), len(source) - (i - round(len(voiced_frame_lpf) / 2))))
            for j in range(j_start, j_end):
                source[i - round(len(voiced_frame_lpf) / 2) + j] += voiced_frame_lpf[j]
    
    # unvoiced component
    for i in range(len(mvf)):
        unvoiced_frame = source_unvoiced[i * frshft : (i+2) * frshft].copy()
        mvf_curr = mvf[i]
        unvoiced_frame_hpf = highpass_filter(unvoiced_frame, mvf_curr * 1.2, Fs, hpf_order)
        unvoiced_frame_hpf *= np.hanning(len(unvoiced_frame_hpf))
        
        source[i * frshft : (i+2) * frshft] += unvoiced_frame_hpf * noise_scaling
        
    
    # scale for SPTK
    scaled_source = np.float32(source / np.max(np.abs(source)) )
    io_wav.write(basefilename_out + '_source_float32.wav', Fs, scaled_source)
    
    # write files for SPTK
    mgc_lsp_coeff.astype('float32').tofile(basefilename_out + '.mgclsp')
    
    # MGC-LSPs -> MGC coefficients
    command = 'lspcheck -m ' + str(order) + ' -s ' + str(Fs / 1000) + ' -c -r 0.1 -g -G 1.0E-10 ' + basefilename_out + '.mgclsp' + ' | ' + \
              'lsp2lpc -m '  + str(order) + ' -s ' + str(Fs / 1000) + ' | ' + \
              'mgc2mgc -m '  + str(order) + ' -a ' + str(alpha) + ' -c ' + str(stage) + ' -n -u ' + \
                      '-M '  + str(order) + ' -A ' + str(alpha) + ' -C ' + str(stage) + ' > ' + basefilename_out + '.mgc'
    run(command, shell=True)
    
    command = 'sox ' + basefilename_out + '_source_float32.wav' + ' -t raw -r ' + str(Fs) + ' - ' + ' | ' + \
              'mglsadf -P 5 -m ' + str(order) + ' -p ' + str(frshft) + \
              ' -a ' + str(alpha) + ' -c ' + str(stage) + ' ' + basefilename_out + '.mgc' + ' | ' + \
              'x2x +fs -o | sox -c 1 -b 16 -e signed-integer -t raw -r ' + str(Fs) + ' - -t wav -r ' + str(Fs) + ' ' + basefilename_out + '_0.wav'
    # print(command)
    run(command, shell=True)
    
    # normalize gain
    command = "sox --norm=-3 " + basefilename_out + '_0.wav' + ' ' + \
        basefilename_out + '.wav'
    # print(command)
    run(command, shell=True)
    
    # remove temp files
    os.remove(basefilename_out + '_0.wav')
    os.remove(basefilename_out + '.mgc')
    os.remove(basefilename_out + '.mgclsp')
    os.remove(basefilename_out + '_source_float32.wav')
    
    # read file for output
    (Fs_out, x_synthesized) = io_wav.read(basefilename_out + '.wav')
    
    return x_synthesized

####################################################################################################################











