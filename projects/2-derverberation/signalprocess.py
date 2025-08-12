'''
breif:  functions for audio signal processsing
author: lianwuchen@tencent.com
date:   2018-03-01
'''

import string
import threading
import numpy as np
import scipy
from numpy.fft import rfft, irfft
from scipy import signal
from scipy.io.wavfile import write as wav_write
from scipy.io.wavfile import read as wav_read

def getWeightsForMLPBF(FS, NFFT, micPos, nAng):
    nBin        = int(NFFT/2)+1
    nMic        = micPos.shape[0]
    weights_mlp = np.zeros([nMic*nBin*2, nAng*nBin*2]) 
    weights     = getDelaySumWeightsMultiAng(FS, NFFT, micPos, nAng)
    for idx_a in range(nAng):
        for idx_b in range(nBin):
            for idx_m in range(nMic):
                w_real = weights[idx_m, idx_b, idx_a].real
                w_imag = weights[idx_m, idx_b, idx_a].imag
                weights_mlp[idx_m*idx_b, idx_a * idx_b]      = w_real
                weights_mlp[idx_m*idx_b, idx_a * idx_b+1]    = w_imag
                weights_mlp[idx_m*idx_b+1, idx_a * idx_b]    = -1*w_imag
                weights_mlp[idx_m*idx_b+1, idx_a * idx_b+1]  = w_real
    return weights_mlp

def getDelaySumWeightsMultiAng(FS, NFFT, micPos, nAng):
    r       = 5
    nMic    = micPos.shape[0]
    objPos  = np.zeros([nAng,3]) 
    weights = np.zeros([nMic, int(NFFT/2)+1, nAng],dtype=complex)  
    for i in range(nAng):
        objPos[i,0] = r * np.sin(i * np.pi/nAng)
        objPos[i,1] = r * np.cos(i * np.pi/nAng)
        weights[:,:,i] = getDelaySumWeights(FS, NFFT, micPos, objPos[i,:])
        #print 'ang' + str(i) 
        #print weights[1,1:10,i]
    return weights
    
def getDelaySumWeights(FS, NFFT, micPos, objPos):
    soundSpeed  = 342
    nMic        = micPos.shape[0]
    delays      = np.zeros([nMic, 1])
    weight      = np.zeros([nMic, int(NFFT/2)+1], dtype=complex)
    for i in range(nMic):
        delays[i]    = np.sqrt(np.sum(np.square(micPos[i,:] - objPos)))/soundSpeed
    for idx_f in range(NFFT/2+1):
        for idx_m in range(nMic):
            a = np.exp(1j * delays[idx_m] * 2 * np.pi * idx_f * FS /NFFT) / nMic
            weight[idx_m,idx_f] = a[0]
    return weight

def hz2mel(hz):
    return 2595*np.log10(1+hz/700.0)

def mel2hz(mel):
    return 700*(10**(mel/2595.0)-1)

def get_filter_banks_MEL(filters_num=40,NFFT=256,samplerate=16000,low_freq=0,high_freq=8000):
    low_mel=hz2mel(low_freq)
    high_mel=hz2mel(high_freq)
    mel_points=np.linspace(low_mel,high_mel,filters_num+2)
    hz_points=mel2hz(mel_points)
    bin=np.floor((NFFT+1)*hz_points/samplerate)
    fbank=np.zeros([filters_num,NFFT/2+1],'float32')
    for j in xrange(0,filters_num):
        for i in xrange(int(bin[j]),int(bin[j+1])):
            fbank[j,i]=(i-bin[j])/(bin[j+1]-bin[j])
        for i in xrange(int(bin[j+1]),int(bin[j+2])):
            fbank[j,i]=(bin[j+2]-i)/(bin[j+2]-bin[j+1])
    return fbank

def get_filter_banks_ERB(NFFT=256,samplerate=16000):
    erbRate = 1.0
    [K, cf] = getERBPartitions(samplerate, NFFT/2+1, erbRate)
    fbank_ERB   = np.zeros([len(cf),NFFT/2+1],'float32')
    idx_s       = 0
    for j in xrange(0,len(K)):
        idx_e = int(idx_s + K[j])
        fbank_ERB[j, idx_s:idx_e] = 1
        idx_s = idx_e
    return fbank_ERB

def getInvTiQ(NFFT=256,samplerate=16000):
    dBSPL1  = 80.0
    f       = np.asarray(range(0, NFFT/2+1, 1), 'float32')
    f       = f * samplerate / NFFT
    f[0]    = 1.0
    invTiQ  = 1/ (getTiQ(f))
    invTiQ  = invTiQ * np.power(10, (dBSPL1 / 20)) / NFFT
    return invTiQ

def getTiQ(f):
    TiQdB   = 3.64 * np.power(f / 1000, -0.8) - 6.5 * np.exp(-0.6 * np.power(((f / 1000) - 3.3), 2)) + np.power(10.0, -3) * np.power((f / 1000), 4)
    TiQdB   = np.minimum(TiQdB, 70)
    TiQ     = np.power(10, (TiQdB / 20))
    return TiQ

def getERBPartitions( sampleRate, K, erbRate=1.0 ):
    ERB_L   = 24.7                # minimum ERB bandwidth (24.7 Hz)
    ERB_Q   = 9.265               # Q factor of auditory filters
    c1      = 1/(2*float(K))
    c2      = 1/float(sampleRate)
    k1      = 1
    b       = 0
    nk      = []
    cf      = []
    while k1 <= K:
        b       = b + 1
        k0      = k1
        bw      = ERB_L + (k0 * sampleRate * c1) / ERB_Q
        k1      = k0 + max(1, round( 2 * K * bw * erbRate * c2 ))
        k1      = min(k1,K+1)
        nk.append(k1 - k0)
        cf.append(0.5 * (k0+k1-3) * sampleRate * c1)
    return nk,cf

def _samples_to_stft_frames(samples, size, shift):
    """
    Calculates STFT frames from samples in time domain.
    :param samples: Number of samples in time domain.
    :param size: FFT size.
    :param shift: Hop in samples.
    :return: Number of STFT frames.
    """
    return np.ceil((samples - size + shift) / shift).astype(np.int)

def _stft_frames_to_samples(frames, size, shift):
    """
    Calculates samples in time domain from STFT frames
    :param frames: Number of STFT frames.
    :param size: FFT size.
    :param shift: Hop in samples.
    :return: Number of samples in time domain.
    """
    return frames * shift + size - shift


def _biorthogonal_window_loopy(analysis_window, shift):
    """
    This version of the synthesis calculation is as close as possible to the
    Matlab impelementation in terms of variable names.

    The results are equal.

    The implementation follows equation A.92 in
    Krueger, A. Modellbasierte Merkmalsverbesserung zur robusten automatischen
    Spracherkennung in Gegenwart von Nachhall und Hintergrundstoerungen
    Paderborn, Universitaet Paderborn, Diss., 2011, 2011
    """
    fft_size = len(analysis_window)
    assert np.mod(fft_size, shift) == 0
    number_of_shifts = len(analysis_window) // shift

    sum_of_squares = np.zeros(shift)
    for synthesis_index in range(0, shift):
        for sample_index in range(0, number_of_shifts + 1):
            analysis_index = synthesis_index + sample_index * shift

            if analysis_index + 1 < fft_size:
                sum_of_squares[synthesis_index] \
                    += analysis_window[analysis_index] ** 2

    sum_of_squares = np.kron(np.ones(number_of_shifts), sum_of_squares)
    synthesis_window = analysis_window / sum_of_squares / fft_size
    return synthesis_window

def audioread(wavfile):
    rate, data = wav_read(wavfile)
    if data.dtype == 'i2':
        data = data/(2.0 ** 15)
    if data.dtype == 'i4':
        data = data/(2.0 ** 31)
    data.astype('float32')
    return rate, data

def stft(time_signal, time_dim=None, nFFT=512, size=256, shift=128,
         window=signal.hamming, fading=False, window_length=None):
    """
    Calculates the short time Fourier transformation of a multi channel multi
    speaker time signal. It is able to add additional zeros for fade-in and
    fade out and should yield an STFT signal which allows perfect
    reconstruction.

    :param time_signal: multi channel time signal.
    :param time_dim: Scalar dim of time.
        Default: None means the biggest dimension
    :param size: Scalar FFT-size.
    :param shift: Scalar FFT-shift. Typically shift is a fraction of size.
    :param window: Window function handle.
    :param fading: Pads the signal with zeros for better reconstruction.
    :param window_length: Sometimes one desires to use a shorter window than
        the fft size. In that case, the window is padded with zeros.
        The default is to use the fft-size as a window size.
    :return: Single channel complex STFT signal
        with dimensions frames times size/2+1.
    """
    
    if time_dim is None:
        time_dim = np.argmax(time_signal.shape)
    # Pad with zeros to have enough samples for the window function to fade.
    if fading:
        pad = [(0, 0)] * time_signal.ndim
        pad[time_dim] = [size - shift, size - shift]
        time_signal = np.pad(time_signal, pad, mode='constant')

    # Pad with trailing zeros, to have an integral number of frames.
    frames = _samples_to_stft_frames(time_signal.shape[time_dim], size, shift)
    samples = _stft_frames_to_samples(frames, size, shift)
    pad = [(0, 0)] * time_signal.ndim
    pad[time_dim] = [0, samples - time_signal.shape[time_dim]]
    #time_signal = np.pad(time_signal, pad, mode='constant')
    time_signal = time_signal[0:samples]

    if window_length is None:
        window = window(size)
    else:
        #window = window(window_length, 4.0)
        window = window(window_length)
        window = np.pad(window, (0, size - window_length), mode='constant')

    time_signal_seg = segment_axis(time_signal, size,
                                   size - shift, axis=time_dim)

    letters = string.ascii_lowercase
    mapping = letters[:time_signal_seg.ndim] + ',' + letters[time_dim + 1] \
              + '->' + letters[:time_signal_seg.ndim]

    return rfft(np.einsum(mapping, time_signal_seg, window), n=nFFT,
                axis=time_dim + 1)


def istft(stft_signal, size=256, shift=128,
          window=signal.hamming, fading=False, window_length=None):
    """
    Calculated the inverse short time Fourier transform to exactly reconstruct
    the time signal.

    :param stft_signal: Single channel complex STFT signal
        with dimensions frames times size/2+1.
    :param size: Scalar FFT-size.
    :param shift: Scalar FFT-shift. Typically shift is a fraction of size.
    :param window: Window function handle.
    :param fading: Removes the additional padding, if done during STFT.
    :param window_length: Sometimes one desires to use a shorter window than
        the fft size. In that case, the window is padded with zeros.
        The default is to use the fft-size as a window size.
    :return: Single channel complex STFT signal
    :return: Single channel time signal.
    """
    assert stft_signal.shape[1] == size // 2 + 1

    if window_length is None:
        window = window(size)
    else:
        #window = window(window_length,4.0)
        window = window(window_length)
        window = np.pad(window, (0, size - window_length), mode='constant')

    window = _biorthogonal_window_loopy(window, shift)
    window *= size

    time_signal = scipy.zeros(stft_signal.shape[0] * shift + size - shift)

    for j, i in enumerate(range(0, len(time_signal) - size + shift, shift)):
        time_signal[i:i + size] += window * np.real(irfft(stft_signal[j]))

    # Compensate fade-in and fade-out
    if fading:
        time_signal = time_signal[
                      size - shift:len(time_signal) - (size - shift)]

    return time_signal


def audiowrite(data, path, samplerate=16000, normalize=False, threaded=True):
    """ Write the audio data ``data`` to the wav file ``path``

    The file can be written in a threaded mode. In this case, the writing
    process will be started at a separate thread. Consequently, the file will
    not be written when this function exits.

    :param data: A numpy array with the audio data
    :param path: The wav file the data should be written to
    :param samplerate: Samplerate of the audio data
    :param normalize: Normalize the audio first so that the values are within
        the range of [INTMIN, INTMAX]. E.g. no clipping occurs
    :param threaded: If true, the write process will be started as a separate
        thread
    :return: The number of clipped samples
    """
    data = data.copy()
    int16_max = np.iinfo(np.int16).max
    int16_min = np.iinfo(np.int16).min
    #import pdb; pdb.set_trace()
    if normalize:
        if not data.dtype.kind == 'f':
            data = data.astype(np.float)
        data /= np.max(np.abs(data))

    if data.dtype.kind == 'f':
        data *= int16_max

    sample_to_clip = np.sum(data > int16_max)
    if sample_to_clip > 0:
        print('Warning, clipping {} samples'.format(sample_to_clip))
    data = np.clip(data, int16_min, int16_max)
    data = data.astype(np.int16)

    if threaded:
        threading.Thread(target=wav_write,
                         args=(path, samplerate, data)).start()
    else:
        wav_write(path, samplerate, data)

    return sample_to_clip
    
    
def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis into overlapping frames.

    example:
    >>> segment_axis(np.arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    arguments:
    a       The array to segment
    length  The length of each frame
    overlap The number of array elements by which the frames should overlap
    axis    The axis to operate on; if None, act on the flattened array
    end     What to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:

            'cut'   Simply discard the extra values
            'wrap'  Copy values from the beginning of the array
            'pad'   Pad with a constant value

    endvalue    The value to use for end='pad'

    The array is not copied unless necessary (either because it is
    unevenly strided and being flattened or because end is set to
    'pad' or 'wrap').
    """

    if axis is None:
        a = np.ravel(a)  # may copy
        axis = 0

    l = a.shape[axis]

    if overlap >= length: raise ValueError(
            "frames cannot overlap by more than 100%")
    if overlap < 0 or length <= 0: raise ValueError(
            "overlap must be nonnegative and length must be positive")

    if l < length or (l - length) % (length - overlap):
        if l > length:
            roundup = length + (1 + (l - length) // (length - overlap)) * (
                length - overlap)
            rounddown = length + ((l - length) // (length - overlap)) * (
                length - overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (length - overlap) or (
            roundup == length and rounddown == 0)
        a = a.swapaxes(-1, axis)

        if end == 'cut':
            a = a[..., :rounddown]
        elif end in ['pad', 'wrap']:  # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = np.empty(s, dtype=a.dtype)
            b[..., :l] = a
            if end == 'pad':
                b[..., l:] = endvalue
            elif end == 'wrap':
                b[..., l:] = a[..., :roundup - l]
            a = b

        a = a.swapaxes(-1, axis)

    l = a.shape[axis]
    if l == 0: raise ValueError(
            "Not enough data points to segment array in 'cut' mode; "
            "try 'pad' or 'wrap'")
    assert l >= length
    assert (l - length) % (length - overlap) == 0
    n = 1 + (l - length) // (length - overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n, length) + a.shape[axis + 1:]
    newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[
                                                                  axis + 1:]

    if not a.flags.contiguous:
        a = a.copy()
        newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[
                                                                      axis + 1:]
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)

    try:
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)
    except TypeError or ValueError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[
                                                                      axis + 1:]
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)


def fastResample(src, tarLen):
    n           = len(src)
    y           = int(np.floor(np.log2(n)))
    nextpow2    = np.power(2,y+1)
    src_pad     = np.pad(src, ((0, nextpow2-n)), mode='constant')
    #print len(src_pad)
    #print int(1.0*tarLen/n*nextpow2)
    src_pad     = signal.resample(src_pad, int(1.0 * tarLen / n * nextpow2))
    return src_pad[0:tarLen]
    

def pcmread(speechFile, nChan):
    speech = np.memmap(speechFile, dtype='h', mode='r')
    speech.astype('float')
    speech = speech / 32768.0
    #if nChan > 1:
    speech = np.reshape(speech, (int(len(speech)/nChan), nChan))
    return speech

def pcmwrite(data, filename):
    fid     = open(filename, 'wb')
    data.astype('int')
    data    = np.minimum(np.maximum(data * 32767, -32767*np.ones(data.shape)), 32767*np.ones(data.shape))
    for idx_s in range(data.shape[0]):
        for idx_c in range(data.shape[1]):
            packed_vaule = struct.pack('h', data[idx_s][idx_c])
            fid.write(packed_vaule)
    fid.close()


if __name__ == "__main__":
    micPos = np.array([[0, 0, 0], [0, 0.035, 0]], dtype=float)
    getDelaySumWeightsMultiAng(16000, 512, micPos, 5)
