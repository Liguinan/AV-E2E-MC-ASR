# coding: utf-8
import librosa
import librosa.display
import soundfile


def extract_single_channel_wav(fpath, fname):
    y, sr = librosa.load(f"{fpath}/{fname}.wav", sr=16000, mono=False)
    soundfile.write(f"{fpath}/clean/{fname}.wav", y[15, :], sr)
    soundfile.write(f"{fpath}/rvb-only/{fname}.wav", y[16, :], sr)
    soundfile.write(f"{fpath}/mixture/{fname}.wav", y[0, :], sr)


# path = "simulated/0-15/0-mc"
# name = "00019"
# extract_single_channel_wav(path, name)
#
# path = "simulated/15-45/0-mc"
# name = "00027"
# extract_single_channel_wav(path, name)
#
# path = "simulated/45-90/0-mc"
# name = "00029"
# extract_single_channel_wav(path, name)
#
# path = "simulated/90-180/0-mc"
# name = "00018"
# extract_single_channel_wav(path, name)


# simulated
# rev1-6331559613336179781-00019
# rev2-6331559613336179781-00027
# rev3-6331559613336179781-00029
# rev4-6332062124509813446-00018