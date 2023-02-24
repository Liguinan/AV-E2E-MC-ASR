# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import matplotlib.pyplot as plt

# name = "simulated/0-15/clean/00011"


def print_figure(name):
    y, sr = librosa.load(f"{name}.wav", sr=16000, mono=False)
    # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig, ax = plt.subplots()
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=256)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='linear', x_axis='s', hop_length=256, sr=sr, ax=ax)
    # ax.set(title='Linear-frequency power spectrogram')
    # ax.label_outer()
    plt.savefig(f"{name}.png")
    # plt.show()

# clean
# name1 = "test/official_clean/00018"
# print_figure(name1)
# name2 = "test/official_clean/00019"
# print_figure(name2)
# name3 = "test/official_clean/00027"
# print_figure(name3)
# name4 = "test/official_clean/00029"
# print_figure(name4)


# mixture
# name1 = "simulated/0-15/0-mc/mixture/00019"
# print_figure(name1)
# name2 = "simulated/15-45/0-mc/mixture/00027"
# print_figure(name2)
# name3 = "simulated/45-90/0-mc/mixture/00029"
# print_figure(name3)
# name4 = "simulated/90-180/0-mc/mixture/00018"
# print_figure(name4)

# # rvb only
# name1 = "simulated/0-15/0-mc/rvb-only/00019"
# print_figure(name1)
# name2 = "simulated/15-45/0-mc/rvb-only/00027"
# print_figure(name2)
# name3 = "simulated/45-90/0-mc/rvb-only/00029"
# print_figure(name3)
# name4 = "simulated/90-180/0-mc/rvb-only/00018"
# print_figure(name4)

# clean
# name1 = "simulated/0-15/0-mc/clean/00019"
# print_figure(name1)
# name2 = "simulated/15-45/0-mc/clean/00027"
# print_figure(name2)
# name3 = "simulated/45-90/0-mc/clean/00029"
# print_figure(name3)
# name4 = "simulated/90-180/0-mc/clean/00018"
# print_figure(name4)


# [1] ao-mvdr-only pipelined
name1 = "simulated/0-15/1-ao-mvdr-only/pipelined/rev1-6331559613336179781-00019"
print_figure(name1)
name2 = "simulated/15-45/1-ao-mvdr-only/pipelined/rev2-6331559613336179781-00027"
print_figure(name2)
name3 = "simulated/45-90/1-ao-mvdr-only/pipelined/rev3-6331559613336179781-00029"
print_figure(name3)
name4 = "simulated/90-180/1-ao-mvdr-only/pipelined/rev4-6332062124509813446-00018"
print_figure(name4)
# [1] ao-mvdr-only joint2
name1 = "simulated/0-15/1-ao-mvdr-only/joint2/rev1-6331559613336179781-00019"
print_figure(name1)
name2 = "simulated/15-45/1-ao-mvdr-only/joint2/rev2-6331559613336179781-00027"
print_figure(name2)
name3 = "simulated/45-90/1-ao-mvdr-only/joint2/rev3-6331559613336179781-00029"
print_figure(name3)
name4 = "simulated/90-180/1-ao-mvdr-only/joint2/rev4-6332062124509813446-00018"
print_figure(name4)

