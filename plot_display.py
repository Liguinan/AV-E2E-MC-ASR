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


# # [1] ao-mvdr-only pipelined
# name1 = "simulated/0-15/1-ao-mvdr-only/pipelined/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/1-ao-mvdr-only/pipelined/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/1-ao-mvdr-only/pipelined/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/1-ao-mvdr-only/pipelined/rev4-6332062124509813446-00018"
# print_figure(name4)
# # [1] ao-mvdr-only joint2
# name1 = "simulated/0-15/1-ao-mvdr-only/joint2/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/1-ao-mvdr-only/joint2/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/1-ao-mvdr-only/joint2/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/1-ao-mvdr-only/joint2/rev4-6332062124509813446-00018"
# print_figure(name4)

# # [2] av-mvdr-only pipelined
# name1 = "simulated/0-15/2-av-mvdr-only/pipelined/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/2-av-mvdr-only/pipelined/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/2-av-mvdr-only/pipelined/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/2-av-mvdr-only/pipelined/rev4-6332062124509813446-00018"
# print_figure(name4)
# # [1] av-mvdr-only joint2
# name1 = "simulated/0-15/2-av-mvdr-only/joint2/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/2-av-mvdr-only/joint2/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/2-av-mvdr-only/joint2/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/2-av-mvdr-only/joint2/rev4-6332062124509813446-00018"
# print_figure(name4)



# # [3] ao-mvdr-dwpe pipelined
# name1 = "simulated/0-15/3-ao-mvdr-dwpe/pipelined/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/3-ao-mvdr-dwpe/pipelined/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/3-ao-mvdr-dwpe/pipelined/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/3-ao-mvdr-dwpe/pipelined/rev4-6332062124509813446-00018"
# print_figure(name4)
# # [3] ao-mvdr-dwpe joint2
# name1 = "simulated/0-15/3-ao-mvdr-dwpe/joint2/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/3-ao-mvdr-dwpe/joint2/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/3-ao-mvdr-dwpe/joint2/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/3-ao-mvdr-dwpe/joint2/rev4-6332062124509813446-00018"
# print_figure(name4)


# # [4] av-mvdr-dwpe pipelined
# name1 = "simulated/0-15/4-av-mvdr-dwpe/pipelined/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/4-av-mvdr-dwpe/pipelined/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/4-av-mvdr-dwpe/pipelined/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/4-av-mvdr-dwpe/pipelined/rev4-6332062124509813446-00018"
# print_figure(name4)
# # [4] av-mvdr-dwpe joint2
# name1 = "simulated/0-15/4-av-mvdr-dwpe/joint2/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/4-av-mvdr-dwpe/joint2/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/4-av-mvdr-dwpe/joint2/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/4-av-mvdr-dwpe/joint2/rev4-6332062124509813446-00018"
# print_figure(name4)


# # [5] ao-mvdr-specm pipelined
# name1 = "simulated/0-15/5-ao-mvdr-specm/pipelined/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/5-ao-mvdr-specm/pipelined/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/5-ao-mvdr-specm/pipelined/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/5-ao-mvdr-specm/pipelined/rev4-6332062124509813446-00018"
# print_figure(name4)
# # [5] ao-mvdr-specm joint2
# name1 = "simulated/0-15/5-ao-mvdr-specm/joint2/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/5-ao-mvdr-specm/joint2/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/5-ao-mvdr-specm/joint2/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/5-ao-mvdr-specm/joint2/rev4-6332062124509813446-00018"
# print_figure(name4)
#
#
# # [6] av-mvdr-specm pipelined
# name1 = "simulated/0-15/6-av-mvdr-specm/pipelined/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/6-av-mvdr-specm/pipelined/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/6-av-mvdr-specm/pipelined/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/6-av-mvdr-specm/pipelined/rev4-6332062124509813446-00018"
# print_figure(name4)
# # [6] av-mvdr-specm joint2
# name1 = "simulated/0-15/6-av-mvdr-specm/joint2/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/6-av-mvdr-specm/joint2/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/6-av-mvdr-specm/joint2/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/6-av-mvdr-specm/joint2/rev4-6332062124509813446-00018"
# print_figure(name4)


# # [7] ao-dwpe-mvdr pipelined
# name1 = "simulated/0-15/7-ao-dwpe-mvdr/pipelined/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/7-ao-dwpe-mvdr/pipelined/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/7-ao-dwpe-mvdr/pipelined/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/7-ao-dwpe-mvdr/pipelined/rev4-6332062124509813446-00018"
# print_figure(name4)
# # [7] ao-dwpe-mvdr joint2
# name1 = "simulated/0-15/7-ao-dwpe-mvdr/joint2/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/7-ao-dwpe-mvdr/joint2/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/7-ao-dwpe-mvdr/joint2/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/7-ao-dwpe-mvdr/joint2/rev4-6332062124509813446-00018"
# print_figure(name4)
#
#
# # [8] av-dwpe-mvdr pipelined
# name1 = "simulated/0-15/8-av-dwpe-mvdr/pipelined/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/8-av-dwpe-mvdr/pipelined/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/8-av-dwpe-mvdr/pipelined/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/8-av-dwpe-mvdr/pipelined/rev4-6332062124509813446-00018"
# print_figure(name4)
# # [8] av-dwpe-mvdr joint2
# name1 = "simulated/0-15/8-av-dwpe-mvdr/joint2/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/8-av-dwpe-mvdr/joint2/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/8-av-dwpe-mvdr/joint2/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/8-av-dwpe-mvdr/joint2/rev4-6332062124509813446-00018"
# print_figure(name4)


# # [9] ao-specm-mvdr pipelined
# name1 = "simulated/0-15/9-ao-specm-mvdr/pipelined/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/9-ao-specm-mvdr/pipelined/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/9-ao-specm-mvdr/pipelined/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/9-ao-specm-mvdr/pipelined/rev4-6332062124509813446-00018"
# print_figure(name4)
# # [9] ao-specm-mvdrjoint2
# name1 = "simulated/0-15/9-ao-specm-mvdr/joint2/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/9-ao-specm-mvdr/joint2/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/9-ao-specm-mvdr/joint2/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/9-ao-specm-mvdr/joint2/rev4-6332062124509813446-00018"
# print_figure(name4)
#
#
# # [10] av-specm-mvdr pipelined
# name1 = "simulated/0-15/10-av-specm-mvdr/pipelined/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/10-av-specm-mvdr/pipelined/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/10-av-specm-mvdr/pipelined/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/10-av-specm-mvdr/pipelined/rev4-6332062124509813446-00018"
# print_figure(name4)
# # [10] av-specm-mvdr joint2
# name1 = "simulated/0-15/10-av-specm-mvdr/joint2/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/10-av-specm-mvdr/joint2/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/10-av-specm-mvdr/joint2/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/10-av-specm-mvdr/joint2/rev4-6332062124509813446-00018"
# print_figure(name4)


# # [11] ao-wpd pipelined
# name1 = "simulated/0-15/11-ao-wpd/pipelined/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/11-ao-wpd/pipelined/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/11-ao-wpd/pipelined/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/11-ao-wpd/pipelined/rev4-6332062124509813446-00018"
# print_figure(name4)
# # [11] ao-wpd joint2
# name1 = "simulated/0-15/11-ao-wpd/joint2/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/11-ao-wpd/joint2/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/11-ao-wpd/joint2/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/11-ao-wpd/joint2/rev4-6332062124509813446-00018"
# print_figure(name4)
#
#
# # [12] av-wpd pipelined
# name1 = "simulated/0-15/12-av-wpd/pipelined/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/12-av-wpd/pipelined/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/12-av-wpd/pipelined/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/12-av-wpd/pipelined/rev4-6332062124509813446-00018"
# print_figure(name4)
# # [12] av-wpd joint2
# name1 = "simulated/0-15/12-av-wpd/joint2/rev1-6331559613336179781-00019"
# print_figure(name1)
# name2 = "simulated/15-45/12-av-wpd/joint2/rev2-6331559613336179781-00027"
# print_figure(name2)
# name3 = "simulated/45-90/12-av-wpd/joint2/rev3-6331559613336179781-00029"
# print_figure(name3)
# name4 = "simulated/90-180/12-av-wpd/joint2/rev4-6332062124509813446-00018"
# print_figure(name4)
