# coding: utf-8
import librosa
import librosa.display
import soundfile


def extract_single_channel_wav(fpath, fname):
    y, sr = librosa.load(f"{fpath}/{fname}.wav", sr=16000, mono=False)
    # soundfile.write(f"{fpath}/clean/{fname}.wav", y[15, :], sr)
    # soundfile.write(f"{fpath}/rvb-only/{fname}.wav", y[16, :], sr)
    soundfile.write(f"{fpath}/mixture/{fname}.wav", y[0, :], sr)


path = "replayed/0-15/0-mc"
name = "6352498867394358158-00019"
extract_single_channel_wav(path, name)

path = "replayed/15-45/0-mc"
name = "6352893145392128503-00036"
extract_single_channel_wav(path, name)

path = "replayed/45-90/0-mc"
name = "6392525815109495392-00008"
extract_single_channel_wav(path, name)



# replayed
# 6392525815109495392-00008 female angle_difference=90  WER=0
# /project_bdda6/bdda/gnli/projects/LRS2/data_prepare/wav_data/replay/record_out/6392525815109495392-00008.wav
# {'lip_path': ['/CAdisk/tomasyu/LRS2_image/LRS2/data/main/6392525815109495392/00008.npz', '/CAdisk/tomasyu/LRS2_image/LRS2/data/main/6392525815109495392/00008.npz'], 'lipemb_path': ['/project_bdda6/bdda/gnli/projects/LRS2/data_prepare/lip_emb_data/test/emb/6392525815109495392/00008.npy'], 'wav_path': '/project_bdda6/bdda/gnli/projects/LRS2/data_prepare/wav_data/replay/record_out/6392525815109495392-00008.wav', 'n_spk': 2, 'spk_id': ['6392525815109495392', '6343608285091658983'], 'spk_doa': ['150', '60'], 'time_idx': [[0.0, 1.792]]}

# 6352893145392128503-00036 female angle_difference=30 WER=1
# /project_bdda6/bdda/gnli/projects/LRS2/data_prepare/wav_data/replay/record_out/6352893145392128503-00036.wav
# {'lip_path': ['/CAdisk/tomasyu/LRS2_image/LRS2/data/main/6352893145392128503/00036.npz', '/CAdisk/tomasyu/LRS2_image/LRS2/data/main/6352893145392128503/00036.npz'], 'lipemb_path': ['/project_bdda6/bdda/gnli/projects/LRS2/data_prepare/lip_emb_data/test/emb/6352893145392128503/00036.npy'], 'wav_path': '/project_bdda6/bdda/gnli/projects/LRS2/data_prepare/wav_data/replay/record_out/6352893145392128503-00036.wav', 'n_spk': 2, 'spk_id': ['6352893145392128503', '6343608285091658983'], 'spk_doa': ['30', '60'], 'time_idx': [[0.0, 2.112]]}

# 6352498867394358158-00019 male angle_difference=15 WER=0
# /project_bdda6/bdda/gnli/projects/LRS2/data_prepare/wav_data/replay/record_out/6352498867394358158-00019.wav
# {'lip_path': ['/CAdisk/tomasyu/LRS2_image/LRS2/data/main/6352498867394358158/00019.npz', '/CAdisk/tomasyu/LRS2_image/LRS2/data/main/6352498867394358158/00019.npz'], 'lipemb_path': ['/project_bdda6/bdda/gnli/projects/LRS2/data_prepare/lip_emb_data/test/emb/6352498867394358158/00019.npy'], 'wav_path': '/project_bdda6/bdda/gnli/projects/LRS2/data_prepare/wav_data/replay/record_out/6352498867394358158-00019.wav', 'n_spk': 2, 'spk_id': ['6352498867394358158', '6371470597064737331'], 'spk_doa': ['15', '30'], 'time_idx': [[0.0, 1.408]]}