#
# Created on Thu Jul 21 2022
# Author@ZHONG TAO
# Copyright (c) 2022 CUHK
#

import unittest
import pickle
import pathlib
from data_tools.data_scp_modify import DataScpModify
import librosa
import numpy 
import time


class DataScpModifyTest(unittest.TestCase):

    @unittest.skip
    def test_replace_data(self):
        input_data_path_origin = '/project_bdda5/bdda/jhxu/separation/choose_trptr_le6.pkl'
        input_data_path_modify = '/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/bdda7_speech_enhancement_scp_data/choose_trptr_le6.pkl'
        with open(input_data_path_origin, "rb") as fp1, open(input_data_path_modify, "rb") as fp2:
            data1 = pickle.load(fp1, encoding="utf8")
            data2 = pickle.load(fp2, encoding="utf8")
            self.assertEqual(list(data1.keys()), list(data2.keys()))
            for key1, key2 in zip(data1, data2):
                self.assertEqual(key1, key2)

                # test wav_path
                source_scp_path = data1[key1]['wav_path']
                target_scp_path = data2[key2]['wav_path']
                parent_dir = str(pathlib.Path(source_scp_path).parent.resolve())
                self.assertTrue(parent_dir in DataScpModify.WAV_MAPPING.keys())
                self.assertEqual(DataScpModify.WAV_MAPPING[parent_dir], str(pathlib.Path(target_scp_path).parent.resolve()))
                self.assertEqual(pathlib.Path(source_scp_path).name, pathlib.Path(target_scp_path).name)
                self.assertEqual(target_scp_path.split('/')[1], "project_bdda7")


                # test libemb_path
                source_scp_path = data1[key1]['lipemb_path'][0]
                target_scp_path = data2[key2]['lipemb_path'][0]
                parent_dir = str(pathlib.Path(source_scp_path).parent.parent.resolve())
                self.assertTrue(parent_dir in DataScpModify.LIPEMB_MAPPING.keys())
                self.assertEqual(DataScpModify.LIPEMB_MAPPING[parent_dir], str(pathlib.Path(target_scp_path).parent.parent.resolve()))
                self.assertEqual(pathlib.Path(source_scp_path).name, pathlib.Path(target_scp_path).name)
                self.assertEqual(pathlib.Path(source_scp_path).parent.name, pathlib.Path(target_scp_path).parent.name)


                self.assertEqual(data1[key1]['n_spk'], data2[key2]['n_spk'])
                self.assertEqual(data1[key1]['spk_doa'], data2[key2]['spk_doa'])
                self.assertEqual(data1[key1]['time_idx'], data2[key2]['time_idx'])

    @unittest.skip
    def test_replace_case1(self):
        key = "QbxinUJcLGg-50009"
        sampling_rate = 16000
        input_data_path_origin = '/project_bdda5/bdda/jhxu/separation/choose_trptr_le6.pkl'
        input_data_path_modify = '/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/bdda7_speech_enhancement_scp_data/choose_trptr_le6.pkl'
        with open(input_data_path_origin, "rb") as fp1, open(input_data_path_modify, "rb") as fp2:
            data1 = pickle.load(fp1, encoding="utf8")
            data2 = pickle.load(fp2, encoding="utf8")
            wav1, _ = librosa.load(data1[key]['wav_path'], sampling_rate, mono=False)
            wav2, _ = librosa.load(data2[key]['wav_path'], sampling_rate, mono=False)
            numpy.testing.assert_equal(wav1, wav2)
    
    @unittest.skip
    def test_replace_data_load(self):
        sampling_rate = 16000
        input_data_path_origin = '/project_bdda5/bdda/jhxu/separation/choose_trptr_le6.pkl'
        input_data_path_modify = '/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/bdda7_speech_enhancement_scp_data/choose_trptr_le6.pkl'
        with open(input_data_path_origin, "rb") as fp1, open(input_data_path_modify, "rb") as fp2:
            data1 = pickle.load(fp1, encoding="utf8")
            data2 = pickle.load(fp2, encoding="utf8")
            self.assertEqual(list(data1.keys()), list(data2.keys()))
            for key1, key2 in zip(data1, data2):
                self.assertEqual(key1, key2)

                # test wav_path
                source_scp_path = data1[key1]['wav_path']
                target_scp_path = data2[key2]['wav_path']
                self.assertNotEqual(source_scp_path, target_scp_path)
                wav1, _ = librosa.load(source_scp_path, sampling_rate, mono=False)
                wav2, _ = librosa.load(target_scp_path, sampling_rate, mono=False)
                self.assertEqual(wav1.shape, wav2.shape)
                numpy.testing.assert_equal(wav1, wav2)

                # test lipemb_path

                source_scp_path = data1[key1]['lipemb_path'][0]
                target_scp_path = data2[key2]['lipemb_path'][0]
                self.assertNotEqual(source_scp_path, target_scp_path)
                lip_video1 = numpy.load(source_scp_path)
                lip_video2 = numpy.load(target_scp_path)
                self.assertEqual(lip_video1.shape, lip_video2.shape)
                numpy.testing.assert_equal(lip_video1, lip_video2)
                print(key1)

    @unittest.skip
    def test_load_time(self):
        sampling_rate = 16000
        input_data_path_origin = '/project_bdda5/bdda/jhxu/separation/choose_trptr_le6.pkl'
        input_data_path_modify = '/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/bdda7_speech_enhancement_scp_data/choose_trptr_le6.pkl'
        time1 = 0.0
        time2 = 0.0
        with open(input_data_path_origin, "rb") as fp1, open(input_data_path_modify, "rb") as fp2:
            data1 = pickle.load(fp1, encoding="utf8")
            data2 = pickle.load(fp2, encoding="utf8")
            self.assertEqual(list(data1.keys()), list(data2.keys()))
            step = 0
            wav_list = list(data1.keys())
            numpy.random.shuffle(wav_list)
            # for key1, key2 in zip(data1, data2):
            for key in wav_list:
                key1 = key2 = key
                self.assertEqual(key1, key2)

                # test wav_path
                source_scp_path = data1[key1]['wav_path']
                target_scp_path = data2[key2]['wav_path']
                print("source_scp_path", source_scp_path)
                print("target_scp_path", target_scp_path)
                self.assertNotEqual(source_scp_path, target_scp_path)
                st1 = time.time()
                wav1, _ = librosa.load(source_scp_path, sampling_rate, mono=False)
                tmp1 = time.time() - st1
                time1 += tmp1
                st2 = time.time()
                wav2, _ = librosa.load(target_scp_path, sampling_rate, mono=False)
                tmp2 = time.time() - st2
                time2 += tmp2
                self.assertEqual(wav1.shape, wav2.shape)
                step += 1
                print(tmp1, tmp2)
                if step == 100:
                    print(time1, time2)
                    break
                



if __name__ == '__main__':
    unittest.main()