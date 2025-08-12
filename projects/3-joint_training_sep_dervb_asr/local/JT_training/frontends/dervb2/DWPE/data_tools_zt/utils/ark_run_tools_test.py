#
# Created on Sun Jul 24 2022
# Author@ZHONG TAO
# Copyright (c) 2022 CUHK
#


import unittest
import json
from data_tools.utils.ark_run_tools import ArkRunTools
import pathlib
import librosa
import numpy 
import time
import pickle


class ArkRunToolsTest(unittest.TestCase):

    @unittest.skip
    def test_ark_run_for_python_wrapper(self):
        wav_scp_path = "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data_tools/log/wav.scp"
        featdir = "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data_tools/log/feats"
        output = ArkRunTools.ark_run_for_python_wrapper(wav_scp_path=wav_scp_path, featdir=featdir)
        print(output)

    @unittest.skip
    def test_ark_reader(self):
        ark_scp = "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data_tools/log/feats/feats.scp"
        with open(ark_scp) as fp:
            for line in fp:
                ark_input = line.split(' ')[1]
                result= ArkRunTools.ark_reader(ark_input)
                print(result)
                break

    @unittest.skip
    def test_ark_reader_case1(self):
        fs = 16000
        ark_scp = "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data_tools/log/feats/feats.scp"
        wav_scp_path = "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data_tools/log/wav.scp"
        with open(wav_scp_path) as fp1, open(ark_scp) as fp2:
            for line1, line2 in zip(fp1, fp2):
                key1, wav_input = line1.strip().split(' ')
                key2, ark_input = line2.split(' ')
                self.assertEqual(key1, key2)
                result1 =  librosa.load(wav_input, fs, mono=False)[0]
                result2 = ArkRunTools.ark_reader(ark_input)
                print(f"wav_dtype={result1.dtype}, ark_dtype={result2.dtype}" )
                numpy.testing.assert_equal(result1, result2/ArkRunTools.C)
                break

    @unittest.skip
    def test_ark_reader_time(self):
        fs = 16000
        ark_scp = "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data_tools/log/feats/feats.scp"
        wav_scp_path = "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data_tools/log/wav.scp"
        ark_time = 0.0
        wav_time = 0.0
        with open(wav_scp_path) as fp1, open(ark_scp) as fp2:
            for line1, line2 in zip(fp1, fp2):
                key1, wav_input = line1.strip().split(' ')
                key2, ark_input = line2.split(' ')
                self.assertEqual(key1, key2)
                wav_st = time.time()
                result1 =  librosa.load(wav_input, fs, mono=False)[0]
                wav_time += time.time() - wav_st
                ark_st = time.time()
                result2 = ArkRunTools.ark_reader(ark_input)
                ark_time += time.time() - ark_st
                # print(result1.dtype, result2.dtype)
        print("ark_time", ark_time, "wav_time", wav_time)

    @unittest.skip
    def test_ark_reader_case2(self):
        fs = 16000
        ark_scp = "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data_tools/log/feats/feats.scp"
        wav_scp_path = "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/wav_scp/wav.scp"
        with open(wav_scp_path) as fp1, open(ark_scp) as fp2:
            for line1, line2 in zip(fp1, fp2):
                key1, wav_input = line1.strip().split(' ')
                key2, ark_input = line2.split(' ')
                self.assertEqual(key1, key2)
                result1 =  librosa.load(wav_input, fs, mono=False)[0]
                result2 = ArkRunTools.ark_reader(ark_input)
                numpy.testing.assert_equal(result1, result2/ArkRunTools.C)

    def test_time_filter_pickle(self):
        original_pickle_file = "/project_bdda6/bdda/gnli/projects/LRS2_new/data/val_rev4.pkl"
        max_time = 6
        output_dir = "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data_tools/log"
        ArkRunTools.time_filter_pickle(original_pickle_file=original_pickle_file,
                                        max_time=max_time, output_dir=output_dir)
        pickle_path = "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data_tools/log/val_rev4_le6.pkl"
        with open(pickle_path, "rb") as fp:
            data = pickle.load(fp, encoding="utf-8")
            for key in data:
                st, et = data[key]['time_idx'][0]
                sp = et - st
                if sp > 5:
                    print(sp)
                self.assertLessEqual(sp, max_time)



if __name__ == '__main__':
    unittest.main()


