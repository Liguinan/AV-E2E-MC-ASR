#
# Created on Thu Jul 21 2022
# Author@ZHONG TAO
# Copyright (c) 2022 CUHK
#

from typing import Mapping
from data_tools.data_tool import DataTool
import pickle
import pathlib

class DataScpModify:

    WAV_MAPPING = {
        "/project_bdda6/bdda/jwyu/simulation/lrs3/pretrain/wav":
                "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/jwyu_bdda6/simulation/lrs3/pretrain/wav",
        "/project_bdda3/bdda/jwyu/JianweiYuFrameWork/lrs3/simulation_tool_tencent/lrs3/trainval/wav": 
                "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/jwyu_bdda3/JianweiYuFrameWork/lrs3/simulation_tool_tencent/lrs3/trainval/wav"

    }
    LIPEMB_MAPPING = {
        "/project_bdda3/bdda/jwyu/JianweiYuFrameWork/lrs3/visual_emb_oxford/LRS3-TED/pretrain/emb":
                "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/jwyu_bdda3/JianweiYuFrameWork/lrs3/visual_emb_oxford/LRS3-TED/pretrain/emb", 
        "/project_bdda3/bdda/jwyu/JianweiYuFrameWork/lrs3/visual_emb_oxford/LRS3-TED/trainval/emb":
                "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/jwyu_bdda3/JianweiYuFrameWork/lrs3/visual_emb_oxford/LRS3-TED/trainval/emb"

    }


    @classmethod
    def replace_data(cls, input_data_path, output_data_path):
        with open(input_data_path, "rb") as fp:
            data = pickle.load(fp, encoding="utf-8")
            for key in data:
                # replace wav_path
                wav_source_scp_path = data[key]['wav_path']
                wav_parent_dir = str(pathlib.Path(wav_source_scp_path).parent.resolve())
                assert wav_parent_dir in list(cls.WAV_MAPPING.keys())
                wav_target_scp_path = DataTool.replace_scp_location(wav_source_scp_path, cls.WAV_MAPPING[wav_parent_dir])
                data[key]['wav_path'] = wav_target_scp_path
                # replace lipemb_path
                lipemb_source_scp_path = data[key]['lipemb_path'][0]
                print(lipemb_source_scp_path)
                lipemb_parent_dir = str(pathlib.Path(lipemb_source_scp_path).parent.parent.resolve())
                libemb_idx = pathlib.Path(lipemb_source_scp_path).parent.name
                assert lipemb_parent_dir in list(cls.LIPEMB_MAPPING.keys())
                lipemb_target_scp_path = DataTool.replace_scp_location(lipemb_source_scp_path, f"{cls.LIPEMB_MAPPING[lipemb_parent_dir]}/{libemb_idx}")
                data[key]['lipemb_path'][0] = lipemb_target_scp_path
        output_data_file = f"{output_data_path}/{pathlib.Path(input_data_path).name}"
        with open(output_data_file, "wb") as fp:
            pickle.dump(data, fp)



def main():
    input_data_path = '/project_bdda5/bdda/jhxu/separation/choose_trptr_le6.pkl'
    output_data_path = '/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/bdda7_speech_enhancement_scp_data'
    DataScpModify.replace_data(input_data_path, output_data_path)

if __name__ == '__main__':
    main()