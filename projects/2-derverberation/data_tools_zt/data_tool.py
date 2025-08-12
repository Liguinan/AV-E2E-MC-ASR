#
# Created on Thu Jul 21 2022
# Author@ZHONG TAO
# Copyright (c) 2022 CUHK
#

#
# Created on Thu Jul 21 2022
# Author@ZHONG TAO
# Copyright (c) 2022 CUHK
#


import pathlib


class DataTool:

    @staticmethod
    def replace_scp_location(source_scp_path, target_scp_dir):
        return f"{target_scp_dir}/{pathlib.Path(source_scp_path).name}"


