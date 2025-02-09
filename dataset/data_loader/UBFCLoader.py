"""The dataloader for UBFC datasets.

Details for the UBFC-RPPG Dataset see https://sites.google.com/view/ybenezeth/ubfcrppg.
If you use this dataset, please cite this paper:
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm

# UBFC 数据集的 GT 有三行，第一行代表 BVP 信号，这里原本只加载了第一行，实际上确实只拟合 BVP 就也行
# 但是在 余梓彤 老师的代码中还需要 HR 数据，因此补充加载了第二行
# 第三行不知道是什么含义，也没见人用过

# 需要注意的是：在 UBFC 中，subject20、24 的 HR 数据有问题，在 physnet 等需要用到 HR 数据的方法中，需要将这两个 subject 删除
# 想要可视化的话以下是代码：

# datapth = '/data/chushuyang/UBFC_RAW/subject20/ground_truth.txt'
# with open(datapth, 'r') as f:
#     data = f.readlines()
# dataBvp = [float(strr) for strr in list(data[0].split())]
# dataBvp = np.array(dataBvp)
# dataHr = [float(strr) for strr in list(data[1].split())] 
# dataHr = np.array(dataHr)
# dataUnknow = [float(strr) for strr in list(data[2].split())]
# dataUnknow = np.array(dataUnknow)
# print(len(dataHr))
# print(len(dataBvp))
# fig, ax = plt.subplots(1, 3, figsize=(20, 5))
# ax[0].plot(np.arange(len(dataBvp)), dataBvp)
# ax[0].set_title('BVP')
# ax[1].plot(np.arange(len(dataHr)), dataHr)
# ax[1].set_title('HR')
# ax[2].plot(np.arange(len(dataUnknow)), dataUnknow)
# ax[2].set_title('Unknow')

class UBFCLoader(BaseLoader):
    """The data loader for the UBFC dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an UBFC dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- subject1/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                     |   |-- subject2/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                     |...
                     |   |-- subjectn/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_data(self, data_path):
        """Returns data directories under the path(For UBFC dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "subject*")
        if not data_dirs:
            raise ValueError(self.name + " dataset get data error!")
        dirs = [{"index": re.search(
            'subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        return dirs

    def get_data_subset(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values"""
        if begin == 0 and end == 1: # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """   invoked by preprocess_dataset for multi_process.   """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        frames = self.read_video(os.path.join(data_dirs[i]['path'],"001vid.avi"))   # 这里要改成001vid.avi，原本是 vid.avi，这个问题是普遍问题，或许是服务器上数据的命名不对
        bvps = self.read_wave(os.path.join(data_dirs[i]['path'],"ground_truth.txt"))
        hrs = self.read_hr(os.path.join(data_dirs[i]['path'],"ground_truth.txt"))

        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess, config_preprocess.LARGE_FACE_BOX)

        count, input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")
            bvp = [float(x) for x in str1[0].split()]
        return np.asarray(bvp)

    @staticmethod
    def read_hr(hr_file):
        """Reads a hr signal file."""
        with open(hr_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")
            hr = [float(x) for x in str1[1].split()]
        return np.asarray(hr)
