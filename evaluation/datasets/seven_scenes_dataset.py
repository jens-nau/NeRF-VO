import os
import cv2
import numpy as np

from evaluation.datasets.base_dataset import EvaluationDataset


class SevenScenesDataset(EvaluationDataset):

    def __init__(self,
                 dir_dataset: str,
                 num_evaluation_frames: int,
                 frame_height: int = 0,
                 frame_width: int = 0) -> None:
        super().__init__(dir_dataset=dir_dataset,
                         dataset_name='7-scenes',
                         num_evaluation_frames=num_evaluation_frames,
                         frame_height=frame_height,
                         frame_width=frame_width)

    def _load_camera_extrinsics(self) -> list:
        files_camera_extrinsics = sorted([
            os.path.join(self.dir_dataset + '/seq-01', file)
            for file in os.listdir(self.dir_dataset + '/seq-01')
            if file.endswith('pose.txt')
        ])

        camera_extrinsics = []
        for file_camera_extrinsics in files_camera_extrinsics:
            with open(file_camera_extrinsics, 'r') as file:
                camera_extrinsics.append(
                    np.array(list(map(float,
                                      file.read().split()))).reshape(4, 4))

        return np.stack(camera_extrinsics)

    def _load_files(self) -> tuple:
        files_color = sorted([
            os.path.join(self.dir_dataset + '/seq-01', file)
            for file in os.listdir(self.dir_dataset + '/seq-01')
            if file.endswith('color.png')
        ])
        files_depth = sorted([
            os.path.join(self.dir_dataset + '/seq-01', file)
            for file in os.listdir(self.dir_dataset + '/seq-01')
            if file.endswith('depth.png')
        ])
        return files_color, files_depth

    def _load_frame_depth(self, frame_index: int) -> np.array:
        frame_depth = cv2.resize(
            cv2.imread(self.files_depth[frame_index], cv2.IMREAD_ANYDEPTH),
            (self.camera_intrinsics['width'],
             self.camera_intrinsics['height']))
        frame_depth[frame_depth == 65535] = 0
        return frame_depth / self.camera_intrinsics['depth_scale']
