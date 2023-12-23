import os
import cv2
import numpy as np

from evaluation.evaluation_utils import interpolate_invalid_camera_extrinsics
from evaluation.datasets.base_dataset import EvaluationDataset


class ScanNetDataset(EvaluationDataset):

    def __init__(self,
                 dir_dataset: str,
                 num_evaluation_frames: int,
                 frame_height: int = 0,
                 frame_width: int = 0) -> None:
        super().__init__(dir_dataset=dir_dataset,
                         dataset_name='scannet',
                         num_evaluation_frames=num_evaluation_frames,
                         frame_height=frame_height,
                         frame_width=frame_width)

    def _load_dataset(self) -> None:
        self.files_color, self.files_depth = self._load_files()
        self.camera_intrinsics = self._load_camera_intrinsics()
        self.camera_extrinsics = self._load_camera_extrinsics()
        self.num_frames = len(self.files_color)

    def _load_camera_intrinsics(self) -> dict:
        camera_intrinsics = {}
        height, width, _ = cv2.imread(self.files_color[0]).shape
        if self.height == 0 or self.width == 0:
            self.height, self.width = height, width
        with open(self.dir_dataset + '/intrinsics/intrinsic_color.txt',
                  'r') as file:
            camera_intrinsics_matrix = np.array(
                list(map(float,
                         file.read().split()))).reshape(4, 4)[:3, :3]
        camera_intrinsics['height'] = height
        camera_intrinsics['width'] = width
        camera_intrinsics['fx'] = camera_intrinsics_matrix[0, 0]
        camera_intrinsics['fy'] = camera_intrinsics_matrix[1, 1]
        camera_intrinsics['cx'] = camera_intrinsics_matrix[0, 2]
        camera_intrinsics['cy'] = camera_intrinsics_matrix[1, 2]
        camera_intrinsics['depth_scale'] = 1000.0

        self.horizontal_padding = int(self.width * 0.1) if int(
            self.width * 0.1) % 2 == 0 else int(self.width * 0.1) + 1
        self.vertical_padding = int(self.height * 0.1) if int(
            self.height * 0.1) % 2 == 0 else int(self.height * 0.1) + 1
        scale_factor_x = (self.width + self.horizontal_padding) / \
            camera_intrinsics['width']
        scale_factor_y = (self.height + self.vertical_padding) / \
            camera_intrinsics['height']
        camera_intrinsics['width'] = self.width
        camera_intrinsics['height'] = self.height
        camera_intrinsics['fx'] *= scale_factor_x
        camera_intrinsics['fy'] *= scale_factor_y
        camera_intrinsics['cx'] *= scale_factor_x
        camera_intrinsics['cy'] *= scale_factor_y
        camera_intrinsics['cx'] -= self.horizontal_padding / 2
        camera_intrinsics['cy'] -= self.vertical_padding / 2
        return camera_intrinsics

    def _load_camera_extrinsics(self) -> list:
        files_camera_extrinsics = sorted([
            os.path.join(self.dir_dataset, 'extrinsics',
                         file) for file in os.listdir(
                             os.path.join(self.dir_dataset, 'extrinsics'))
        ])
        camera_extrinsics = []
        for file_camera_extrinsics in files_camera_extrinsics:
            with open(file_camera_extrinsics, 'r') as file:
                camera_extrinsics.append(
                    np.array(list(map(float,
                                      file.read().split()))).reshape(4, 4))

        return interpolate_invalid_camera_extrinsics(
            camera_extrinsics=np.stack(camera_extrinsics))

    def _load_files(self) -> tuple:
        files_color = sorted([
            os.path.join(self.dir_dataset, 'color', file)
            for file in os.listdir(os.path.join(self.dir_dataset, 'color'))
        ])
        files_depth = sorted([
            os.path.join(self.dir_dataset, 'depth', file)
            for file in os.listdir(os.path.join(self.dir_dataset, 'depth'))
        ])
        return files_color, files_depth

    def _load_frame_color(self, frame_index: int) -> np.array:
        file_color = self.files_color[frame_index]
        frame_color = cv2.resize(
            cv2.cvtColor(cv2.imread(file_color), cv2.COLOR_BGR2RGB),
            (self.width + self.horizontal_padding,
             self.height + self.vertical_padding))
        return frame_color[int(self.vertical_padding /
                               2):-int(self.vertical_padding / 2),
                           int(self.horizontal_padding /
                               2):-int(self.horizontal_padding / 2)]

    def _load_frame_depth(self, frame_index: int) -> np.array:
        file_depth = self.files_depth[frame_index]
        frame_depth = cv2.resize(
            cv2.imread(file_depth, cv2.IMREAD_ANYDEPTH) /
            self.camera_intrinsics['depth_scale'],
            (self.width + self.horizontal_padding,
             self.height + self.vertical_padding))
        return frame_depth[int(self.vertical_padding /
                               2):-int(self.vertical_padding / 2),
                           int(self.horizontal_padding /
                               2):-int(self.horizontal_padding / 2)]
