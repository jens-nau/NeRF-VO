import os
import cv2
import numpy as np

from nerf_vo.data.base_dataset import BaseDataset


class ScanNetDataset(BaseDataset):

    def _load_dataset(self) -> None:
        self.files_color = self._load_files_color(
        )[self.first_frame_index:self.last_frame_index:self.stride]
        self.camera_intrinsics = self._load_camera_intrinsics()

    def _load_files_color(self) -> list:
        return sorted([
            os.path.join(self.dir_dataset, 'color', file)
            for file in os.listdir(os.path.join(self.dir_dataset, 'color'))
        ])

    def _load_camera_intrinsics(self) -> dict:
        camera_intrinsics = {}
        height, width, _ = cv2.imread(self.files_color[0]).shape
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

    def _get_frame(self, frame_index: int) -> dict:
        file_color = self.files_color[frame_index]
        frame_color = cv2.resize(
            cv2.cvtColor(cv2.imread(file_color), cv2.COLOR_BGR2RGB),
            (self.width + self.horizontal_padding,
             self.height + self.vertical_padding))
        frame_color = frame_color[int(self.vertical_padding /
                                      2):-int(self.vertical_padding / 2),
                                  int(self.horizontal_padding /
                                      2):-int(self.horizontal_padding / 2)]

        return {
            'frame_index': frame_index,
            'camera_intrinsics': self.camera_intrinsics,
            'frame_color': frame_color,
            'last_frame': (frame_index >= self.__len__() - 1),
        }
