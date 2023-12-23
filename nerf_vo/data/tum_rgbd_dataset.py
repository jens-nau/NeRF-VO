import os
import cv2
import numpy as np

from nerf_vo.data.base_dataset import BaseDataset
from nerf_vo.data.data_utils import load_camera_intrinsics, read_timestamp_data, associate_timestamp_data


class TUMRGBDDataset(BaseDataset):

    def _load_dataset(self) -> None:
        self.files_color = self._load_files_color(
        )[self.first_frame_index:self.last_frame_index:self.stride]
        self.camera_intrinsics = self._load_camera_intrinsics()

    def _load_files_color(self) -> list:
        color_timestamp_data = read_timestamp_data(
            dir_dataset=self.dir_dataset, mode='color')
        depth_timestamp_data = read_timestamp_data(
            dir_dataset=self.dir_dataset, mode='depth')
        camera_extrinsics_timestamp_data = read_timestamp_data(
            dir_dataset=self.dir_dataset, mode='camera_extrinsics')

        association_color_depth = associate_timestamp_data(
            source_timestamps=list(color_timestamp_data.keys()),
            target_timestamps=list(depth_timestamp_data.keys()))
        association_color_camera_extrinsics = associate_timestamp_data(
            source_timestamps=[
                timestamp_color
                for timestamp_color, _ in association_color_depth
            ],
            target_timestamps=list(camera_extrinsics_timestamp_data.keys()))

        files_color = [
            color_timestamp_data[timestamp_color]
            for timestamp_color in sorted([
                timestamp_color
                for timestamp_color, _ in association_color_camera_extrinsics
            ])
        ]
        return [
            self.dir_dataset + f'/{file_color[0]}'
            for file_color in files_color
        ]

    def _load_camera_intrinsics(self) -> dict:
        dataset_name = os.path.basename(self.dir_dataset)
        if 'freiburg1' in dataset_name:
            dataset_name = 'fr1'
        elif 'freiburg2' in dataset_name:
            dataset_name = 'fr2'
        elif 'freiburg3' in dataset_name:
            dataset_name = 'fr3'
        else:
            raise NotImplementedError

        self.horizontal_padding = int(self.width * 0.1) if int(
            self.width * 0.1) % 2 == 0 else int(self.width * 0.1) + 1
        self.vertical_padding = int(self.height * 0.1) if int(
            self.height * 0.1) % 2 == 0 else int(self.height * 0.1) + 1
        camera_intrinsics = load_camera_intrinsics(
            dir_dataset=self.dir_dataset, dataset_name=dataset_name)
        self.raw_camera_intrinsics_matrix = np.array(
            [[camera_intrinsics['fx'], 0, camera_intrinsics['cx']],
             [0, camera_intrinsics['fy'], camera_intrinsics['cy']], [0, 0, 1]])
        self.raw_distortion_coefficients = np.array([
            camera_intrinsics['k1'], camera_intrinsics['k2'],
            camera_intrinsics['p1'], camera_intrinsics['p2'],
            camera_intrinsics['k3']
        ])

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
        camera_intrinsics = {
            key: value
            for key, value in camera_intrinsics.items()
            if key not in ['k1', 'k2', 'k3', 'p1', 'p2']
        }
        return camera_intrinsics

    def _get_frame(self, frame_index: int) -> dict:
        file_color = self.files_color[frame_index]
        frame_color = cv2.resize(
            cv2.undistort(
                cv2.cvtColor(cv2.imread(file_color), cv2.COLOR_BGR2RGB),
                self.raw_camera_intrinsics_matrix,
                self.raw_distortion_coefficients),
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
