import os
import cv2
import numpy as np
import scipy

from evaluation.evaluation_utils import read_timestamp_data, associate_timestamp_data
from evaluation.datasets.base_dataset import EvaluationDataset


class TUMRGBDDataset(EvaluationDataset):

    def __init__(self,
                 dir_dataset: str,
                 num_evaluation_frames: int,
                 frame_height: int = 0,
                 frame_width: int = 0) -> None:
        dataset_name = os.path.basename(dir_dataset)
        if 'freiburg1' in dataset_name:
            dataset_name = 'fr1'
        elif 'freiburg2' in dataset_name:
            dataset_name = 'fr2'
        elif 'freiburg3' in dataset_name:
            dataset_name = 'fr3'
        else:
            raise NotImplementedError

        super().__init__(dir_dataset=dir_dataset,
                         dataset_name=dataset_name,
                         num_evaluation_frames=num_evaluation_frames,
                         frame_height=frame_height,
                         frame_width=frame_width)

    def _load_dataset(self) -> None:
        self.files_color, self.files_depth, self.camera_tquads_origin2frame = self._load_files(
        )
        self.camera_intrinsics = self._load_camera_intrinsics()
        self.camera_extrinsics = self._load_camera_extrinsics()
        self.num_frames = len(self.files_color)

    def _load_camera_intrinsics(self) -> dict:
        camera_intrinsics = super()._load_camera_intrinsics()
        self.horizontal_padding = int(self.width * 0.1) if int(
            self.width * 0.1) % 2 == 0 else int(self.width * 0.1) + 1
        self.vertical_padding = int(self.height * 0.1) if int(
            self.height * 0.1) % 2 == 0 else int(self.height * 0.1) + 1
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

    def _load_camera_extrinsics(self) -> list:
        camera_extrinsics = np.tile(
            np.eye(4), (self.camera_tquads_origin2frame.shape[0], 1, 1))
        camera_extrinsics[:, :3, 3] = self.camera_tquads_origin2frame[:, :3]
        camera_extrinsics[:, :3, :
                          3] = scipy.spatial.transform.Rotation.from_quat(
                              self.camera_tquads_origin2frame[:,
                                                              3:]).as_matrix()
        return camera_extrinsics

    def _load_files(self) -> tuple:
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
        files_depth = [
            depth_timestamp_data[timestamp_depth]
            for timestamp_depth in sorted([
                timestamp_depth
                for timestamp_color, timestamp_depth in association_color_depth
                if timestamp_color in [
                    timestamp_color for timestamp_color, _ in
                    association_color_camera_extrinsics
                ]
            ])
        ]
        camera_tquads_origin2frame = np.stack([
            camera_extrinsics_timestamp_data[timestamp_camera_extrinsics]
            for timestamp_camera_extrinsics in sorted([
                timestamp_camera_extrinsics for _, timestamp_camera_extrinsics
                in association_color_camera_extrinsics
            ])
        ])

        files_color = [
            self.dir_dataset + f'/{file_color[0]}'
            for file_color in files_color
        ]
        files_depth = [
            self.dir_dataset + f'/{file_depth[0]}'
            for file_depth in files_depth
        ]

        return files_color, files_depth, camera_tquads_origin2frame

    def _load_frame_color(self, frame_index: int) -> np.array:
        file_color = self.files_color[frame_index]
        frame_color = cv2.resize(
            cv2.undistort(
                cv2.cvtColor(cv2.imread(file_color), cv2.COLOR_BGR2RGB),
                self.raw_camera_intrinsics_matrix,
                self.raw_distortion_coefficients),
            (self.width + self.horizontal_padding,
             self.height + self.vertical_padding))
        return frame_color[int(self.vertical_padding /
                               2):-int(self.vertical_padding / 2),
                           int(self.horizontal_padding /
                               2):-int(self.horizontal_padding / 2)]

    def _load_frame_depth(self, frame_index: int) -> np.array:
        file_depth = self.files_depth[frame_index]
        frame_depth = cv2.resize(
            cv2.undistort(
                cv2.imread(file_depth, cv2.IMREAD_ANYDEPTH) /
                self.camera_intrinsics['depth_scale'],
                self.raw_camera_intrinsics_matrix,
                self.raw_distortion_coefficients),
            (self.width + self.horizontal_padding,
             self.height + self.vertical_padding))
        return frame_depth[int(self.vertical_padding /
                               2):-int(self.vertical_padding / 2),
                           int(self.horizontal_padding /
                               2):-int(self.horizontal_padding / 2)]
