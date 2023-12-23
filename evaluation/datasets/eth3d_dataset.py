import cv2
import numpy as np
import scipy

from evaluation.evaluation_utils import read_timestamp_data, associate_timestamp_data
from evaluation.datasets.base_dataset import EvaluationDataset


class ETH3DDataset(EvaluationDataset):

    def __init__(self,
                 dir_dataset: str,
                 num_evaluation_frames: int,
                 frame_height: int = 0,
                 frame_width: int = 0) -> None:
        super().__init__(dir_dataset=dir_dataset,
                         dataset_name='eth3d',
                         num_evaluation_frames=num_evaluation_frames,
                         frame_height=frame_height,
                         frame_width=frame_width)

    def _load_camera_intrinsics(self) -> None:
        camera_intrinsics = {}
        height, width, _ = cv2.imread(self.files_color[0]).shape
        if self.height == 0 or self.width == 0:
            self.height, self.width = height, width
        with open(self.dir_dataset + '/calibration.txt', 'r') as file:
            calibration = np.array(list(map(float,
                                            file.read().split()))).reshape(4)
        camera_intrinsics['height'] = height
        camera_intrinsics['width'] = width
        camera_intrinsics['fx'] = calibration[0]
        camera_intrinsics['fy'] = calibration[1]
        camera_intrinsics['cx'] = calibration[2]
        camera_intrinsics['cy'] = calibration[3]
        camera_intrinsics['depth_scale'] = 5000.0
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
        self.camera_tquads_origin2frame = np.stack([
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

        return files_color, files_depth
