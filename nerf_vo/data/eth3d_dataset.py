import cv2
import numpy as np

from nerf_vo.data.base_dataset import BaseDataset
from nerf_vo.data.data_utils import read_timestamp_data, associate_timestamp_data


class ETH3DDataset(BaseDataset):

    def _load_files_color(self) -> list:
        color_timestamp_data = read_timestamp_data(
            dir_dataset=self.dir_dataset, mode="color")
        depth_timestamp_data = read_timestamp_data(
            dir_dataset=self.dir_dataset, mode="depth")
        camera_extrinsics_timestamp_data = read_timestamp_data(
            dir_dataset=self.dir_dataset, mode="camera_extrinsics")

        association_color_depth = associate_timestamp_data(
            source_timestamps=list(color_timestamp_data.keys()),
            target_timestamps=list(depth_timestamp_data.keys()),
        )
        association_color_camera_extrinsics = associate_timestamp_data(
            source_timestamps=[
                timestamp_color
                for timestamp_color, _ in association_color_depth
            ],
            target_timestamps=list(camera_extrinsics_timestamp_data.keys()),
        )

        files_color = [
            color_timestamp_data[timestamp_color]
            for timestamp_color in sorted([
                timestamp_color
                for timestamp_color, _ in association_color_camera_extrinsics
            ])
        ]
        return [
            self.dir_dataset + f"/{file_color[0]}"
            for file_color in files_color
        ]

    def _load_camera_intrinsics(self) -> dict:
        camera_intrinsics = {}
        height, width, _ = cv2.imread(self.files_color[0]).shape
        with open(self.dir_dataset + "/calibration.txt", "r") as file:
            calibration = np.array(list(map(float,
                                            file.read().split()))).reshape(4)
        camera_intrinsics["height"] = height
        camera_intrinsics["width"] = width
        camera_intrinsics["fx"] = calibration[0]
        camera_intrinsics["fy"] = calibration[1]
        camera_intrinsics["cx"] = calibration[2]
        camera_intrinsics["cy"] = calibration[3]
        camera_intrinsics["depth_scale"] = 5000.0
        return camera_intrinsics
