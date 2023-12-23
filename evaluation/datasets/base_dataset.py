import os
import cv2
import numpy as np
import open3d as o3d

from abc import abstractmethod

from evaluation.evaluation_utils import load_camera_intrinsics, scale_camera_intrinsics, integrate_mesh


class EvaluationDataset:

    def __init__(self,
                 dir_dataset: str,
                 dataset_name: str,
                 num_evaluation_frames: int,
                 frame_height: int = 0,
                 frame_width: int = 0) -> None:
        self.dir_dataset = dir_dataset
        self.dataset_name = dataset_name
        self.height = frame_height
        self.width = frame_width
        self._load_dataset()
        self.evaluation_frames = range(
            0, self.num_frames, int(self.num_frames / num_evaluation_frames))
        self.num_evaluation_frames = len(self.evaluation_frames)

    def _load_dataset(self) -> None:
        self.files_color, self.files_depth = self._load_files()
        self.camera_intrinsics = scale_camera_intrinsics(
            self._load_camera_intrinsics(),
            height=self.height,
            width=self.width)
        self.camera_extrinsics = self._load_camera_extrinsics()
        self.num_frames = len(self.files_color)

    def _load_camera_intrinsics(self) -> dict:
        if self.height == 0 or self.width == 0:
            self.height, self.width, _ = cv2.imread(self.files_color[0]).shape
        return load_camera_intrinsics(dir_dataset=self.dir_dataset,
                                      dataset_name=self.dataset_name)

    @abstractmethod
    def _load_camera_extrinsics(self) -> list:
        raise NotImplementedError

    @abstractmethod
    def _load_files(self) -> tuple:
        raise NotImplementedError

    def _load_frame_color(self, frame_index: int) -> np.array:
        return cv2.resize(
            cv2.cvtColor(cv2.imread(self.files_color[frame_index]),
                         cv2.COLOR_BGR2RGB),
            (self.camera_intrinsics['width'],
             self.camera_intrinsics['height']))

    def _load_frame_depth(self, frame_index: int) -> np.array:
        return cv2.resize(
            cv2.imread(self.files_depth[frame_index], cv2.IMREAD_ANYDEPTH) /
            self.camera_intrinsics['depth_scale'],
            (self.camera_intrinsics['width'],
             self.camera_intrinsics['height']))

    def _load_frames(self,
                     type: str = 'color',
                     mode: str = 'evaluation_frames',
                     keyframes: list = None) -> list:
        frame_indices = []
        if mode == 'evaluation_frames':
            frame_indices = self.evaluation_frames
        elif mode == 'keyframes':
            frame_indices = keyframes
        elif mode == 'all':
            frame_indices = range(self.num_frames)
        else:
            raise NotImplementedError
        if type == 'color':
            return [
                self._load_frame_color(frame_index)
                for frame_index in frame_indices
            ]
        elif type == 'depth':
            return [
                self._load_frame_depth(frame_index)
                for frame_index in frame_indices
            ]
        else:
            raise NotImplementedError

    def frames_color(self,
                     mode: str = 'evaluation_frames',
                     keyframes: list = None) -> list:
        return self._load_frames(type='color', mode=mode, keyframes=keyframes)

    def frames_depth(self,
                     mode: str = 'evaluation_frames',
                     keyframes: list = None) -> list:
        return self._load_frames(type='depth', mode=mode, keyframes=keyframes)

    def mesh(self) -> tuple:
        file_mesh = os.path.dirname(
            self.dir_dataset
        ) + f'/{os.path.basename(self.dir_dataset)}_mesh_fused.ply'
        if not os.path.exists(file_mesh):
            integrate_mesh(file_mesh=file_mesh,
                           camera_intrinsics=self.camera_intrinsics,
                           camera_extrinsics=self.camera_extrinsics,
                           frames_color=self.frames_color(mode='all'),
                           frames_depth=self.frames_depth(mode='all'))
        return o3d.io.read_triangle_mesh(file_mesh), file_mesh
