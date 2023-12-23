import os
import numpy as np

from evaluation.datasets.base_dataset import EvaluationDataset


class ReplicaDataset(EvaluationDataset):

    def __init__(self,
                 dir_dataset: str,
                 num_evaluation_frames: int,
                 frame_height: int = 0,
                 frame_width: int = 0) -> None:
        super().__init__(dir_dataset=dir_dataset,
                         dataset_name='replica',
                         num_evaluation_frames=num_evaluation_frames,
                         frame_height=frame_height,
                         frame_width=frame_width)

    def _load_camera_extrinsics(self) -> list:
        with open(self.dir_dataset + '/traj.txt', "r") as file:
            lines = file.readlines()

        return [
            np.array(list(map(float, line.split()))).reshape(4, 4)
            for line in lines
        ]

    def _load_files(self) -> tuple:
        files_color = sorted([
            os.path.join(self.dir_dataset + '/results', file)
            for file in os.listdir(self.dir_dataset + '/results')
            if file.endswith('.jpg')
        ])
        files_depth = sorted([
            os.path.join(self.dir_dataset + '/results', file)
            for file in os.listdir(self.dir_dataset + '/results')
            if file.endswith('.png')
        ])
        return files_color, files_depth
