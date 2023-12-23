import os
import cv2
import json
import tqdm
import lpips
import numpy as np
import open3d as o3d
import pandas as pd

from evaluation.datasets.base_dataset import EvaluationDataset
from evaluation.datasets.eth3d_dataset import ETH3DDataset
from evaluation.datasets.replica_dataset import ReplicaDataset
from evaluation.datasets.scannet_dataset import ScanNetDataset
from evaluation.datasets.seven_scenes_dataset import SevenScenesDataset
from evaluation.datasets.tum_rgbd_dataset import TUMRGBDDataset
from evaluation.evaluation_utils import set_logging_prefix, calculate_absolute_trajectory_error, calculate_depth_metrics_2d, calculate_color_metrics_2d, calculate_metrics_3d


class Evaluator:

    def __init__(self, config: str, dataset: EvaluationDataset = None) -> None:
        self.config = config
        self.dir_dataset = config['dir_dataset']
        self.dir_prediction = config['dir_prediction']
        self.dir_result = config['dir_result']

        with open(self.dir_prediction +
                  '/mapping_keyframe2frame.json') as file:
            self.keyframes = json.load(file)

        if dataset is None:
            if config['dataset_name'] == 'eth3d':
                dataset_class = ETH3DDataset
            elif config['dataset_name'] == 'replica':
                dataset_class = ReplicaDataset
            elif config['dataset_name'] == 'tum-rgbd':
                dataset_class = TUMRGBDDataset
            elif config['dataset_name'] == '7-scenes':
                dataset_class = SevenScenesDataset
            elif config['dataset_name'] == 'scannet':
                dataset_class = ScanNetDataset
            else:
                raise NotImplementedError
            self.dataset = dataset_class(
                dir_dataset=self.dir_dataset,
                num_evaluation_frames=config['num_evaluation_frames'],
                frame_height=config['evaluation_frame_height'],
                frame_width=config['evaluation_frame_width'])
        else:
            self.dataset = dataset

        os.makedirs(
            self.dir_result) if not os.path.exists(self.dir_result) else None

    def calculate_metrics_trajectory(self) -> dict:
        camera_extrinsics_keyframes_gt = np.stack([
            self.dataset.camera_extrinsics[frame_index]
            for frame_index in self.keyframes
        ])
        metrics_trajectory = []

        for trajectory in ['keyframes_tracking', 'keyframes_mapping']:
            with open(self.dir_prediction +
                      f'/matrices/matrices_origin2frame_{trajectory}.json'
                      ) as file:
                camera_extrinsics_keyframes_pred = np.array(json.load(file))
            absolute_trajectory_error = calculate_absolute_trajectory_error(
                matrices_origin2frame_gt=camera_extrinsics_keyframes_gt,
                matrices_origin2frame_pred=camera_extrinsics_keyframes_pred,
                with_scale=True)
            absolute_trajectory_error['trajectory'] = trajectory
            metrics_trajectory.append(absolute_trajectory_error)

        metrics_trajectory = pd.DataFrame(metrics_trajectory)
        metrics_trajectory.to_csv(self.dir_result + '/metrics_trajectory.csv',
                                  index=False)

        return metrics_trajectory[[
            'absolute_trajectory_error_rmse', 'absolute_trajectory_error_mean',
            'absolute_trajectory_error_median',
            'absolute_trajectory_error_std', 'absolute_trajectory_error_max',
            'absolute_trajectory_error_min'
        ]].squeeze().to_dict()

    def calculate_metrics_2d(self, mode: str = 'evaluation_frames') -> dict:
        if mode == 'evaluation_frames':
            folder_name = 'evaluation_frames'
        elif mode == 'keyframes':
            folder_name = 'keyframes'
        elif mode == 'all':
            folder_name = 'all_frames'
        else:
            raise NotImplementedError

        frames_color_gt = self.dataset.frames_color(mode=mode,
                                                    keyframes=self.keyframes)
        frames_depth_gt = self.dataset.frames_depth(mode=mode,
                                                    keyframes=self.keyframes)

        files_color_pred = sorted([
            os.path.join(self.dir_prediction + f'/{folder_name}/color', file)
            for file in os.listdir(self.dir_prediction +
                                   f'/{folder_name}/color')
            if file.endswith('.jpg')
        ])
        frames_color_pred = [
            cv2.cvtColor(cv2.imread(file_color), cv2.COLOR_BGR2RGB)
            for file_color in files_color_pred
        ]
        files_depth_pred = sorted([
            os.path.join(self.dir_prediction + f'/{folder_name}/depth', file)
            for file in os.listdir(self.dir_prediction +
                                   f'/{folder_name}/depth')
            if file.endswith('.png')
        ])
        frames_depth_pred = [
            cv2.imread(file_depth, cv2.IMREAD_ANYDEPTH) /
            self.dataset.camera_intrinsics['depth_scale']
            for file_depth in files_depth_pred
        ]

        metrics_2d = []
        lpips_loss = lpips.LPIPS(net='alex')

        for frame_depth_gt, frame_depth_pred, frame_color_gt, frame_color_pred in tqdm.tqdm(
                zip(frames_depth_gt, frames_depth_pred, frames_color_gt,
                    frames_color_pred)):
            temp_metrics_2d = {}
            temp_metrics_2d.update(
                calculate_depth_metrics_2d(frame_depth_gt=frame_depth_gt,
                                           frame_depth_pred=frame_depth_pred))
            temp_metrics_2d.update(
                calculate_color_metrics_2d(frame_color_gt=frame_color_gt,
                                           frame_color_pred=frame_color_pred,
                                           lpips_loss=lpips_loss))
            metrics_2d.append(temp_metrics_2d)

        metrics_2d = pd.DataFrame(metrics_2d)
        metrics_2d.to_csv(self.dir_result + f'/metrics_2d_{folder_name}.csv',
                          index=False)
        metrics_2d = metrics_2d.mean().to_dict()
        with open(self.dir_result + f'/metrics_2d_{folder_name}.json',
                  'w') as file:
            json.dump(metrics_2d, file)

        return metrics_2d

    def calculate_metrics_3d(self) -> dict:
        metrics_3d = []

        mesh_gt, _ = self.dataset.mesh()
        files_mesh_pred = [
            os.path.join(self.dir_prediction + '/mesh', file)
            for file in os.listdir(self.dir_prediction + '/mesh')
            if file.endswith('.ply')
        ]

        for file_mesh_pred in tqdm.tqdm(files_mesh_pred):
            if os.path.splitext(os.path.basename(
                    file_mesh_pred))[0] == 'mesh_from_nerf_raw':
                continue
            mesh_pred = o3d.io.read_triangle_mesh(file_mesh_pred)
            temp_metrics_3d = calculate_metrics_3d(mesh_gt=mesh_gt,
                                                   mesh_pred=mesh_pred)
            temp_metrics_3d['mesh'] = os.path.splitext(
                os.path.basename(file_mesh_pred))[0]
            metrics_3d.append(temp_metrics_3d)

        metrics_3d = pd.DataFrame(metrics_3d)
        metrics_3d.to_csv(self.dir_result + '/metrics_3d.csv', index=False)

        return metrics_3d[[
            'accuracy', 'completion', 'precision', 'recall', 'f1score'
        ]].squeeze().to_dict()
