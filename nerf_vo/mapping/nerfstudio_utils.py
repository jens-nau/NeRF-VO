from __future__ import annotations

import os
import sys
import torch
import random
import functools

from typing import Dict, List, Literal, Optional, Tuple, Type
from pathlib import Path
from dataclasses import field, dataclass

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                 'nerf_vo/thirdparty/nerfstudio'))

from nerf_vo.thirdparty.nerfstudio.nerfstudio.cameras import camera_utils
from nerf_vo.thirdparty.nerfstudio.nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerf_vo.thirdparty.nerfstudio.nerfstudio.cameras.cameras import Cameras, CameraType
from nerf_vo.thirdparty.nerfstudio.nerfstudio.cameras.rays import RayBundle, RaySamples
from nerf_vo.thirdparty.nerfstudio.nerfstudio.data.datamanagers.base_datamanager import DataManagerConfig, DataManager
from nerf_vo.thirdparty.nerfstudio.nerfstudio.data.pixel_samplers import PixelSampler, PixelSamplerConfig
from nerf_vo.thirdparty.nerfstudio.nerfstudio.data.scene_box import SceneBox
from nerf_vo.thirdparty.nerfstudio.nerfstudio.field_components.field_heads import FieldHeadNames
from nerf_vo.thirdparty.nerfstudio.nerfstudio.model_components.losses import distortion_loss, interlevel_loss, orientation_loss, pred_normal_loss, scale_gradients_by_distance_squared, ds_nerf_depth_loss, monosdf_normal_loss
from nerf_vo.thirdparty.nerfstudio.nerfstudio.model_components.ray_generators import RayGenerator
from nerf_vo.thirdparty.nerfstudio.nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig, DepthNerfactoModel


class DynamicDataset(torch.utils.data.Dataset):

    def __init__(self,
                 num_frames: int,
                 frame_height: int,
                 frame_width: int,
                 device: torch.device = torch.device('cuda:0'),
                 use_normals: bool = True,
                 dir_prediction: str = None) -> None:
        super().__init__()
        self.device = device
        self.use_normals = use_normals
        self.num_frames = num_frames
        self.num_active_frames = 0
        self.frame_height = frame_height
        self.frame_width = frame_width

        self.normalization_matrix = None

        aabb_scale = 1.0
        self.scene_box = SceneBox(
            aabb=torch.tensor([[-aabb_scale, -aabb_scale, -aabb_scale],
                               [aabb_scale, aabb_scale, aabb_scale]],
                              dtype=torch.float32,
                              device=device))

        self.camera_intrinsics = torch.zeros((self.num_frames, 4),
                                             dtype=torch.float32,
                                             device=device).share_memory_()
        self.camera_extrinsics = torch.tile(
            torch.eye(4, dtype=torch.float32, device=self.device),
            (self.num_frames, 1, 1)).share_memory_()
        self.frames_color = torch.zeros(
            (self.num_frames, frame_height, frame_width, 3),
            dtype=torch.float32,
            device=device).share_memory_()
        self.frames_depth = torch.zeros(
            (self.num_frames, frame_height, frame_width, 1),
            dtype=torch.float32,
            device=device).share_memory_()
        if use_normals:
            self.frames_normal = torch.zeros(
                (self.num_frames, frame_height, frame_width, 3),
                dtype=torch.float32,
                device=device).share_memory_()

        if dir_prediction is not None:
            dataset = torch.load(f'{dir_prediction}/dataset.pt')
            self.num_active_frames = dataset['camera_extrinsics'].shape[0]
            self.camera_intrinsics = dataset['camera_intrinsics'].to(device)
            self.camera_extrinsics[:self.num_active_frames] = dataset[
                'camera_extrinsics'].to(device)
            self.frames_color[:self.num_active_frames] = dataset[
                'frames_color'].to(device)
            self.frames_depth[:self.num_active_frames] = dataset[
                'frames_depth'].to(device)
            if use_normals:
                self.frames_normal[:self.num_active_frames] = dataset[
                    'frames_normal'].to(device)

        self.cameras = Cameras(
            fx=self.camera_intrinsics[:, 0],
            fy=self.camera_intrinsics[:, 1],
            cx=self.camera_intrinsics[:, 2],
            cy=self.camera_intrinsics[:, 3],
            distortion_params=camera_utils.get_distortion_params(
                k1=0,
                k2=0,
                k3=0,
                k4=0,
                p1=0,
                p2=0,
            ),
            height=self.frame_height,
            width=self.frame_width,
            camera_to_worlds=self.camera_extrinsics[:, :3],
            camera_type=CameraType.PERSPECTIVE,
        )
        self.metadata = {}

    def __getitem__(self, frame_index: int) -> dict:
        return self.get_frame(frame_index=frame_index)

    def __len__(self) -> int:
        return self.num_active_frames if self.num_active_frames > 0 else self.num_frames

    def get_frame(self, frame_index: int) -> dict:
        data = {
            'image_idx': frame_index,
            'image': self.frames_color[frame_index],
            'depth_image': self.frames_depth[frame_index],
        }
        if self.use_normals:
            frame_normal = (torch.linalg.solve(
                self.camera_extrinsics[frame_index, :3, :3],
                self.frames_normal[frame_index].permute(2, 0, 1).reshape(
                    3, self.frame_height * self.frame_width)).reshape(
                        3, self.frame_height, self.frame_width).permute(
                            1, 2, 0) + 1) / 2
            data['normal_image'] = frame_normal

        return data

    def get_dataset(self) -> dict:
        data = {
            'image_idx':
            torch.arange(0,
                         self.num_active_frames,
                         dtype=torch.long,
                         device=self.device),
            'image':
            self.frames_color[:self.num_active_frames],
            'depth_image':
            self.frames_depth[:self.num_active_frames],
        }
        if self.use_normals:
            frames_normal = (torch.linalg.solve(
                self.camera_extrinsics[:self.num_active_frames, :3, :3],
                self.frames_normal[:self.num_active_frames].permute(
                    0, 3, 1, 2).reshape(self.num_active_frames, 3,
                                        self.frame_height * self.frame_width)
            ).reshape(self.num_active_frames, 3, self.frame_height,
                      self.frame_width).permute(0, 2, 3, 1) + 1) / 2
            data['normal_image'] = frames_normal

        return data

    def update(self, input: dict) -> None:
        output = self.prepare_update(input=input)
        self.insert_update(input=output)

    def prepare_update(self, input: dict) -> dict:
        assert input['keyframe_indices'].max().item() < self.num_frames

        if input['camera_extrinsics'].shape[0] == input['frames_color'].shape[
                0]:
            indices = input['keyframe_indices']
            num_active_frames = input['keyframe_indices'].max().item() + 1
            camera_intrinsics = input['camera_intrinsics'].detach().clone()
            frames_color = input['frames_color'].detach().clone().permute(
                0, 2, 3, 1)
            if self.use_normals:
                frames_normal = input['frames_normal'].detach().clone(
                ).permute(0, 2, 3, 1)

        else:
            indices = torch.arange(
                self.num_active_frames,
                self.num_active_frames + input['frames_color'].shape[0])
            num_active_frames = self.num_active_frames + \
                input['frames_color'].shape[0]
            camera_intrinsics = input['camera_intrinsics'].detach().clone()
            frames_color = input['frames_color'].detach().clone().permute(
                0, 2, 3, 1)
            if self.use_normals:
                frames_normal = input['frames_normal'].detach().clone(
                ).permute(0, 2, 3, 1)

        camera_extrinsics = input['camera_extrinsics'].detach().clone()
        if self.normalization_matrix is None:
            self.normalization_matrix = torch.linalg.solve(
                camera_extrinsics[0],
                torch.tensor(
                    [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                    dtype=camera_extrinsics.dtype,
                    device=camera_extrinsics.device),
                left=True)
        camera_extrinsics = (self.normalization_matrix
                             @ camera_extrinsics.permute(2, 1, 0)).permute(
                                 2, 1, 0)
        frames_depth = input['frames_depth'].detach().clone().permute(
            0, 2, 3, 1)

        output = {
            'indices': indices,
            'keyframe_indices': input['keyframe_indices'],
            'num_active_frames': num_active_frames,
            'camera_intrinsics': camera_intrinsics,
            'camera_extrinsics': camera_extrinsics,
            'frames_color': frames_color,
            'frames_depth': frames_depth,
        }
        if self.use_normals:
            output['frames_normal'] = frames_normal
        return output

    def insert_update(self, input: dict) -> None:
        self.camera_intrinsics[
            input['indices']] = input['camera_intrinsics'].to(self.device)
        self.camera_extrinsics[input['keyframe_indices'].to(
            self.device)] = input['camera_extrinsics'].to(self.device)
        self.frames_color[input['indices']] = input['frames_color'].to(
            self.device)
        self.frames_depth[
            input['keyframe_indices']] = input['frames_depth'].to(self.device)
        if self.use_normals:
            self.frames_normal[input['indices']] = input['frames_normal'].to(
                self.device)
        self.num_active_frames = input['num_active_frames']

    def save_dataset(self, dir_prediction: str) -> None:
        dataset = {
            'camera_intrinsics': self.camera_intrinsics,
            'camera_extrinsics':
            self.camera_extrinsics[:self.num_active_frames],
            'frames_color': self.frames_color[:self.num_active_frames],
            'frames_depth': self.frames_depth[:self.num_active_frames],
        }
        if self.use_normals:
            dataset['frames_normal'] = self.frames_normal[:self.
                                                          num_active_frames]
        torch.save(dataset, f'{dir_prediction}/dataset.pt')


@dataclass
class DynamicDataManagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: DynamicDataManager)
    train_num_rays_per_batch: int = 4096
    eval_num_rays_per_batch: int = 4096
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    num_frames: int = 128
    frame_height: int = 480
    frame_width: int = 640
    use_normals: bool = True
    dir_prediction: str = None


class DynamicDataManager(DataManager):

    config: DynamicDataManagerConfig

    def __init__(
        self,
        config: DynamicDataManagerConfig,
        device: torch.device = torch.device('cuda:0'),
        test_mode: Literal['test', 'val', 'inference'] = 'test',
        world_size: int = 1,
        local_rank: int = 0,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.test_mode = test_mode

        self.train_dataset = DynamicDataset(
            num_frames=self.config.num_frames,
            frame_height=self.config.frame_height,
            frame_width=self.config.frame_width,
            device=self.device,
            use_normals=self.config.use_normals,
            dir_prediction=self.config.dir_prediction)
        self.eval_dataset = None

        super().__init__()

    def setup_train(self):
        self.train_pixel_sampler = PixelSamplerConfig().setup(
            num_rays_per_batch=self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device))

    def setup_eval(self):
        pass

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        dataset = self.train_dataset.get_dataset()
        batch = self.train_pixel_sampler.sample(dataset)
        ray_indices = batch['indices']
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        return self.next_train(step=step)

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        frame_index = random.randint(0,
                                     self.train_dataset.num_active_frames - 1)
        ray_bundle = self.train_dataset.cameras.generate_rays(
            camera_indices=frame_index, keep_shape=True)
        batch = self.train_dataset[frame_index]
        return frame_index, ray_bundle, batch

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        return Path()

    def get_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        return {}


@dataclass
class ExtendedNerfactoModelConfig(DepthNerfactoModelConfig):

    _target: Type = field(default_factory=lambda: ExtendedNerfactoModel)
    normal_loss_mult: float = 1e-5,


class ExtendedNerfactoModel(DepthNerfactoModel):

    config: ExtendedNerfactoModelConfig

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if 'normal_image' in batch and self.config.normal_loss_mult > 0.0:
            metrics_dict['normal_loss'] = monosdf_normal_loss(
                normal_pred=outputs['normals'],
                normal_gt=batch['normal_image'])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if 'normal_loss' in metrics_dict:
            loss_dict['normal_loss'] = self.config.normal_loss_mult * \
                metrics_dict['normal_loss']
        return loss_dict
