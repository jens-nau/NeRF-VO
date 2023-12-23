import os
import sys
import glob
import torch
import numpy as np
import argparse

from nerf_vo.mapping.mapping_utils import step_check

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                 'build/instant_ngp'))

import pyngp

file_instant_ngp_config = 'nerf_vo/thirdparty/nerf_slam/thirdparty/instant-ngp/configs/nerf/base.json'


class InstantNGP:

    def __init__(
        self,
        args: argparse.Namespace,
        device: torch.device = torch.device('cuda:0')
    ) -> None:
        self.args = args
        self.device = device
        self.is_initialized = False
        self.is_shut_down = False

        self.step = 0

        mode = pyngp.TestbedMode.Nerf
        self.ngp = pyngp.Testbed(mode, 0)
        bounding_box = pyngp.BoundingBox(np.array([-np.inf, -np.inf, -np.inf]),
                                         np.array([np.inf, np.inf, np.inf]))
        self.ngp.create_empty_nerf_dataset(n_images=args.num_keyframes,
                                           nerf_scale=1.0,
                                           nerf_offset=np.array(
                                               [0.0, 0.0, 0.0]),
                                           aabb_scale=4,
                                           render_aabb=bounding_box)
        self.ngp.nerf.training.n_images_for_training = 0
        self.ngp.reload_network_from_file(file_instant_ngp_config)

        self.ngp.shall_train = True
        self.ngp.nerf.training.optimize_extrinsics = True
        self.ngp.nerf.training.depth_loss_type = pyngp.LossType.L2

        self.ngp.frame()

    def __call__(self, input: dict) -> None:
        if self.step == self.args.mapping_iterations:
            self.shut_down()
        else:
            if input is not None:
                self.update(input=input)
            if self.is_initialized:
                self.train()

    def update(self, input: dict) -> None:
        camera_extrinsics = input['camera_extrinsics'].clone()
        frames_color = input['frames_color'].clone().permute(0, 2, 3, 1)
        frames_color = torch.where(
            frames_color > 0.04045,
            torch.pow((frames_color + 0.055) / 1.055, 2.4),
            frames_color / 12.92)
        frames_color = torch.cat(
            (frames_color,
             torch.ones((frames_color.shape[0], frames_color.shape[1],
                         frames_color.shape[2], 1),
                        dtype=frames_color.dtype,
                        device=frames_color.device)),
            dim=3)
        frames_depth = input['frames_depth'].clone().permute(0, 2, 3, 1)

        if 'frames_depth_covariance' in input:
            frames_depth_covariance = input['frames_depth_covariance'].clone(
            ).permute(0, 2, 3, 1)
        else:
            frames_depth_covariance = torch.ones(
                (frames_color.shape[0], frames_color.shape[1],
                 frames_color.shape[2], 1),
                dtype=torch.float32,
                device=self.device)

        self.ngp.nerf.training.update_training_images(
            frame_ids=input['keyframe_indices'].contiguous().cpu().numpy(
            ).tolist(),
            poses=list(camera_extrinsics[:, :3].contiguous().cpu().numpy()),
            images=list(frames_color.contiguous().cpu().numpy()),
            depths=list(frames_depth.contiguous().cpu().numpy()),
            depths_cov=list(
                frames_depth_covariance.contiguous().cpu().numpy()),
            resolution=np.array(
                [self.args.frame_width, self.args.frame_height]),
            principal_point=input['camera_intrinsics'][0, 2:].cpu().numpy(),
            focal_length=input['camera_intrinsics'][0, :2].cpu().numpy(),
            depth_scale=1.0,
            depth_cov_scale=1.0)
        self.is_initialized = True
        torch.cuda.empty_cache()

    def train(self) -> None:
        self.ngp.frame()
        if step_check(self.step, self.args.mapping_snapshot_iterations):
            self.save_snapshot()
        self.step += 1

    def shut_down(self) -> None:
        self.save_snapshot()
        self.is_shut_down = True

    def save_snapshot(self) -> None:
        self.ngp.save_snapshot(
            self.args.dir_prediction +
            f'/snapshots/snapshot{self.step:06d}.msgpack', False)
