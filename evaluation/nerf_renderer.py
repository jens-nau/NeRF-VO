import os
import sys
import json
import glob
import math
import yaml
import numpy as np
import torch
import open3d as o3d

from abc import abstractmethod
from typing import cast
from pathlib import Path

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                 'nerf_vo/thirdparty/nerfstudio'))

from nerf_vo.thirdparty.nerfstudio.nerfstudio.utils.poses import multiply
from nerf_vo.thirdparty.nerfstudio.nerfstudio.utils.eval_utils import eval_load_checkpoint
from nerf_vo.thirdparty.nerfstudio.nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig
from nerf_vo.thirdparty.nerfstudio.nerfstudio.fields.sdf_field import SDFField
from nerf_vo.thirdparty.nerfstudio.nerfstudio.exporter.marching_cubes import generate_mesh_with_multires_marching_cubes
from nerf_vo.thirdparty.nerfstudio.nerfstudio.exporter.exporter_utils import generate_point_cloud
from nerf_vo.thirdparty.nerfstudio.nerfstudio.cameras.cameras import Cameras, CameraType
from nerf_vo.thirdparty.nerfstudio.nerfstudio.cameras import camera_utils

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                 'nerf_vo/build/instant_ngp'))

import pyngp


class NeRFRenderer():

    def __init__(self, mapping_model=None, dir_prediction: str = None) -> None:
        if mapping_model is None:
            self.load_nerf_from_snapshot(dir_prediction=dir_prediction)
        else:
            self.load_nerf_from_mapping_model(mapping_model=mapping_model)

    @abstractmethod
    def load_nerf_from_snapshot(self, dir_prediction: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_nerf_from_mapping_model(self, mapping_model) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_camera_extrinsics(self, frame_index: int) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def render_frame(self, camera_intrinsics: dict,
                     camera_extrinsics: np.array) -> tuple:
        raise NotImplementedError

    def render_frame_color(self, camera_intrinsics: dict,
                           camera_extrinsics: np.array) -> np.array:
        color, _ = self.render_frame(camera_intrinsics=camera_intrinsics,
                                     camera_extrinsics=camera_extrinsics)
        return color

    def render_frame_depth(self, camera_intrinsics: dict,
                           camera_extrinsics: np.array) -> np.array:
        _, depth = self.render_frame(camera_intrinsics=camera_intrinsics,
                                     camera_extrinsics=camera_extrinsics)
        return depth

    def render_frame_color_from_training_frame(self, camera_intrinsics: dict,
                                               frame_index: int) -> np.array:
        return self.render_frame_color(
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=self.get_camera_extrinsics(
                frame_index=frame_index))

    def render_frame_depth_from_training_frame(self, camera_intrinsics: dict,
                                               frame_index: int) -> np.array:
        return self.render_frame_depth(
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=self.get_camera_extrinsics(
                frame_index=frame_index))

    @abstractmethod
    def render_mesh(self, file_mesh: str, resolution: np.array,
                    lower_bound: np.array, upper_bound: np.array) -> None:
        raise NotImplementedError


class NerfstudioRenderer(NeRFRenderer):

    def load_nerf_from_snapshot(self, dir_prediction: str) -> None:
        yaml_files = glob.glob(dir_prediction + '/nerfstudio/**/config.yml',
                               recursive=True)

        if yaml_files:
            file_config = yaml_files[0]
            print(f'Using config: {file_config}')
            self.load_pipeline(file_config=file_config,
                               dir_prediction=dir_prediction)
            with open(dir_prediction +
                      '/matrices/matrices_origin2frame_training.json') as file:
                self.matrices_origin2frame_training = np.array(json.load(file))
        else:
            print(f'Could not find config in {dir_prediction + "/nerfstudio"}')

    def load_nerf_from_mapping_model(self, mapping_model) -> None:
        self.pipeline = mapping_model.trainer.pipeline
        self.matrices_origin2frame_training = np.tile(
            np.eye(4),
            (self.pipeline.datamanager.train_dataset.num_active_frames, 1, 1))
        self.matrices_origin2frame_training[:, :3] = multiply(
            self.pipeline.model.camera_optimizer(
                torch.arange(self.pipeline.datamanager.train_dataset.
                             num_active_frames).to(
                                 self.pipeline.datamanager.device)),
            self.pipeline.datamanager.train_dataset.cameras.
            camera_to_worlds[:self.pipeline.datamanager.train_dataset.
                             num_active_frames].
            to(self.pipeline.datamanager.device)).detach().cpu().numpy()
        self.pipeline.eval()

    def get_camera_extrinsics(self, frame_index: int) -> np.array:
        matrix_origin2frame = self.matrices_origin2frame_training[
            frame_index].copy()
        # Transform from NeRF axis convention (Y-axis points up and Z-axis points against the viewing direction) to standard axis convention (Y-axis points down and Z-axis points in the viewing direction)
        matrix_origin2frame[0:3, 1:3] *= -1
        return matrix_origin2frame

    def render_frame(self, camera_intrinsics: dict,
                     camera_extrinsics: np.array) -> tuple:
        # Transform from standard axis convention (Y-axis points down and Z-axis points in the viewing direction) to NeRF axis convention (Y-axis points up and Z-axis points against the viewing direction)
        camera_extrinsics[0:3, 1:3] *= -1
        cameras = Cameras(
            fx=camera_intrinsics['fx'],
            fy=camera_intrinsics['fy'],
            cx=camera_intrinsics['cx'],
            cy=camera_intrinsics['cy'],
            distortion_params=camera_utils.get_distortion_params(
                k1=0,
                k2=0,
                k3=0,
                k4=0,
                p1=0,
                p2=0,
            ),
            height=camera_intrinsics['height'],
            width=camera_intrinsics['width'],
            camera_to_worlds=torch.Tensor(camera_extrinsics).unsqueeze(0)
            [:, :3],
            camera_type=CameraType.PERSPECTIVE,
        ).to(self.pipeline.device)
        camera_ray_bundle = cameras.generate_rays(camera_indices=0,
                                                  keep_shape=True)
        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(
                camera_ray_bundle)

        color = (outputs['rgb'].cpu().numpy() * 255).astype(np.uint8)
        if isinstance(self.pipeline.model.config, DepthNerfactoModelConfig):
            depth = (outputs['depth'] /
                     camera_ray_bundle.metadata["directions_norm"]
                     ).cpu().numpy()[..., 0]
        else:
            depth = outputs['depth'].cpu().numpy()[..., 0]
        return color, depth

    def render_mesh(self, file_mesh: str, resolution: np.array,
                    lower_bound: np.array, upper_bound: np.array) -> None:
        if hasattr(self.pipeline.model.config, "sdf_field"):
            # NOTE: resolution must be divisible by 512
            resolution = math.ceil(resolution.min() / 512) * 512
            multi_res_mesh = generate_mesh_with_multires_marching_cubes(
                geometry_callable_field=lambda x: cast(
                    SDFField, self.pipeline.model.field).forward_geonetwork(x)
                [:, 0].contiguous(),
                resolution=resolution,
                bounding_box_min=tuple(lower_bound),
                bounding_box_max=tuple(upper_bound),
            )
            multi_res_mesh.export(Path(file_mesh))
        elif self.pipeline.model.config.predict_normals == True:
            NUM_POINTS = 33554432  # NOTE: Does not match default rendering resolution
            NUM_RAYS_PER_BATCH = 32768

            self.pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = NUM_RAYS_PER_BATCH

            pcd = generate_point_cloud(
                pipeline=self.pipeline,
                num_points=NUM_POINTS,
                reorient_normals=True,
                rgb_output_name='rgb',
                depth_output_name='depth',
                normal_output_name='normals',
                use_bounding_box=True,
                bounding_box_min=tuple(lower_bound),
                bounding_box_max=tuple(upper_bound),
            )
            torch.cuda.empty_cache()

            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd=pcd, depth=9)
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            o3d.io.write_triangle_mesh(file_mesh, mesh)
        else:
            raise NotImplementedError

    def load_pipeline(self, file_config: str, dir_prediction: str) -> None:
        config = yaml.load(Path(file_config).read_text(), Loader=yaml.Loader)
        config.load_dir = config.get_checkpoint_dir()
        config.pipeline.datamanager.dir_prediction = dir_prediction
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = config.pipeline.setup(device=device, test_mode='test')
        self.pipeline.eval()
        _, _ = eval_load_checkpoint(config=config, pipeline=self.pipeline)


class InstantNGPRenderer(NeRFRenderer):

    def load_nerf_from_snapshot(self, dir_prediction: str) -> None:
        dir_snapshots = dir_prediction + '/snapshots'
        if os.path.exists(dir_snapshots):
            files_snapshots = [
                os.path.join(dir_snapshots, file)
                for file in sorted(os.listdir(dir_snapshots))
                if '.msgpack' in file
            ]
            if len(files_snapshots) > 0:
                file_snapshot = files_snapshots[-1]

        if file_snapshot is not None:
            print(f'Using snapshot: {file_snapshot}')
        else:
            print(f'Could not find snapshot in {dir_snapshots}')

        self.load_ngp_from_snapshot(file_snapshot=file_snapshot)

    def load_nerf_from_mapping_model(self, mapping_model) -> None:
        self.ngp = mapping_model.ngp

    def get_camera_extrinsics(self, frame_index: int) -> np.array:
        matrix_origin2frame = np.eye(4)
        matrix_origin2frame[:3] = self.ngp.nerf.training.get_camera_extrinsics(
            frame_idx=frame_index)
        # Cycle axes from NGP format to NeRF format XYZ -> YZX
        matrix_origin2frame = matrix_origin2frame[[1, 2, 0, 3]]
        # Transform from NeRF axis convention (Y-axis points up and Z-axis points against the viewing direction) to standard axis convention (Y-axis points down and Z-axis points in the viewing direction)
        matrix_origin2frame[0:3, 1:3] *= -1
        return matrix_origin2frame

    def render_frame(self, camera_intrinsics: dict,
                     camera_extrinsics: np.array) -> tuple:
        color = self.render_frame_color(
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics.copy())
        depth = self.render_frame_depth(
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics.copy())
        return color, depth

    def render_frame_color(self, camera_intrinsics: dict,
                           camera_extrinsics: np.array) -> np.array:
        self._set_rendering_defaults(camera_intrinsics=camera_intrinsics)
        self._set_camera_extrinsics(camera_extrinsics=camera_extrinsics)
        self.ngp.render_mode = pyngp.Shade
        color = self.ngp.render(width=camera_intrinsics['width'],
                                height=camera_intrinsics['height'],
                                spp=1,
                                linear=True)
        color[..., 0:3] = np.divide(color[..., 0:3],
                                    color[..., 3:4],
                                    out=np.zeros_like(color[..., 0:3]),
                                    where=color[..., 3:4] != 0)
        color[...,
              0:3] = np.where(color[..., 0:3] > 0.0031308,
                              1.055 * (color[..., 0:3]**(1.0 / 2.4)) - 0.055,
                              12.92 * color[..., 0:3])
        color = (np.clip(color, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        color = color[:, :, :3]
        return color

    def render_frame_depth(self, camera_intrinsics: dict,
                           camera_extrinsics: np.array) -> np.array:
        self._set_rendering_defaults(camera_intrinsics=camera_intrinsics)
        self._set_camera_extrinsics(camera_extrinsics=camera_extrinsics)
        self.ngp.render_mode = pyngp.Depth
        depth = self.ngp.render(width=camera_intrinsics['width'],
                                height=camera_intrinsics['height'],
                                spp=1,
                                linear=True)[..., 0]
        return depth

    def render_mesh(self, file_mesh: str, resolution: np.array,
                    lower_bound: np.array, upper_bound: np.array) -> None:
        bounding_box = pyngp.BoundingBox(lower_bound, upper_bound)
        self.ngp.compute_and_save_marching_cubes_mesh(file_mesh, resolution,
                                                      bounding_box)

    def _set_rendering_defaults(self, camera_intrinsics: dict) -> None:
        self.ngp.nerf.sharpen = 0.0
        self.ngp.exposure = 0.0
        self.ngp.fov_axis = 0
        self.ngp.fov = (2 * math.atan(0.5 * camera_intrinsics['width'] /
                                      camera_intrinsics['fx'])) * 180 / np.pi
        self.ngp.nerf.render_with_lens_distortion = True
        self.ngp.nerf.render_min_transmittance = 1e-4

    def _set_camera_extrinsics(self, camera_extrinsics: np.array) -> None:
        # Transform from standard axis convention (Y-axis points down and Z-axis points in the viewing direction) to NeRF axis convention (Y-axis points up and Z-axis points against the viewing direction)
        camera_extrinsics[0:3, 1:3] *= -1
        # Cycle axes from NeRF format to NGP format YZX -> XYZ
        camera_extrinsics = camera_extrinsics[[2, 0, 1]]
        self.ngp.set_nerf_camera_matrix(camera_extrinsics)

    def load_ngp_from_snapshot(self, file_snapshot: str) -> None:
        raise NotImplementedError


class NeRFSLAMNGPRenderer(InstantNGPRenderer):

    def __init__(self, mapping_model=None, dir_prediction: str = None) -> None:
        sys.path += [
            os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(
                os.path.abspath(os.path.join(__file__, "../..")) +
                '/nerf_slam', "build_ngp*", "**/*.pyd"),
                                                       recursive=True)
        ]
        sys.path += [
            os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(
                os.path.abspath(os.path.join(__file__, "../..")) +
                '/nerf_slam', "build_ngp*", "**/*.so"),
                                                       recursive=True)
        ]
        import pyngp

        super().__init__(mapping_model=mapping_model,
                         dir_prediction=dir_prediction)

    def load_ngp_from_snapshot(self, file_snapshot: str) -> None:
        self.ngp = pyngp.Testbed(pyngp.TestbedMode.Nerf, 0)
        self.ngp.load_snapshot(path=file_snapshot)
