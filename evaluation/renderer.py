import os
import cv2
import json
import tqdm
import numpy as np
import open3d as o3d

from evaluation.datasets.base_dataset import EvaluationDataset
from evaluation.datasets.eth3d_dataset import ETH3DDataset
from evaluation.datasets.replica_dataset import ReplicaDataset
from evaluation.datasets.scannet_dataset import ScanNetDataset
from evaluation.datasets.seven_scenes_dataset import SevenScenesDataset
from evaluation.datasets.tum_rgbd_dataset import TUMRGBDDataset
from evaluation.nerf_renderer import NeRFRenderer, NerfstudioRenderer, NeRFSLAMNGPRenderer
from evaluation.evaluation_utils import integrate_mesh


class Renderer:

    def __init__(self,
                 config: dict,
                 dataset: EvaluationDataset = None,
                 nerf: NeRFRenderer = None) -> None:
        self.config = config
        self.dir_dataset = config['dir_dataset']
        self.dir_prediction = config['dir_prediction']

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

        if nerf is None:
            if config['mapping_module'] == 'instant-ngp':
                nerf = NeRFSLAMNGPRenderer(dir_prediction=self.dir_prediction)
            elif config['mapping_module'] == 'nerfstudio':
                nerf = NerfstudioRenderer(dir_prediction=self.dir_prediction)
            else:
                raise NotImplementedError
        else:
            self.nerf = nerf

        self._calculate_pred2gt_transformation()

    def _process_mode(self, mode: str) -> tuple:
        if mode == 'evaluation_frames':
            folder_name = 'evaluation_frames'
            indices = self.dataset.evaluation_frames
        elif mode == 'keyframes':
            folder_name = 'keyframes'
            indices = self.keyframes
        elif mode == 'all':
            folder_name = 'all_frames'
            indices = range(len(self.dataset.num_frames))
        else:
            raise NotImplementedError
        return folder_name, indices

    def _calculate_pred2gt_transformation(self) -> None:
        depth_scales_pred2gt = []

        frames_depth_gt_keyframes = self.dataset.frames_depth(mode='keyframes', keyframes=self.keyframes)
        for index, frame_depth_gt in tqdm.tqdm(
                enumerate(frames_depth_gt_keyframes)):
            frame_depth_pred = self.nerf.render_frame_depth_from_training_frame(
                camera_intrinsics=self.dataset.camera_intrinsics,
                frame_index=index)
            mask = (frame_depth_gt > 0) * (frame_depth_pred > 0) * \
                (frame_depth_gt < 5) * (frame_depth_pred < 5)
            frame_depth_pred = frame_depth_pred[mask]
            frame_depth_gt = frame_depth_gt[mask]
            depth_scales_pred2gt.append(frame_depth_gt.mean() /
                                        frame_depth_pred.mean())

        depth_scale_pred2gt = np.median(depth_scales_pred2gt)

        matrix_scale_pred2gt = np.diag(
            [depth_scale_pred2gt, depth_scale_pred2gt, depth_scale_pred2gt, 1])
        camera_extrinsics_frame0_gt = self.dataset.camera_extrinsics[0]
        camera_extrinsics_frame0_pred = self.nerf.get_camera_extrinsics(
            frame_index=0)
        matrix_pred2gt = camera_extrinsics_frame0_gt @ np.linalg.inv(
            camera_extrinsics_frame0_pred)
        matrix_pred2gt_scaled = camera_extrinsics_frame0_gt @ matrix_scale_pred2gt @ np.linalg.inv(
            camera_extrinsics_frame0_pred)

        self.pred2gt_transformation = {
            'scale_pred2gt': depth_scale_pred2gt,
            'matrix_pred2gt': matrix_pred2gt,
            'matrix_pred2gt_scaled': matrix_pred2gt_scaled,
        }

    def _render_frame(self, camera_extrinsics: np.array, file_color: str,
                      file_depth: str) -> None:
        raw_color, raw_depth = self.nerf.render_frame(
            camera_intrinsics=self.dataset.camera_intrinsics,
            camera_extrinsics=camera_extrinsics)
        color = cv2.cvtColor(raw_color, cv2.COLOR_BGR2RGB)
        depth = (raw_depth.copy() *
                 self.pred2gt_transformation['scale_pred2gt'] *
                 self.dataset.camera_intrinsics['depth_scale']).astype(
                     np.uint16)
        cv2.imwrite(file_color, color)
        cv2.imwrite(file_depth, depth)

    def _render_mesh_from_frames(self,
                                 mode: str = 'evaluation_frames') -> None:
        folder_name, indices = self._process_mode(mode=mode)

        os.makedirs(self.dir_prediction +
                    '/mesh') if not os.path.exists(self.dir_prediction +
                                                   '/mesh') else None
        if not os.path.exists(self.dir_prediction + f'/{folder_name}'):
            self.render_frames(mode=mode)

        file_mesh = self.dir_prediction + f'/mesh/mesh_from_{mode}.ply'
        camera_extrinsics = np.stack(
            [self.dataset.camera_extrinsics[index] for index in indices])
        files_color = sorted([
            os.path.join(self.dir_prediction + f'/{folder_name}/color', file)
            for file in os.listdir(self.dir_prediction +
                                   f'/{folder_name}/color')
            if file.endswith('.jpg')
        ])
        frames_color = [
            cv2.cvtColor(cv2.imread(file_color), cv2.COLOR_BGR2RGB)
            for file_color in files_color
        ]
        files_depth = sorted([
            os.path.join(self.dir_prediction + f'/{folder_name}/depth', file)
            for file in os.listdir(self.dir_prediction +
                                   f'/{folder_name}/depth')
            if file.endswith('.png')
        ])
        frames_depth = [
            cv2.imread(file_depth, cv2.IMREAD_ANYDEPTH) /
            self.dataset.camera_intrinsics['depth_scale']
            for file_depth in files_depth
        ]
        integrate_mesh(file_mesh=file_mesh,
                       camera_intrinsics=self.dataset.camera_intrinsics,
                       camera_extrinsics=camera_extrinsics,
                       frames_color=frames_color,
                       frames_depth=frames_depth)

    def _render_mesh_from_nerf(self) -> None:
        VOXEL_SIZE = 1 / 64

        os.makedirs(self.dir_prediction +
                    '/mesh') if not os.path.exists(self.dir_prediction +
                                                   '/mesh') else None

        _, file_mesh_gt = self.dataset.mesh()
        pcd_gt = o3d.io.read_point_cloud(file_mesh_gt)
        points_pcd_gt = np.asarray(pcd_gt.points)

        min_max_bounds_pcd_gt = np.concatenate(
            (np.min(points_pcd_gt, axis=0).reshape(
                1, 3), np.max(points_pcd_gt, axis=0).reshape(1, 3)),
            axis=0)
        bound_vertices_pcd_gt = (np.array(
            np.meshgrid(*min_max_bounds_pcd_gt.T)).T).reshape(2**3, 3)
        bound_vertices_pcd_gt = np.concatenate(
            (bound_vertices_pcd_gt, np.ones((8, 1))), axis=1)
        bound_vertices_pcd_gt2pred = np.linalg.inv(
            self.pred2gt_transformation['matrix_pred2gt_scaled']
        ) @ bound_vertices_pcd_gt.T
        resolution = ((np.max(bound_vertices_pcd_gt2pred, axis=1)[:3] -
                       np.min(bound_vertices_pcd_gt2pred, axis=1)[:3]) *
                      self.pred2gt_transformation['scale_pred2gt'] /
                      VOXEL_SIZE).astype(int)
        self.nerf.render_mesh(file_mesh=self.dir_prediction +
                              '/mesh/mesh_from_nerf_raw.ply',
                              resolution=resolution,
                              lower_bound=np.min(bound_vertices_pcd_gt2pred,
                                                 axis=1)[:3],
                              upper_bound=np.max(bound_vertices_pcd_gt2pred,
                                                 axis=1)[:3])

        mesh_pred_raw = o3d.io.read_triangle_mesh(
            self.dir_prediction + '/mesh/mesh_from_nerf_raw.ply')
        mesh_pred_raw.transform(
            self.pred2gt_transformation['matrix_pred2gt_scaled'])
        pred_mesh = mesh_pred_raw
        pred_mesh = mesh_pred_raw.crop(
            o3d.geometry.AxisAlignedBoundingBox(
                np.min(bound_vertices_pcd_gt, axis=0)[:3],
                np.max(bound_vertices_pcd_gt, axis=0)[:3]))
        o3d.io.write_triangle_mesh(
            self.dir_prediction + '/mesh/mesh_from_nerf.ply', pred_mesh)

    def render_camera_extrinsics_keyframes(self) -> None:
        with open(self.dir_prediction +
                  f'/matrices/matrices_origin2frame_keyframes_tracking.json'
                  ) as file:
            camera_extrinsics_keyframes_tracking = np.array(json.load(file))
        camera_extrinsics_keyframes_tracking[:, :3,
                                             3] *= self.pred2gt_transformation[
                                                 'scale_pred2gt']
        with open(
                self.dir_prediction +
                f'/matrices/matrices_origin2frame_keyframes_tracking.json',
                'w') as file:
            json.dump(camera_extrinsics_keyframes_tracking.tolist(), file)

        camera_extrinsics_keyframes_mapping = np.stack([
            self.nerf.get_camera_extrinsics(frame_index=frame_index)
            for frame_index in range(len(self.keyframes))
        ])
        camera_extrinsics_keyframes_mapping[:, :3,
                                            3] *= self.pred2gt_transformation[
                                                'scale_pred2gt']
        with open(
                self.dir_prediction +
                f'/matrices/matrices_origin2frame_keyframes_mapping.json',
                'w') as file:
            json.dump(camera_extrinsics_keyframes_mapping.tolist(), file)

    def render_frames(self, mode: str = 'evaluation_frames') -> None:
        folder_name, indices = self._process_mode(mode=mode)

        os.makedirs(self.dir_prediction + f'/{folder_name}/color'
                    ) if not os.path.exists(self.dir_prediction +
                                            f'/{folder_name}/color') else None
        os.makedirs(self.dir_prediction + f'/{folder_name}/depth'
                    ) if not os.path.exists(self.dir_prediction +
                                            f'/{folder_name}/depth') else None

        camera_extrinsics = np.stack(
            [self.dataset.camera_extrinsics[index] for index in indices])
        camera_extrinsics_gt2pred = self.transform_camera_extrinsics_gt2pred(
            camera_extrinsics=camera_extrinsics,
            pred2gt_transformation=self.pred2gt_transformation)

        for index, camera_extrinsics in tqdm.tqdm(
                zip(indices, camera_extrinsics_gt2pred)):
            file_color = self.dir_prediction + \
                f'/{folder_name}/color/{index:06d}.jpg'
            file_depth = self.dir_prediction + \
                f'/{folder_name}/depth/{index:06d}.png'
            self._render_frame(camera_extrinsics=camera_extrinsics,
                               file_color=file_color,
                               file_depth=file_depth)

    def render_mesh(self,
                    source: str = 'frames',
                    mode: str = 'evaluation_frames') -> None:
        if source == 'frames':
            self._render_mesh_from_frames(mode=mode)
        elif source == 'nerf':
            self._render_mesh_from_nerf()
        else:
            raise NotImplementedError

    @staticmethod
    def transform_camera_extrinsics_gt2pred(
            camera_extrinsics: np.array,
            pred2gt_transformation: dict) -> np.array:
        camera_extrinsics_gt2pred = np.tile(np.eye(4),
                                            (camera_extrinsics.shape[0], 1, 1))
        camera_extrinsics_gt2pred[:, :3, 3] = (
            np.linalg.inv(pred2gt_transformation['matrix_pred2gt_scaled'])
            @ camera_extrinsics.T).T[:, :3, 3]
        camera_extrinsics_gt2pred[:, :3, :3] = (
            np.linalg.inv(pred2gt_transformation['matrix_pred2gt'])
            @ camera_extrinsics.T).T[:, :3, :3]
        return camera_extrinsics_gt2pred

    @staticmethod
    def transform_matrices_pred2gt(camera_extrinsics: np.array,
                                   pred2gt_transformation: dict) -> np.array:
        camera_extrinsics_pred2gt = np.tile(np.eye(4),
                                            (camera_extrinsics.shape[0], 1, 1))
        camera_extrinsics_pred2gt[:, :3, 3] = (
            pred2gt_transformation['matrix_pred2gt_scaled']
            @ camera_extrinsics.T).T[:, :3, 3]
        camera_extrinsics_pred2gt[:, :3, :3] = (
            pred2gt_transformation['matrix_pred2gt']
            @ camera_extrinsics.T).T[:, :3, :3]
        return camera_extrinsics_pred2gt
