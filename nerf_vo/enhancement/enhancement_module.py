import torch

from collections import deque

from nerf_vo.multiprocessing.process_module import ProcessModule


class EnhancementModule(ProcessModule):

    def initialize_module(self) -> None:
        self.process_name = 'enhancement'
        if self.name != 'none':
            from nerf_vo.enhancement.omnidata_estimator import OmnidataEstimator
            self.method = OmnidataEstimator(args=self.args,
                                            device=self.device,
                                            mode=self.name)
            if self.args.tracking_module == 'droid-slam':
                self.seen_keyframes = torch.zeros(self.args.num_keyframes,
                                                  dtype=torch.bool,
                                                  device=self.device)
                self.buffer_frames_depth = torch.zeros(
                    (self.args.num_keyframes, 1, self.args.frame_height,
                     self.args.frame_width),
                    dtype=torch.float32,
                    device=self.device)
                self.buffer_frames_normal = torch.zeros(
                    (self.args.num_keyframes, 3, self.args.frame_height,
                     self.args.frame_width),
                    dtype=torch.float32,
                    device=self.device)
        if self.args.tracking_module == 'dpvo':
            self.buffer_camera_intrinsics = deque(
                maxlen=self.args.removal_window - 2)
            self.buffer_frames_color = deque(maxlen=self.args.removal_window -
                                             2)
            self.buffer_frames_depth = deque(maxlen=self.args.removal_window -
                                             2)

        super().initialize_module()

    def step(self, input: dict) -> tuple:
        super().step(input=None)
        if input is not None:
            input['keyframe_indices'] = input['keyframe_indices'].clone()
            input['camera_intrinsics'] = input['camera_intrinsics'].clone()
            input['camera_extrinsics'] = input['camera_extrinsics'].clone()
            input['frames_color'] = input['frames_color'] / 255.0

            if self.name != 'none':
                if self.args.tracking_module == 'dpvo':
                    frames_depth, frames_normal = self.method(
                        frames_color=input['frames_color'].clone())

                    if frames_depth is None:
                        raise NotImplementedError

                    self.buffer_camera_intrinsics.extend(
                        input['camera_intrinsics'])
                    self.buffer_frames_color.extend(input['frames_color'])
                    self.buffer_frames_depth.extend(frames_depth)
                    if frames_depth.shape[0] <= self.args.removal_window - 2:
                        frames_depth = torch.stack(
                            list(self.buffer_frames_depth))

                    dpvo_patches = input.pop('dpvo_patches')
                    dpvo_patches = self.dpvo_remove_outliers(
                        dpvo_patches=dpvo_patches)
                    dpvo_patches = dpvo_patches[:, :, :, 1, 1]
                    dpvo_patches[:, :, :2] = dpvo_patches[:, :, :2] * 4
                    dpvo_patches[:, :, 2] = 1 / dpvo_patches[:, :, 2]
                    dpvo_patches[:, :, 2] = dpvo_patches[:, :, 2].clip(0, 5)

                    patches_frames_depth = frames_depth[
                        torch.arange(0,
                                     dpvo_patches.shape[0],
                                     dtype=torch.long,
                                     device=self.device).
                        repeat(dpvo_patches.shape[1], 1).t().reshape(-1), 0,
                        dpvo_patches[:, :, 1].reshape(-1).long(),
                        dpvo_patches[:, :, 0].reshape(-1).long()].reshape(
                            dpvo_patches.shape[0], dpvo_patches.shape[1])
                    scale = torch.std(dpvo_patches[:, :, 2, None, None],
                                      dim=[1, 2, 3],
                                      keepdim=True) / torch.std(
                                          patches_frames_depth[:, :, None,
                                                               None],
                                          dim=[1, 2, 3],
                                          keepdim=True)
                    shift = torch.mean(
                        frames_depth, dim=[1, 2, 3], keepdim=True) * (
                            (torch.mean(dpvo_patches[:, :, 2, None, None],
                                        dim=[1, 2, 3],
                                        keepdim=True) /
                             torch.mean(patches_frames_depth[:, :, None, None],
                                        dim=[1, 2, 3],
                                        keepdim=True)) - scale)
                    frames_depth = frames_depth * scale + shift
                    frames_depth = torch.clip(frames_depth, 0, 5)
                else:
                    raise NotImplementedError

                if frames_normal is not None:
                    input['frames_normal'] = torch.nn.functional.normalize(
                        frames_normal * 2.0 - 1.0, p=2, dim=1)
            else:
                if self.args.tracking_module == 'droid-slam':
                    frames_depth = 1 / \
                        input.pop('droid_slam_inverse_depth')[:, None, :, :]
                    frames_depth_covariance = input.pop(
                        'droid_slam_depth_covariance').clone()[:, None, :, :]
                    input['frames_depth_covariance'] = frames_depth_covariance
                else:
                    raise NotImplementedError

            input['frames_depth'] = frames_depth

            if self.args.mapping_module == 'nerfstudio':
                input['camera_extrinsics'][:, :3, 1:3] *= -1

            if input['last_frame']:
                with self.shared_variables['status_lock']:
                    self.shared_variables['status']['tracking'] = 'shutdown'
                self.shutdown = True

            torch.cuda.empty_cache()
            return input, False
        else:
            return None, True

    def dpvo_remove_outliers(self, dpvo_patches: torch.Tensor) -> torch.Tensor:
        dpvo_patches[:, :, 2] += torch.rand(
            (dpvo_patches.shape[0], dpvo_patches.shape[1], 1, 1),
            dtype=torch.float32,
            device=self.device) * 1e-4
        outlier_mask = (dpvo_patches[:, :, 2, 1, 1] < torch.quantile(
            dpvo_patches[:, :, 2, 1, 1], q=1 / 12,
            dim=1)[:, None]) | (dpvo_patches[:, :, 2, 1, 1] > torch.quantile(
                dpvo_patches[:, :, 2, 1, 1], q=11 / 12, dim=1)[:, None])
        try:
            return dpvo_patches[~outlier_mask].reshape(
                dpvo_patches.shape[0], int(dpvo_patches.shape[1] * 5 / 6), 3,
                dpvo_patches.shape[3], dpvo_patches.shape[4])
        except:
            print('outlier mask mismatch')
            dpvo_patches[dpvo_patches < 1e-3] = torch.mean(dpvo_patches)
            return dpvo_patches
