import os
import sys
import json
import gtsam
import numpy as np
import torch
import argparse
import lietorch
import droid_backends

from icecream import ic
from collections import OrderedDict

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                 'thirdparty/nerf_slam'))

import nerf_vo.thirdparty.nerf_slam.networks.geom.projective_ops as pops

from nerf_vo.thirdparty.nerf_slam.utils.flow_viz import cvx_upsample
from nerf_vo.thirdparty.nerf_slam.networks.modules.corr import CorrBlock, AltCorrBlock
from nerf_vo.thirdparty.nerf_slam.networks.modules.extractor import BasicEncoder
from nerf_vo.thirdparty.nerf_slam.networks.droid_net import UpdateModule

file_droid_slam_weights = 'nerf_vo/build/droid_slam/droid.pth'


def gtsam_pose_to_torch(pose: gtsam.Pose3,
                        device: torch.device = torch.device('cuda:0'),
                        dtype=torch.dtype) -> torch.tensor:
    t = pose.translation()
    q = pose.rotation().toQuaternion()
    return torch.tensor(
        [t[0], t[1], t[2], q.x(), q.y(),
         q.z(), q.w()],
        device=device,
        dtype=dtype)


class DROIDSLAM(torch.nn.Module):

    def __init__(
        self,
        args: argparse.Namespace,
        device: torch.device = torch.device('cuda:0')
    ) -> None:
        super().__init__()
        self.args = args
        self.device = device
        self.is_initialized = False
        self.is_shut_down = False

        self.buffer = args.num_keyframes
        self.compute_covariances = args.compute_covariances
        self.global_ba = args.perform_global_bundle_adjustment

        self.keyframe_warmup = 8
        self.max_age = 25
        self.max_factors = 48
        # NOTE: 2.5 according to NeRF-SLAM paper
        self.motion_filter_thresh = 2.4

        # Distance to consider a keyframe, threshold to create a new keyframe [m]
        self.keyframe_thresh = 4.0
        # Add edges between frames within this distance
        self.frontend_thresh = 16.0
        # frontend optimization window
        self.frontend_window = 25
        # force edges between frames within radius
        self.frontend_radius = 2
        # non-maximal supression of edges
        self.frontend_nms = 1
        # weight for translation / rotation components of flow
        self.beta = 0.3

        self.backend_thresh = 22.0
        self.backend_radius = 2
        self.backend_nms = 3

        # number of iterations for first optimization
        self.iters1 = 4
        # number of iterations for second optimization
        self.iters2 = 2

        # DownSamplingFactor: resolution of the images with respect to the features extracted.
        # 8.0 means that the features are at 1/8th of the original resolution.
        self.dsf = 8

        # Type of correlation computation to use: "volume" or "alt"
        # "volume" takes a lot of memory (but is faster), "alt" takes less memory and should be as fast as volume but it's not
        self.corr_impl = "volume"

        self.feature_net = BasicEncoder(output_dim=128, norm_fn='instance')
        self.context_net = BasicEncoder(output_dim=256, norm_fn='none')
        self.update_net = UpdateModule()

        weights = self.load_weights(file_droid_slam_weights)
        self.load_state_dict(weights)
        self.to(self.device)
        self.eval()

        self.kf_idx = 0
        self.kf_idx_to_f_idx = {}
        self.f_idx_to_kf_idx = {}
        self.last_kf_idx = 0
        self.last_k = None

        self.initial_priors = None
        self.last_state = gtsam.Values()
        self.is_warmed_up = False

        self.world_T_body_t0 = gtsam.Pose3(
            np.array([
                [-7.6942980e-02, -3.1037781e-01, 9.4749427e-01, 8.9643948e-02],
                [
                    -2.8366595e-10, -9.5031142e-01, -3.1130061e-01,
                    4.1829333e-01
                ],
                [9.9703550e-01, -2.3952398e-02, 7.3119797e-02, 4.8306200e-01],
                [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]
            ]))
        self.body_t0_T_world = gtsam_pose_to_torch(
            self.world_T_body_t0.inverse(), self.device, torch.float)
        self.body_T_cam0 = gtsam.Pose3(np.eye(4))
        self.world_T_cam0_t0 = self.world_T_body_t0 * self.body_T_cam0
        self.cam0_t0_T_world = gtsam_pose_to_torch(
            self.world_T_cam0_t0.inverse(), self.device, torch.float)
        self.cam0_T_body = gtsam_pose_to_torch(self.body_T_cam0.inverse(),
                                               self.device, torch.float)

        self.is_initialized = True

    def forward(self, input) -> dict:
        k = input['frame_index']
        input['camera_intrinsics'] = torch.tensor([
            input['camera_intrinsics']['fx'], input['camera_intrinsics']['fy'],
            input['camera_intrinsics']['cx'], input['camera_intrinsics']['cy']
        ],
                                                  dtype=torch.float32,
                                                  device=self.device)
        input['frame_color'] = torch.from_numpy(input['frame_color']).permute(
            2, 0, 1).to(self.device)

        imgs_k = input['frame_color'].clone()[None, None]
        imgs_norm_k = self._normalize_imgs(imgs_k)

        if self.last_k is None:
            ic(k)
            assert k == 0
            assert self.kf_idx == 0
            assert self.last_kf_idx == 0

            self.initialize_buffers(imgs_k.shape[-2:])
            self.cam0_images[self.kf_idx] = input['frame_color'].clone()
            self.cam0_intrinsics[self.kf_idx] = (
                1.0 / self.dsf) * input['camera_intrinsics'].clone()

            self.features_imgs[self.kf_idx] = self.__feature_encoder(
                imgs_norm_k)
            self.contexts_imgs[self.kf_idx], self.cst_contexts_imgs[
                self.kf_idx] = self.__context_encoder(imgs_norm_k)

            self.last_k = k
            self.last_kf_idx = self.kf_idx
            self.kf_idx_to_f_idx[self.kf_idx] = k
            self.f_idx_to_kf_idx[k] = self.kf_idx
            output = self.get_output_packet(batch=input)
            self.kf_idx += 1
            return output

        assert k > 0
        assert self.kf_idx < self.buffer

        current_imgs_features = self.__feature_encoder(imgs_norm_k)
        if not self.has_enough_motion(current_imgs_features):
            if input['last_frame']:
                self.kf_idx_to_f_idx.pop(self.kf_idx, None)
                self.kf_idx -= 1
                print(
                    "Last frame reached, and no new motion: starting GLOBAL BA"
                )
                output = self.terminate(batch=input)
                return output
            return None

        self.cam0_images[self.kf_idx] = input['frame_color'].clone()
        self.cam0_intrinsics[self.kf_idx] = (
            1.0 / self.dsf) * input['camera_intrinsics'].clone()

        self.features_imgs[self.kf_idx] = current_imgs_features
        self.contexts_imgs[self.kf_idx], self.cst_contexts_imgs[
            self.kf_idx] = self.__context_encoder(imgs_norm_k)
        self.kf_idx_to_f_idx[self.kf_idx] = k
        self.f_idx_to_kf_idx[k] = self.kf_idx

        if not self.is_warmed_up:
            if self.kf_idx >= self.keyframe_warmup:
                self.__initialize()
        else:
            if not self.__update():
                self.rm_keyframe(self.kf_idx - 1)
                self.f_idx_to_kf_idx.pop(self.kf_idx_to_f_idx[self.kf_idx - 1])
                self.kf_idx_to_f_idx[self.kf_idx - 1] = k
                self.f_idx_to_kf_idx[k] = self.kf_idx - 1
                if self.kf_idx + 1 >= self.buffer or input['last_frame']:
                    print("Last frame reached: starting GLOBAL BA")
                    self.kf_idx_to_f_idx.pop(self.kf_idx)
                    self.kf_idx -= 1
                    output = self.terminate(batch=input)
                    return output
                return None

        self.last_k = k
        self.last_kf_idx = self.kf_idx

        output = self.get_output_packet(batch=input)

        if self.kf_idx + 1 >= self.buffer or input['last_frame']:
            print("Buffer full or last frame reached: starting GLOBAL BA")
            output = self.terminate(batch=input)
            return output

        self.kf_idx += 1

        return output

    def __initialize(self) -> None:
        assert (self.kf_idx > 4)
        assert (self.kf_idx >= self.keyframe_warmup)

        self.add_neighborhood_factors(kf0=0, kf1=self.kf_idx, radius=3)

        # # NOTE: Depth prior
        # frame_depth_estimate = 1 / torch.nn.functional.interpolate(self.frames_depth_estimate[self.kf_idx][None], size=(self.ht, self.wd), mode='bilinear')[0, 0]
        # frame_depth_estimate = (frame_depth_estimate - torch.mean(frame_depth_estimate)) / torch.std(frame_depth_estimate)
        # frame_depth_estimate = (frame_depth_estimate * torch.std(self.cam0_idepths[:self.kf_idx])) + torch.mean(self.cam0_idepths[:self.kf_idx])
        # frame_depth_estimate = torch.clip(frame_depth_estimate, 0)
        # self.cam0_idepths[self.kf_idx] = frame_depth_estimate

        for _ in range(8):
            self.update(kf0=None, kf1=None, use_inactive=True)

        self.add_proximity_factors(kf0=0,
                                   kf1=0,
                                   rad=2,
                                   nms=2,
                                   thresh=self.frontend_thresh,
                                   remove=False)

        for _ in range(8):
            self.update(kf0=None, kf1=None, use_inactive=True)

        self.cam0_T_world[self.kf_idx +
                          1] = self.cam0_T_world[self.kf_idx].clone()
        self.world_T_body[self.kf_idx +
                          1] = self.world_T_body[self.kf_idx].clone()
        self.cam0_idepths[self.kf_idx +
                          1] = self.cam0_idepths[self.kf_idx - 3:self.kf_idx +
                                                 1].mean()
        self.cam0_depths_cov[self.kf_idx +
                             1] = self.cam0_depths_cov[self.kf_idx -
                                                       3:self.kf_idx +
                                                       1].mean()

        self.is_warmed_up = True

        self.viz_idx[:self.kf_idx + 1] = True

        self.rm_factors(self.ii < (self.keyframe_warmup - 4), store=True)

    def __update(self) -> bool:
        if self.correlation_volumes is not None:
            self.rm_factors(self.age > self.max_age, store=True)

        self.add_proximity_factors(kf0=self.kf_idx - 4,
                                   kf1=max(
                                       self.kf_idx + 1 - self.frontend_window,
                                       0),
                                   rad=self.frontend_radius,
                                   nms=self.frontend_nms,
                                   thresh=self.frontend_thresh,
                                   beta=self.beta,
                                   remove=True)

        # NOTE: Depth prior
        # frame_depth_estimate = 1 / torch.nn.functional.interpolate(
        #     self.frames_depth_estimate[self.kf_idx][None], size=(self.ht, self.wd), mode='bilinear')[0, 0]
        # frame_depth_estimate = (
        #     frame_depth_estimate - torch.mean(frame_depth_estimate)) / torch.std(frame_depth_estimate)
        # frame_depth_estimate = (frame_depth_estimate * torch.std(
        #     self.cam0_idepths[:self.kf_idx])) + torch.mean(self.cam0_idepths[:self.kf_idx])
        # frame_depth_estimate = torch.clip(frame_depth_estimate, 0)
        # self.cam0_idepths[self.kf_idx] = frame_depth_estimate

        for itr in range(self.iters1):
            self.update(kf0=None, kf1=None, use_inactive=True)

        d = self.distance([self.kf_idx - 2], [self.kf_idx - 1],
                          beta=self.beta,
                          bidirectional=True)

        if d.item() < self.keyframe_thresh:
            return False
        else:
            for itr in range(self.iters2):
                self.update(None, None, use_inactive=True)

            next_kf = self.kf_idx + 1
            if next_kf < self.buffer:
                self.cam0_T_world[next_kf] = self.cam0_T_world[self.kf_idx]
                self.world_T_body[next_kf] = self.world_T_body[self.kf_idx]
                self.cam0_idepths[next_kf] = self.cam0_idepths[
                    self.kf_idx].mean()
                self.cam0_depths_cov[next_kf] = self.cam0_depths_cov[
                    self.kf_idx]
            return True

    def terminate(self, batch) -> dict:
        if self.global_ba:
            torch.cuda.empty_cache()
            print("#" * 32)
            self.backend(7)

            torch.cuda.empty_cache()
            print("#" * 32)
            self.backend(12)
        else:
            torch.cuda.empty_cache()

        self.viz_idx[:self.last_kf_idx + 1] = True
        output = self.get_output_packet(batch=batch)

        self.save_mapping_keyframe2frame()
        self.save_matrices_origin2frame_keyframes()

        self.is_shut_down = True

        return output

    @torch.cuda.amp.autocast(enabled=True)
    def update(self,
               kf0=None,
               kf1=None,
               itrs=2,
               use_inactive=False,
               EP=1e-7,
               motion_only=False,
               interpolate_traj=False) -> None:
        with torch.cuda.amp.autocast(enabled=False):
            coords1, mask, (Ji, Jj,
                            Jz) = self.reproject(self.ii,
                                                 self.jj,
                                                 cam_T_body=self.cam0_T_body,
                                                 jacobian=True)
            motion = torch.cat(
                [coords1 - self.coords0, self.gru_estimated_flow - coords1],
                dim=-1)
            motion = motion.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

        corr = self.correlation_volumes(coords1)

        self.gru_hidden_states, flow_delta, gru_estimated_flow_weight, damping, upmask = self.update_net(
            self.gru_hidden_states,
            self.gru_contexts_input,
            corr,
            flow=motion,
            ii=self.ii,
            jj=self.jj)

        if kf0 is None:
            kf0 = max(0, self.ii.min().item())

        with torch.cuda.amp.autocast(enabled=False):
            self.gru_estimated_flow = coords1 + \
                flow_delta.to(dtype=torch.float)
            self.gru_estimated_flow_weight = gru_estimated_flow_weight.to(
                dtype=torch.float)

            self.damping[torch.unique(self.ii)] = damping

            if use_inactive:
                mask = (self.ii_inactive >= kf0 - 3) & (self.jj_inactive
                                                        >= kf0 - 3)
                ii = torch.cat([self.ii_inactive[mask], self.ii], 0)
                jj = torch.cat([self.jj_inactive[mask], self.jj], 0)
                gru_estimated_flow = torch.cat([
                    self.gru_estimated_flow_inactive[:, mask],
                    self.gru_estimated_flow
                ], 1)
                gru_estimated_flow_weight = torch.cat([
                    self.gru_estimated_flow_weight_inactive[:, mask],
                    self.gru_estimated_flow_weight
                ], 1)
            else:
                ii, jj, gru_estimated_flow, gru_estimated_flow_weight = self.ii, self.jj, self.gru_estimated_flow, self.gru_estimated_flow_weight

            if interpolate_traj:
                damping = .2 * \
                    self.damping[torch.unique(
                        torch.cat([self.ii, self.jj], 0))].contiguous() + EP
            else:
                damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP

            gru_estimated_flow = gru_estimated_flow.view(
                -1, self.ht, self.wd, 2).permute(0, 3, 1, 2).contiguous()
            gru_estimated_flow_weight = gru_estimated_flow_weight.view(
                -1, self.ht, self.wd, 2).permute(0, 3, 1, 2).contiguous()

            # # NOTE: Optical flow loss
            # for index in torch.unique(ii):

            #     # NOTE: Reprojection error weighting
            #     target_indices = torch.unique(jj[ii == index])
            #     fx, fy, cx, cy = (self.cam0_intrinsics[0]).unbind(dim=-1)
            #     matrix_camera_intrinsics = torch.tensor(
            #         [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=self.device)
            #     matrix_camera_intrinsics_inverse = torch.tensor(
            #         [[1 / fx, 0, -(cx / fx)], [0, 1 / fy, -(cy / fy)], [0, 0, 1]], dtype=torch.float32, device=self.device)

            #     v_source, u_source = torch.meshgrid(torch.arange(self.ht).to(
            #         self.device).float(), torch.arange(self.wd).to(self.device).float())
            #     pixel_coordinates_source = torch.stack(
            #         (u_source.reshape(-1), v_source.reshape(-1), torch.ones(self.ht * self.wd, dtype=torch.float32, device=self.device)))
            #     homogeneous_coordinates_source = torch.tile((1 / self.cam0_idepths[index].reshape(-1)) * (
            #         matrix_camera_intrinsics_inverse @ pixel_coordinates_source), (target_indices.shape[0], 1, 1))
            #     homogeneous_coordinates_target = ((lietorch.SE3(self.cam0_T_world[target_indices][None, :]).inv() * lietorch.SE3(self.cam0_T_world[index][None, None, :]))[:, :, None] * torch.cat(
            #         (homogeneous_coordinates_source, torch.ones(target_indices.shape[0], 1, self.ht * self.wd, dtype=torch.float32, device=self.device)), dim=1).permute(0, 2, 1)[None, :, :, :])[0, :, :, :3]
            #     pixel_coordinates_target = (matrix_camera_intrinsics[None] @ (homogeneous_coordinates_target / homogeneous_coordinates_target[:, :, 2][:, :, None]).permute(
            #         0, 2, 1)).reshape(target_indices.shape[0], 3, self.ht, self.wd)
            #     reprojection_error = torch.abs(
            #         pixel_coordinates_target[:, :2, :, :] - gru_estimated_flow[ii == index])

            #     # NOTE: Next frame sampling
            #     self.confidence_map[index] = torch.sum(
            #         gru_estimated_flow_weight[ii == index], dim=1)[0]
            #     self.correspondence_indices[index] = torch.ones(
            #         (self.ht, self.wd), dtype=torch.long, device=self.device) * jj[ii == index][0]
            #     self.correspondence_field[index] = gru_estimated_flow[ii == index].permute(0, 2, 3, 1)[
            #         0]
            #     self.reprojection_error[index] = reprojection_error.permute(0, 2, 3, 1)[
            #         0]

            #     # # NOTE: Multi frame sampling
            #     # confidence_maps = torch.sum(gru_estimated_flow_weight[ii == index], dim=1)
            #     # argmax_confidence_map = torch.argmax(confidence_maps, dim=0)
            #     # confidence_map = confidence_maps[argmax_confidence_map, torch.arange(argmax_confidence_map.shape[0])[:, None], torch.arange(argmax_confidence_map.shape[1])[None, :]]
            #     # self.confidence_map[index] = confidence_map
            #     # self.correspondence_indices[index] = jj[ii == index][argmax_confidence_map]
            #     # self.correspondence_field[index] = gru_estimated_flow[ii == index].permute(0, 2, 3, 1)[argmax_confidence_map, torch.arange(argmax_confidence_map.shape[0])[:, None], torch.arange(argmax_confidence_map.shape[1])[None, :]]
            #     # self.reprojection_error[index] = reprojection_error.permute(0, 2, 3, 1)[argmax_confidence_map, torch.arange(argmax_confidence_map.shape[0])[:, None], torch.arange(argmax_confidence_map.shape[1])[None, :]]

            #     # # NOTE: Multi frame sampling w/ discount rate
            #     # self.confidence_map *= 0.9995
            #     # confidence_maps = torch.sum(gru_estimated_flow_weight[ii == index], dim=1)
            #     # argmax_confidence_map = torch.argmax(confidence_maps, dim=0)
            #     # confidence_map = confidence_maps[argmax_confidence_map, torch.arange(argmax_confidence_map.shape[0])[:, None], torch.arange(argmax_confidence_map.shape[1])[None, :]]
            #     # update_mask = torch.where(confidence_map > self.confidence_map[index], torch.ones_like(confidence_map, dtype=torch.bool), torch.zeros_like(confidence_map, dtype=torch.bool))
            #     # self.confidence_map[index][update_mask] = confidence_map[update_mask]
            #     # self.correspondence_indices[index][update_mask] = jj[ii == index][argmax_confidence_map][update_mask]
            #     # self.correspondence_field[index][update_mask] = gru_estimated_flow[ii == index].permute(0, 2, 3, 1)[argmax_confidence_map, torch.arange(argmax_confidence_map.shape[0])[:, None], torch.arange(argmax_confidence_map.shape[1])[None, :]][update_mask]
            #     # self.reprojection_error[index][update_mask] = reprojection_error.permute(0, 2, 3, 1)[argmax_confidence_map, torch.arange(argmax_confidence_map.shape[0])[:, None], torch.arange(argmax_confidence_map.shape[1])[None, :]][update_mask]

            # # NOTE: Confidence masking
            # confidence_mask = torch.where(self.confidence_map > 1.5, torch.ones_like(
            #     self.confidence_map, dtype=torch.bool), torch.zeros_like(self.confidence_map, dtype=torch.bool))
            # self.correspondence_indices[confidence_mask == False] = -1

            self.ba(gru_estimated_flow,
                    gru_estimated_flow_weight,
                    damping,
                    ii,
                    jj,
                    kf0,
                    kf1,
                    itrs=itrs,
                    lm=1e-4,
                    ep=0.1,
                    motion_only=motion_only,
                    compute_covariances=self.compute_covariances)

            kx = torch.unique(self.ii)
            self.cam0_idepths_up[kx] = cvx_upsample(
                self.cam0_idepths[kx].unsqueeze(-1), upmask).squeeze()
            self.cam0_depths_cov_up[kx] = cvx_upsample(
                self.cam0_depths_cov[kx].unsqueeze(-1), upmask,
                pow=1.0).squeeze()

            kf1 = max(ii.max().item(), jj.max().item())
            assert kf1 == self.kf_idx
            self.viz_idx[kf0:self.kf_idx + 1] = True

        self.age += 1

    @torch.cuda.amp.autocast(enabled=False)
    def update_lowmem(self,
                      t0=None,
                      t1=None,
                      itrs=2,
                      use_inactive=False,
                      EP=1e-7,
                      steps=8) -> None:
        kfs, cameras, ch, ht, wd = self.features_imgs.shape
        corr_op = AltCorrBlock(
            self.features_imgs.view(1, kfs * cameras, ch, ht, wd))

        for step in range(steps):
            print(f"Global BA Iteration #{step}/{steps}")
            with torch.cuda.amp.autocast(enabled=False):
                coords1, mask, _ = self.reproject(self.ii, self.jj)
                motion = torch.cat([
                    coords1 - self.coords0, self.gru_estimated_flow - coords1
                ],
                                   dim=-1)
                motion = motion.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

            s = 8
            for i in range(0, self.jj.max() + 1, s):
                print(f"ConvGRU Iteration #{i/s}/{(self.jj.max() + 1)/s}")
                v = (self.ii >= i) & (self.ii < i + s)
                iis = self.ii[v]
                jjs = self.jj[v]

                corr = corr_op(coords1[:, v], cameras * iis,
                               cameras * jjs + (iis == jjs).long())

                with torch.cuda.amp.autocast(enabled=True):
                    gru_hidden_states, flow_delta, gru_estimated_flow_weight, damping, upmask = \
                        self.update_net(
                            self.gru_hidden_states[:, v], self.gru_contexts_input[:, iis], corr, motion[:, v], iis, jjs)

                kx = torch.unique(iis)
                all_kf_ids = torch.unique(iis)

                self.gru_hidden_states[:, v] = gru_hidden_states
                self.gru_estimated_flow[:,
                                        v] = coords1[:,
                                                     v] + flow_delta.float()
                self.gru_estimated_flow_weight[:,
                                               v] = gru_estimated_flow_weight.float(
                                               )
                self.damping[all_kf_ids] = damping

                self.cam0_idepths_up[all_kf_ids] = cvx_upsample(
                    self.cam0_idepths[all_kf_ids].unsqueeze(-1),
                    upmask).squeeze()
                self.cam0_depths_cov_up[all_kf_ids] = cvx_upsample(
                    self.cam0_depths_cov[all_kf_ids].unsqueeze(-1),
                    upmask,
                    pow=1.0).squeeze()

            damping = .2 * \
                self.damping[torch.unique(self.ii)].contiguous() + EP
            gru_estimated_flow = self.gru_estimated_flow.view(
                -1, ht, wd, 2).permute(0, 3, 1, 2).contiguous()
            gru_estimated_flow_weight = self.gru_estimated_flow_weight.view(
                -1, ht, wd, 2).permute(0, 3, 1, 2).contiguous()

            ic("Global BA!")

            self.ba(gru_estimated_flow,
                    gru_estimated_flow_weight,
                    damping,
                    self.ii,
                    self.jj,
                    kf0=0,
                    kf1=None,
                    itrs=itrs,
                    lm=1e-5,
                    ep=1e-2,
                    motion_only=False,
                    compute_covariances=False)

    def ba(self,
           gru_estimated_flow,
           gru_estimated_flow_weight,
           damping,
           ii,
           jj,
           kf0=0,
           kf1=None,
           itrs=2,
           lm=1e-4,
           ep=0.1,
           motion_only=False,
           compute_covariances=True) -> None:
        if kf1 is None:
            kf1 = max(ii.max().item(), jj.max().item()) + 1

        N = kf1 - kf0
        HW = self.ht * self.wd
        kx = torch.unique(ii)

        kf_ids = [i + kf0 for i in range(kf1 - kf0)]
        f_ids = [self.kf_idx_to_f_idx[kf_id] for kf_id in kf_ids]

        Xii = np.array([gtsam.symbol_shorthand.X(f_id) for f_id in f_ids])

        initial_priors = None
        if f_ids[0] == 0:
            if self.world_T_cam0_t0:
                _, initial_priors = self.get_gt_priors_and_values(
                    kf_ids[0], f_ids[0])
            else:
                raise (
                    "You need to add initial prior, or you'll have ill-cond hessian!"
                )

        for _ in range(itrs):
            x0 = gtsam.Values()
            linear_factor_graph = gtsam.GaussianFactorGraph()

            for i in range(N):
                kf_id = i + kf0
                x0.insert(
                    Xii[i],
                    gtsam.Pose3(
                        lietorch.SE3(
                            self.world_T_body[kf_id]).matrix().cpu().numpy()))

            H, v, Q, E, w = droid_backends.reduced_camera_matrix(
                self.cam0_T_world, self.world_T_body, self.cam0_idepths,
                self.cam0_intrinsics[0], self.cam0_T_body,
                torch.zeros_like(self.cam0_idepths), gru_estimated_flow,
                gru_estimated_flow_weight, damping, ii, jj, kf0, kf1)

            vision_factors = gtsam.GaussianFactorGraph()
            H = torch.nn.functional.unfold(H[None, None], (6, 6),
                                           stride=6).permute(2, 0, 1).view(
                                               N, N, 6, 6)
            v = torch.nn.functional.unfold(v[None, None], (6, 1),
                                           stride=6).permute(2, 0,
                                                             1).view(N, 6)
            H[range(N), range(N)] /= N
            v[:] /= N
            upper_triangular_indices = torch.triu_indices(N, N)
            for i, j in zip(upper_triangular_indices[0],
                            upper_triangular_indices[1]):
                if i == j:
                    vision_factors.add(
                        gtsam.HessianFactor(Xii[i], H[i, i].cpu().numpy(),
                                            v[i].cpu().numpy(), 0.0))
                else:
                    vision_factors.add(
                        gtsam.HessianFactor(Xii[i], Xii[j], H[i,
                                                              i].cpu().numpy(),
                                            H[i, j].cpu().numpy(),
                                            v[i].cpu().numpy(),
                                            H[j, j].cpu().numpy(),
                                            v[j].cpu().numpy(), 0.0))
            linear_factor_graph.push_back(vision_factors)

            if initial_priors is not None:
                linear_factor_graph.push_back(initial_priors.linearize(x0))

            gtsam_delta = linear_factor_graph.optimizeDensely()
            self.last_state = x0.retract(gtsam_delta)

            poses = gtsam.utilities.allPose3s(self.last_state)
            pose_keys = poses.keys()
            for i, key in enumerate(pose_keys):
                f_idx = gtsam.Symbol(key).index()
                kf_idx = self.f_idx_to_kf_idx[f_idx]
                self.world_T_body[kf_idx] = gtsam_pose_to_torch(
                    poses.atPose3(key), device=self.device, dtype=torch.float)

            self.cam0_T_world[:] = (
                lietorch.SE3(self.cam0_T_body[None]) *
                lietorch.SE3(self.world_T_body).inv()).vec()
            xi_delta = torch.as_tensor(gtsam_delta.vector(pose_keys),
                                       device=self.device,
                                       dtype=torch.float).view(-1, 6)
            droid_backends.solve_depth(xi_delta, self.cam0_idepths, Q, E, w,
                                       ii, jj, kf0, kf1)
            self.cam0_idepths.clamp_(min=0.001)

        if compute_covariances:
            H, v = linear_factor_graph.hessian()
            L = None
            try:
                L = torch.linalg.cholesky(
                    torch.as_tensor(H, device=self.device, dtype=torch.float))
            except Exception as e:
                print(e)
            if L is not None:
                identity = torch.eye(L.shape[0], device=L.device)
                L_inv = torch.linalg.solve_triangular(L, identity, upper=False)
                if torch.isnan(L_inv).any():
                    print("NANs in L_inv!!")
                    raise

                P = N
                D = L.shape[0] // P
                assert D == 6

                Ei = E[:P]
                Ejz = E[P:P + ii.shape[0]]
                M = Ejz.shape[0]
                assert M == ii.shape[0]
                kx, kk = torch.unique(ii, return_inverse=True)
                K = kx.shape[0]

                min_ii_jj = min(ii.min(), jj.min())

                Ej = torch.zeros(K, K, D, HW, device=self.device)
                Ej[jj - min_ii_jj, ii - min_ii_jj] = Ejz
                Ej = Ej[kf0 - min_ii_jj:kf1 - min_ii_jj].view(P, K, D, HW)
                Ej[range(P),
                   kf0 - min_ii_jj:kf1 - min_ii_jj, :, :] = Ei[range(P), :, :]

                E_sum = Ej
                E_sum = E_sum.view(P, K, D, HW)
                E_sum = E_sum.permute(0, 2, 1, 3).reshape(P * D, K * HW)
                Q_ = Q.view(K * HW, 1)
                F = torch.matmul(Q_ * E_sum.t(), L_inv)
                F2 = torch.pow(F, 2)
                delta_cov = F2.sum(dim=-1)
                z_cov = Q_.squeeze() + delta_cov
                z_cov = z_cov.view(K, self.ht, self.wd)

                for i, key in enumerate(pose_keys):
                    f_idx = gtsam.Symbol(key).index()
                    kf_idx = self.f_idx_to_kf_idx[f_idx]

                depth_cov = z_cov / self.cam0_idepths[kx]**4
                self.cam0_depths_cov[kx] = depth_cov

    def backend(self, steps=12) -> None:
        self.max_factors = 16 * self.kf_idx
        self.corr_impl = "alt"
        self.use_uncertainty = False

        self.ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.age = torch.as_tensor([], dtype=torch.long, device=self.device)

        self.correlation_volumes = None
        self.gru_hidden_states = None
        self.gru_contexts_input = None
        self.damping = 1e-6 * torch.ones_like(self.cam0_idepths)

        self.gru_estimated_flow = torch.zeros([1, 0, self.ht, self.wd, 2],
                                              device=self.device,
                                              dtype=torch.float)
        self.gru_estimated_flow_weight = torch.zeros(
            [1, 0, self.ht, self.wd, 2], device=self.device, dtype=torch.float)

        self.ii_inactive = torch.as_tensor([],
                                           dtype=torch.long,
                                           device=self.device)
        self.jj_inactive = torch.as_tensor([],
                                           dtype=torch.long,
                                           device=self.device)
        self.ii_bad = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj_bad = torch.as_tensor([], dtype=torch.long, device=self.device)

        self.gru_estimated_flow_inactive = torch.zeros(
            [1, 0, self.ht, self.wd, 2], device=self.device, dtype=torch.float)
        self.gru_estimated_flow_weight_inactive = torch.zeros(
            [1, 0, self.ht, self.wd, 2], device=self.device, dtype=torch.float)

        self.add_proximity_factors(rad=self.backend_radius,
                                   nms=self.backend_nms,
                                   thresh=self.backend_thresh,
                                   beta=self.beta)

        self.update_lowmem(steps=steps)
        self.clear_edges()
        self.viz_idx[:self.kf_idx] = True

    def initialize_buffers(self, image_size) -> None:
        self.img_height = h = image_size[0]
        self.img_width = w = image_size[1]

        self.coords0 = pops.coords_grid(h // self.dsf,
                                        w // self.dsf,
                                        device=self.device)
        self.ht, self.wd = self.coords0.shape[:2]

        self.cam0_images = torch.zeros(self.buffer,
                                       3,
                                       h,
                                       w,
                                       dtype=torch.uint8,
                                       device=self.device).share_memory_()

        self.cam0_intrinsics = torch.zeros(self.buffer,
                                           4,
                                           dtype=torch.float,
                                           device=self.device).share_memory_()
        self.cam0_T_world = torch.zeros(self.buffer,
                                        7,
                                        dtype=torch.float,
                                        device=self.device).share_memory_()
        self.world_T_body = torch.zeros(self.buffer,
                                        7,
                                        dtype=torch.float,
                                        device=self.device)
        self.cam0_idepths = torch.ones(self.buffer,
                                       h // self.dsf,
                                       w // self.dsf,
                                       dtype=torch.float,
                                       device=self.device)
        self.cam0_depths_cov = torch.ones(self.buffer,
                                          h // self.dsf,
                                          w // self.dsf,
                                          dtype=torch.float,
                                          device=self.device)

        self.cam0_idepths_up = torch.zeros(self.buffer,
                                           h,
                                           w,
                                           dtype=torch.float,
                                           device=self.device).share_memory_()
        self.cam0_depths_cov_up = torch.ones(
            self.buffer, h, w, dtype=torch.float,
            device=self.device).share_memory_()

        # self.frames_normal_estimate = torch.zeros(
        #     self.buffer, 3, h, w, dtype=torch.uint8, device=self.device).share_memory_()
        # self.frames_depth_estimate = torch.zeros(
        #     self.buffer, 1, h, w, dtype=torch.float32, device=self.device).share_memory_()
        # self.confidence_map = torch.zeros(
        #     self.buffer, h // self.dsf, w // self.dsf, dtype=torch.float32, device=self.device).share_memory_()
        # self.correspondence_indices = torch.zeros(
        #     self.buffer, h // self.dsf, w // self.dsf, dtype=torch.long, device=self.device).share_memory_()
        # self.correspondence_field = torch.zeros(
        #     self.buffer, h // self.dsf, w // self.dsf, 2, dtype=torch.float32, device=self.device).share_memory_()
        # self.reprojection_error = torch.zeros(
        #     self.buffer, h // self.dsf, w // self.dsf, 2, dtype=torch.float32, device=self.device).share_memory_()

        self.cam0_T_world[:] = self.cam0_t0_T_world
        self.world_T_body[:] = gtsam_pose_to_torch(self.world_T_body_t0,
                                                   device=self.device,
                                                   dtype=torch.float)

        self.features_imgs = torch.zeros(self.buffer,
                                         1,
                                         128,
                                         h // self.dsf,
                                         w // self.dsf,
                                         dtype=torch.half,
                                         device=self.device)
        self.contexts_imgs = torch.zeros(self.buffer,
                                         1,
                                         128,
                                         h // self.dsf,
                                         w // self.dsf,
                                         dtype=torch.half,
                                         device=self.device)
        self.cst_contexts_imgs = torch.zeros(self.buffer,
                                             1,
                                             128,
                                             h // self.dsf,
                                             w // self.dsf,
                                             dtype=torch.half,
                                             device=self.device)

        self.correlation_volumes = None
        self.gru_hidden_states = None
        self.gru_contexts_input = None
        self.gru_estimated_flow = torch.zeros(
            [1, 0, h // self.dsf, w // self.dsf, 2],
            device=self.device,
            dtype=torch.float)
        self.gru_estimated_flow_weight = torch.zeros(
            [1, 0, h // self.dsf, w // self.dsf, 2],
            device=self.device,
            dtype=torch.float)
        self.damping = 1e-6 * torch.ones_like(self.cam0_idepths)

        self.ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.age = torch.as_tensor([], dtype=torch.long, device=self.device)

        self.ii_inactive = torch.as_tensor([],
                                           dtype=torch.long,
                                           device=self.device)
        self.jj_inactive = torch.as_tensor([],
                                           dtype=torch.long,
                                           device=self.device)
        self.ii_bad = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj_bad = torch.as_tensor([], dtype=torch.long, device=self.device)

        self.gru_estimated_flow_inactive = torch.zeros(
            [1, 0, h // self.dsf, w // self.dsf, 2],
            device=self.device,
            dtype=torch.float)
        self.gru_estimated_flow_weight_inactive = torch.zeros(
            [1, 0, h // self.dsf, w // self.dsf, 2],
            device=self.device,
            dtype=torch.float)

        self.viz_idx = torch.zeros(self.buffer,
                                   device=self.device,
                                   dtype=torch.bool)

    def get_output_packet(self, batch: dict) -> dict:
        frame_indices, = torch.where(self.viz_idx)
        output_packet = None

        output_device = self.device

        if len(frame_indices) != 0:
            # ic('DROID-SLAM - keyframe id:', frame_indices.max().item())
            camera_intrinsics = torch.index_select(self.cam0_intrinsics, 0,
                                                   frame_indices) * self.dsf
            camera_extrinsics = lietorch.SE3(
                torch.index_select(self.cam0_T_world, 0,
                                   frame_indices)).inv().matrix()
            frames_color = torch.index_select(self.cam0_images, 0,
                                              frame_indices)
            frames_inverse_depth = torch.index_select(self.cam0_idepths_up, 0,
                                                      frame_indices)
            frames_depth_covariance = torch.index_select(
                self.cam0_depths_cov_up, 0, frame_indices)
            # frames_normal_estimate = torch.index_select(
            #     self.frames_normal_estimate, 0, frame_indices)
            # frames_depth_estimate = torch.index_select(
            #     self.frames_depth_estimate, 0, frame_indices)
            # confidence_map = torch.index_select(
            #     self.confidence_map, 0, frame_indices)
            # correspondence_indices = torch.index_select(
            #     self.correspondence_indices, 0, frame_indices)
            # correspondence_field = torch.index_select(
            #     self.correspondence_field, 0, frame_indices)
            # reprojection_error = torch.index_select(
            #     self.reprojection_error, 0, frame_indices)

            output_packet = {
                'keyframe_indices':
                frame_indices.to(device=output_device),
                'camera_intrinsics':
                camera_intrinsics.to(device=output_device),
                'camera_extrinsics':
                camera_extrinsics.to(device=output_device),
                'frames_color':
                frames_color.to(device=output_device),
                'droid_slam_inverse_depth':
                frames_inverse_depth.to(device=output_device),
                'droid_slam_depth_covariance':
                frames_depth_covariance.to(device=output_device),
                # 'depth_covariances': depth_covariances.to(device=output_device),
                # 'frames_normal_estimate': frames_normal_estimate.to(device=output_device),
                # 'frames_depth_estimate': frames_depth_estimate.to(device=output_device),
                # 'confidence_map': confidence_map.to(device=output_device),
                # 'correspondence_indices': correspondence_indices.to(device=output_device),
                # 'correspondence_field': correspondence_field.to(device=output_device),
                # 'reprojection_error': reprojection_error,
                # 'mapping_keyframe2frame': self.kf_idx_to_f_idx,
                'last_frame':
                batch['last_frame'],
            }

            self.viz_idx[:] = False
        else:
            if batch['last_frame']:
                output_packet = {'last_frame': True}

        torch.cuda.empty_cache()

        return output_packet

    @torch.cuda.amp.autocast(enabled=True)
    def rm_keyframe(self, kf_idx) -> None:
        self.cam0_images[kf_idx] = self.cam0_images[kf_idx + 1]
        self.cam0_T_world[kf_idx] = self.cam0_T_world[kf_idx + 1]
        self.world_T_body[kf_idx] = self.world_T_body[kf_idx + 1]
        self.cam0_idepths[kf_idx] = self.cam0_idepths[kf_idx + 1]
        self.cam0_depths_cov[kf_idx] = self.cam0_depths_cov[kf_idx + 1]
        self.cam0_intrinsics[kf_idx] = self.cam0_intrinsics[kf_idx + 1]
        # self.frames_normal_estimate[kf_idx] = self.frames_normal_estimate[kf_idx+1]
        # self.frames_depth_estimate[kf_idx] = self.frames_depth_estimate[kf_idx+1]
        self.features_imgs[kf_idx] = self.features_imgs[kf_idx + 1]
        self.contexts_imgs[kf_idx] = self.contexts_imgs[kf_idx + 1]
        self.cst_contexts_imgs[kf_idx] = self.cst_contexts_imgs[kf_idx + 1]

        mask = (self.ii_inactive == kf_idx) | (self.jj_inactive == kf_idx)

        self.ii_inactive[self.ii_inactive >= kf_idx] -= 1
        self.jj_inactive[self.jj_inactive >= kf_idx] -= 1

        if torch.any(mask):
            self.ii_inactive = self.ii_inactive[~mask]
            self.jj_inactive = self.jj_inactive[~mask]
            self.gru_estimated_flow_inactive = self.gru_estimated_flow_inactive[:,
                                                                                ~mask]
            self.gru_estimated_flow_weight_inactive = self.gru_estimated_flow_weight_inactive[:,
                                                                                              ~mask]

        mask = (self.ii == kf_idx) | (self.jj == kf_idx)

        self.ii[self.ii >= kf_idx] -= 1
        self.jj[self.jj >= kf_idx] -= 1

        self.rm_factors(mask, store=False)

    def add_neighborhood_factors(self, kf0, kf1, radius=3) -> None:
        ii, jj = torch.meshgrid(torch.arange(kf0, kf1 + 1),
                                torch.arange(kf0, kf1 + 1))
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        distances = torch.abs(ii - jj)
        keep_radius = distances <= radius
        keep_stereo = distances > 0
        keep = keep_stereo & keep_radius

        self.add_factors(ii[keep], jj[keep])

    def add_proximity_factors(self,
                              kf0=0,
                              kf1=0,
                              rad=2,
                              nms=2,
                              beta=0.25,
                              thresh=16.0,
                              remove=False) -> None:
        t = self.kf_idx + 1
        ix = torch.arange(kf0, t)
        jx = torch.arange(kf1, t)

        ii, jj = torch.meshgrid(ix, jx)
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        d = self.distance(ii, jj, beta=beta)
        d[(ii - rad) < jj] = np.inf
        d[d > 100] = np.inf

        ii1 = torch.cat([self.ii, self.ii_bad, self.ii_inactive], 0)
        jj1 = torch.cat([self.jj, self.jj_bad, self.jj_inactive], 0)
        for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
            for di in range(-nms, nms + 1):
                for dj in range(-nms, nms + 1):
                    if abs(di) + abs(dj) <= max(min(abs(i - j) - 2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (kf0 <= i1 < t) and (kf1 <= j1 < t):
                            d[(i1 - kf0) * (t - kf1) + (j1 - kf1)] = np.inf

        es = []
        for i in range(kf0, t):
            for j in range(max(i - rad - 1, 0), i):
                es.append((i, j))
                es.append((j, i))
                d[(i - kf0) * (t - kf1) + (j - kf1)] = np.inf

        ix = torch.argsort(d)
        for k in ix:
            if d[k].item() > thresh:
                continue

            if len(es) > self.max_factors:
                break

            i = ii[k]
            j = jj[k]

            es.append((i, j))
            es.append((j, i))

            for di in range(-nms, nms + 1):
                for dj in range(-nms, nms + 1):
                    if abs(di) + abs(dj) <= max(min(abs(i - j) - 2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (kf0 <= i1 < t) and (kf1 <= j1 < t):
                            d[(i1 - kf0) * (t - kf1) + (j1 - kf1)] = np.inf

        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove)

    @torch.cuda.amp.autocast(enabled=True)
    def add_factors(self, ii, jj, remove=False) -> None:
        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)
        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        ii, jj = self.__filter_repeated_edges(ii, jj)
        if ii.shape[0] == 0:
            return

        old_factors_count = self.ii.shape[0]
        new_factors_count = ii.shape[0]
        if self.max_factors > 0 and \
           old_factors_count + new_factors_count > self.max_factors \
           and self.correlation_volumes is not None and remove:
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= (self.max_factors - new_factors_count),
                            store=True)

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        if self.corr_impl == "volume":
            is_stereo = (ii == jj).long()
            feature_img_ii = self.features_imgs[None, ii, 0]
            feature_img_jj = self.features_imgs[None, jj, is_stereo]
            corr = CorrBlock(feature_img_ii, feature_img_jj)
            self.correlation_volumes = self.correlation_volumes.cat(corr) \
                if self.correlation_volumes is not None else corr

        gru_hidden_state = self.contexts_imgs[None, ii, 0]
        self.gru_hidden_states = torch.cat([self.gru_hidden_states, gru_hidden_state], 1) \
            if self.gru_hidden_states is not None else gru_hidden_state

        gru_context_input = self.cst_contexts_imgs[None, ii, 0]
        self.gru_contexts_input = torch.cat([self.gru_contexts_input, gru_context_input], 1) \
            if self.gru_contexts_input is not None else gru_context_input

        with torch.cuda.amp.autocast(enabled=False):
            target, _, _ = self.reproject(ii, jj)
            weight = torch.zeros_like(target)

        # Init gru flow with the one from reprojection!
        self.gru_estimated_flow = torch.cat([self.gru_estimated_flow, target],
                                            1)
        self.gru_estimated_flow_weight = torch.cat(
            [self.gru_estimated_flow_weight, weight], 1)

    @torch.cuda.amp.autocast(enabled=True)
    def rm_factors(self, mask, store=False) -> None:
        if store:
            self.ii_inactive = torch.cat([self.ii_inactive, self.ii[mask]], 0)
            self.jj_inactive = torch.cat([self.jj_inactive, self.jj[mask]], 0)
            self.gru_estimated_flow_inactive = torch.cat([
                self.gru_estimated_flow_inactive, self.gru_estimated_flow[:,
                                                                          mask]
            ], 1)
            self.gru_estimated_flow_weight_inactive = torch.cat([
                self.gru_estimated_flow_weight_inactive,
                self.gru_estimated_flow_weight[:, mask]
            ], 1)

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]

        if self.corr_impl == "volume":
            self.correlation_volumes = self.correlation_volumes[~mask]

        if self.gru_hidden_states is not None:
            self.gru_hidden_states = self.gru_hidden_states[:, ~mask]

        if self.gru_contexts_input is not None:
            self.gru_contexts_input = self.gru_contexts_input[:, ~mask]

        self.gru_estimated_flow = self.gru_estimated_flow[:, ~mask]
        self.gru_estimated_flow_weight = self.gru_estimated_flow_weight[:,
                                                                        ~mask]

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        return_distance_matrix = False
        if ii is None:
            return_distance_matrix = True
            N = self.kf_idx
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))

        ii, jj = self.format_indicies(ii, jj, device=self.device)

        if bidirectional:
            poses = self.cam0_T_world[:self.kf_idx + 1].clone()
            d1 = droid_backends.frame_distance(poses, self.cam0_idepths,
                                               self.cam0_intrinsics[0], ii, jj,
                                               beta)
            d2 = droid_backends.frame_distance(poses, self.cam0_idepths,
                                               self.cam0_intrinsics[0], jj, ii,
                                               beta)
            d = .5 * (d1 + d2)
        else:
            d = droid_backends.frame_distance(self.cam0_T_world,
                                              self.cam0_idepths,
                                              self.cam0_intrinsics[0], ii, jj,
                                              beta)

        if return_distance_matrix:
            return d.reshape(N, N)

        return d

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def has_enough_motion(self, current_imgs_features):

        last_img_features = self.features_imgs[self.last_kf_idx][0]
        current_img_features = current_imgs_features[0]
        last_img_context = self.contexts_imgs[self.last_kf_idx][0]
        last_img_gru_input = self.cst_contexts_imgs[self.last_kf_idx][0]

        corr = CorrBlock(last_img_features[None, None],
                         current_img_features[None, None])(self.coords0[None,
                                                                        None])

        _, delta, weight = self.update_net(last_img_context[None, None],
                                           last_img_gru_input[None,
                                                              None], corr)

        has_enough_motion = delta.norm(
            dim=-1).mean().item() > self.motion_filter_thresh
        return has_enough_motion

    def reproject(self, ii, jj, cam_T_body=None, jacobian=False):
        ii, jj = self.format_indicies(ii, jj, device=self.device)
        Gs = lietorch.SE3(self.cam0_T_world[None])

        coords, valid_mask, (Ji, Jj, Jz) = pops.projective_transform(
            Gs,
            self.cam0_idepths[None],
            self.cam0_intrinsics[None],
            ii,
            jj,
            cam_T_body=cam_T_body,
            jacobian=jacobian)
        return coords, valid_mask, (Ji, Jj, Jz)

    def clear_edges(self):
        self.rm_factors(self.ii >= 0)
        self.gru_hidden_states = None
        self.gru_contexts_input = None

    def normalize(self, last_kf=-1):
        s = self.cam0_idepths[:last_kf].mean()
        self.cam0_idepths[:last_kf] /= s
        self.cam0_T_world[:last_kf, :3] *= s
        self.viz_idx[:last_kf] = True

    def get_gt_priors_and_values(self, kf_id, f_id):
        gt_pose = self.world_T_cam0_t0

        pose_key = gtsam.symbol_shorthand.X(f_id)
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]))
        pose_prior = gtsam.PriorFactorPose3(pose_key, gtsam.Pose3(gt_pose),
                                            pose_noise)

        x0 = gtsam.Values()
        x0.insert(pose_key, gtsam.Pose3(gt_pose))

        graph = gtsam.NonlinearFactorGraph()
        graph.push_back(pose_prior)
        return x0, graph

    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, images):
        context_maps, gru_input_maps = self.context_net(images).split(
            [128, 128], dim=2)
        return context_maps.tanh().squeeze(0), gru_input_maps.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, images):
        return self.feature_net(images).squeeze(0)

    def load_weights(self, weights_file):
        state_dict = OrderedDict([(k.replace("module.", ""), v)
                                  for (k,
                                       v) in torch.load(weights_file).items()])
        state_dict = OrderedDict([(k.replace("fnet.", "feature_net."), v)
                                  for (k, v) in state_dict.items()])
        state_dict = OrderedDict([(k.replace("cnet.", "context_net."), v)
                                  for (k, v) in state_dict.items()])
        state_dict = OrderedDict([(k.replace("update.", "update_net."), v)
                                  for (k, v) in state_dict.items()])

        state_dict["update_net.weight.2.weight"] = state_dict[
            "update_net.weight.2.weight"][:2]
        state_dict["update_net.weight.2.bias"] = state_dict[
            "update_net.weight.2.bias"][:2]
        state_dict["update_net.delta.2.weight"] = state_dict[
            "update_net.delta.2.weight"][:2]
        state_dict["update_net.delta.2.bias"] = state_dict[
            "update_net.delta.2.bias"][:2]

        return state_dict

    def __filter_repeated_edges(self, ii, jj):
        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set([(i.item(), j.item()) for i, j in zip(self.ii, self.jj)] +
                   [(i.item(), j.item())
                    for i, j in zip(self.ii_inactive, self.jj_inactive)])

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    def _normalize_imgs(self, images, droid_normalization=True):
        img_normalized = images[:, :, :3, ...] / 255.0
        if droid_normalization:
            mean = torch.as_tensor([0.485, 0.456, 0.406],
                                   device=self.device)[:, None, None]
            stdv = torch.as_tensor([0.229, 0.224, 0.225],
                                   device=self.device)[:, None, None]
        else:
            mean = img_normalized.mean(dim=(3, 4), keepdim=True)
            stdv = img_normalized.std(dim=(3, 4), keepdim=True)
        img_normalized = img_normalized.sub_(mean).div_(stdv)
        return img_normalized

    @staticmethod
    def format_indicies(ii, jj, device: torch.device = torch.device('cuda:0')):
        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device=device, dtype=torch.long).reshape(-1)
        jj = jj.to(device=device, dtype=torch.long).reshape(-1)

        return ii, jj

    def save_mapping_keyframe2frame(self) -> None:
        ic('Saving keyframe2frame mapping...')
        mapping_keyframe2frame = [
            int(frame_index * self.args.frame_stride)
            for frame_index in list(self.kf_idx_to_f_idx.values())
        ]
        with open(self.args.dir_prediction + '/mapping_keyframe2frame.json',
                  'w') as file:
            json.dump(mapping_keyframe2frame, file)

    def save_matrices_origin2frame_keyframes(self) -> None:
        ic('Saving keyframe origin2frame matrices...')
        matrices_origin2frame_keyframes = lietorch.SE3(
            self.cam0_T_world[:self.last_kf_idx +
                              1]).inv().matrix().cpu().numpy()
        with open(
                self.args.dir_prediction +
                f'/matrices/matrices_origin2frame_keyframes_tracking.json',
                'w') as file:
            json.dump(matrices_origin2frame_keyframes.tolist(), file)
