import json
import torch
import argparse
import lietorch

from collections import deque

from nerf_vo.thirdparty.dpvo.dpvo.config import cfg
from nerf_vo.thirdparty.dpvo.dpvo.dpvo import DPVO

file_dpvo_weights = 'nerf_vo/build/dpvo/dpvo.pth'
file_dpvo_config = 'nerf_vo/thirdparty/dpvo/config/default.yaml'


class DPVOHandler:

    def __init__(
        self,
        args: argparse.Namespace,
        device: torch.device = torch.device('cuda:0')
    ) -> None:
        self.args = args
        self.device = device
        self.is_initialized = False
        self.is_shut_down = False

        self.num_keyframes = args.num_keyframes
        self.camera_intrinsics = []
        self.frames_color = []

        self.config = cfg
        self.config.merge_from_file(file_dpvo_config)
        self.config.BUFFER_SIZE = args.num_keyframes
        self.config.PATCHES_PER_FRAME = args.patches_per_frame
        self.config.REMOVAL_WINDOW = args.removal_window
        self.config.OPTIMIZATION_WINDOW = args.optimization_window
        self.config.PATCH_LIFETIME = args.patch_lifetime
        self.config.KEYFRAME_THRESH = args.keyframe_threshold

        self.buffer_camera_intrinsics = deque(
            maxlen=self.config.KEYFRAME_INDEX)
        self.buffer_frames_color = deque(maxlen=self.config.KEYFRAME_INDEX)
        self.buffer_frames_depth = deque(maxlen=4)
        self.last_frame_is_keyframe = False

        self.initialize(frame_height=args.frame_height,
                        frame_width=args.frame_width)

    def __call__(self, input) -> dict:
        input['camera_intrinsics'] = torch.tensor([
            input['camera_intrinsics']['fx'], input['camera_intrinsics']['fy'],
            input['camera_intrinsics']['cx'], input['camera_intrinsics']['cy']
        ],
                                                  dtype=torch.float32,
                                                  device=self.device)
        input['frame_color'] = torch.from_numpy(input['frame_color']).permute(
            2, 0, 1).to(self.device)

        self.buffer_camera_intrinsics.append(input['camera_intrinsics'])
        self.buffer_frames_color.append(input['frame_color'])

        keyframe_indices = self.dpvo(tstamp=input['frame_index'],
                                     image=input['frame_color'],
                                     intrinsics=input['camera_intrinsics'])

        if input['last_frame']:
            if keyframe_indices is None:
                keyframe_indices = torch.arange(
                    max(self.dpvo.n - self.config.REMOVAL_WINDOW, 0),
                    self.dpvo.n - self.config.KEYFRAME_INDEX + 2,
                    dtype=torch.long,
                    device=self.device)
            else:
                self.last_frame_is_keyframe = True

        if keyframe_indices is not None:
            if not self.dpvo.is_initialized:
                if self.dpvo.n <= 8 - self.config.KEYFRAME_INDEX:
                    self.camera_intrinsics.append(input['camera_intrinsics'])
                    self.frames_color.append(input['frame_color'])
            else:
                self.camera_intrinsics.append(self.buffer_camera_intrinsics[0])
                self.frames_color.append(self.buffer_frames_color[0])

                output = {
                    'keyframe_indices':
                    keyframe_indices,
                    'camera_intrinsics':
                    torch.stack(self.camera_intrinsics),
                    'camera_extrinsics':
                    lietorch.SE3(
                        self.dpvo.poses_[keyframe_indices]).inv().matrix(),
                    'frames_color':
                    torch.stack(self.frames_color),
                    'dpvo_patches':
                    self.dpvo.patches_[keyframe_indices],
                    'last_frame':
                    input['last_frame'],
                }
                self.camera_intrinsics = []
                self.frames_color = []

                if input['last_frame']:
                    self.shut_down()

                torch.cuda.empty_cache()
                return output

        return None

    def initialize(self, frame_height: int, frame_width: int) -> None:
        self.dpvo = DPVO(cfg=self.config,
                         network=file_dpvo_weights,
                         ht=frame_height,
                         wd=frame_width)
        self.is_initialized = True

    def shut_down(self) -> None:
        mapping_keyframe2frame = [
            int(frame_index * self.args.frame_stride) for frame_index in self.
            dpvo.tstamps_[:self.dpvo.n - self.config.KEYFRAME_INDEX +
                          (1 if self.last_frame_is_keyframe else 2)].cpu()
        ]
        with open(self.args.dir_prediction + '/mapping_keyframe2frame.json',
                  'w') as file:
            json.dump(mapping_keyframe2frame, file)

        matrices_origin2frame_keyframes = lietorch.SE3(
            self.dpvo.poses_[:self.dpvo.n - self.config.KEYFRAME_INDEX +
                             (1 if self.last_frame_is_keyframe else 2)]).inv(
                             ).matrix().cpu().numpy()
        with open(
                self.args.dir_prediction +
                f'/matrices/matrices_origin2frame_keyframes_tracking.json',
                'w') as file:
            json.dump(matrices_origin2frame_keyframes.tolist(), file)

        self.is_shut_down = True
