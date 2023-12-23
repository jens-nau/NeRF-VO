import torch
import argparse
import torchvision

from typing import Literal, Tuple
from nerf_vo.thirdparty.omnidata.omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel

omnidata_dpt_depth_v2 = 'nerf_vo/build/omnidata_models/omnidata_dpt_depth_v2.ckpt'
omnidata_dpt_normal_v2 = 'nerf_vo/build/omnidata_models/omnidata_dpt_normal_v2.ckpt'


class OmnidataEstimator:

    def __init__(
        self,
        args: argparse.Namespace,
        device: torch.device = torch.device('cuda:0'),
        mode: Literal['depth-normal', 'depth',
                      'normal'] = 'depth-normal') -> None:
        self.args = args
        self.device = device
        self.mode = mode
        self.is_initialized = False
        self.is_shut_down = False

        self.reference_height = args.frame_height
        self.reference_width = args.frame_width
        self.processing_height = 384
        self.processing_width = 384
        self.batch_size = 1

        if 'depth' in mode:
            self.depth_estimator = self.init_model(mode='depth')
            self.depth_normalize_transform = torchvision.transforms.Normalize(
                mean=0.5, std=0.5)
        if 'normal' in mode:
            self.normal_estimator = self.init_model(mode='normal')

        torch.cuda.empty_cache()

        self.is_initialized = True

    def __call__(
            self,
            frames_color: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frames_depth = None
        frames_normal = None
        if 'depth' in self.mode:
            frames_depth = self.estimate(frames_color=frames_color.clone(),
                                         mode='depth')
        if 'normal' in self.mode:
            frames_normal = self.estimate(frames_color=frames_color,
                                          mode='normal')
        return frames_depth, frames_normal

    def init_model(self, mode: Literal['depth', 'normal']) -> DPTDepthModel:
        model = DPTDepthModel(backbone='vitb_rn50_384',
                              num_channels=(3 if mode == 'normal' else 1))
        checkpoint = torch.load((omnidata_dpt_normal_v2 if mode == 'normal'
                                 else omnidata_dpt_depth_v2),
                                map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def estimate(self, frames_color: torch.Tensor,
                 mode: Literal['depth', 'normal']) -> torch.Tensor:
        with torch.no_grad():
            num_frames = frames_color.shape[0]
            frames_color_processing_size = torch.nn.functional.interpolate(
                frames_color,
                size=(self.processing_height, self.processing_width),
                mode='bicubic')
            if mode == 'depth':
                frames_color_processing_size = torch.stack([
                    self.depth_normalize_transform(frame_color)
                    for frame_color in frames_color_processing_size
                ])
            frames_estimation_processing_size = torch.zeros(
                (num_frames, (3 if mode == 'normal' else 1),
                 self.processing_height, self.processing_width),
                dtype=torch.float32,
                device=frames_color_processing_size.device)
            for index in range(0, num_frames, self.batch_size):
                batch_size = self.batch_size if num_frames - \
                    index >= self.batch_size else num_frames - index
                frames_color_processing_size_batch = frames_color_processing_size[
                    index:index + batch_size]
                frames_estimation_processing_size_batch = (
                    self.depth_estimator
                    if mode == 'depth' else self.normal_estimator
                )(frames_color_processing_size_batch).clamp(min=0, max=1)
                if mode == 'depth':
                    frames_estimation_processing_size_batch = frames_estimation_processing_size_batch[:,
                                                                                                      None, :, :]
                frames_estimation_processing_size[
                    index:index +
                    batch_size] = frames_estimation_processing_size_batch
                torch.cuda.empty_cache()
            frames_estimation = torch.nn.functional.interpolate(
                frames_estimation_processing_size,
                size=(self.reference_height, self.reference_width),
                mode='bicubic')
            return frames_estimation
