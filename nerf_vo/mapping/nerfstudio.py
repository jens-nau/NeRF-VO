import os
import sys
import json
import numpy as np
import torch
# import wandb
import argparse

from rich import box, style
from pathlib import Path
from rich.panel import Panel
from rich.table import Table

from nerf_vo.mapping.mapping_utils import set_logging_prefix, step_check
from nerf_vo.mapping.nerfstudio_utils import DynamicDataManagerConfig, ExtendedNerfactoModelConfig

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                 'nerf_vo/thirdparty/nerfstudio'))

from nerf_vo.thirdparty.nerfstudio.nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerf_vo.thirdparty.nerfstudio.nerfstudio.configs.base_config import ViewerConfig
from nerf_vo.thirdparty.nerfstudio.nerfstudio.engine.trainer import TrainerConfig
from nerf_vo.thirdparty.nerfstudio.nerfstudio.engine.callbacks import TrainingCallbackLocation
from nerf_vo.thirdparty.nerfstudio.nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerf_vo.thirdparty.nerfstudio.nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerf_vo.thirdparty.nerfstudio.nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerf_vo.thirdparty.nerfstudio.nerfstudio.utils import profiler
from nerf_vo.thirdparty.nerfstudio.nerfstudio.utils.poses import multiply
from nerf_vo.thirdparty.nerfstudio.nerfstudio.utils.rich_utils import CONSOLE


class Nerfstudio:

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

        self.config = TrainerConfig(
            project_name='nerf_vo',
            experiment_name=args.experiment,
            method_name='extended_nerfacto',
            output_dir=Path(args.dir_prediction + '/nerfstudio'),
            relative_model_dir=Path('../../../../snapshots'),
            save_only_latest_checkpoint=False,
            steps_per_save=args.mapping_snapshot_iterations,
            steps_per_eval_batch=512,
            steps_per_eval_image=512,
            steps_per_eval_all_images=512,
            max_num_iterations=args.mapping_iterations,
            mixed_precision=True,
            pipeline=VanillaPipelineConfig(
                datamanager=DynamicDataManagerConfig(
                    train_num_rays_per_batch=4096,
                    eval_num_rays_per_batch=4096,
                    camera_optimizer=CameraOptimizerConfig(mode='SE3'),
                    num_frames=args.num_keyframes,
                    frame_height=args.frame_height,
                    frame_width=args.frame_width,
                    use_normals=True
                    if 'normal' in args.enhancement_module else False,
                ),
                model=ExtendedNerfactoModelConfig(
                    interlevel_loss_mult=1.0,
                    distortion_loss_mult=0.002,
                    orientation_loss_mult=0,
                    pred_normal_loss_mult=0,
                    depth_loss_mult=0.001,
                    normal_loss_mult=0.000005,
                    predict_normals=True,
                    is_euclidean_depth=False,
                    depth_sigma=0.001,
                    should_decay_sigma=False,
                ),
            ),
            optimizers={
                'proposal_networks': {
                    'optimizer': AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                    'scheduler': None,
                },
                'fields': {
                    'optimizer': AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                    'scheduler': None,
                },
                "camera_opt": {
                    "optimizer":
                    AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                    "scheduler":
                    ExponentialDecaySchedulerConfig(
                        lr_final=1e-5, max_steps=args.mapping_iterations),
                },
            },
            viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
            vis='viewer',
        )

        self.config.set_timestamp()
        self.config.print_to_terminal()
        self.config.save_config()
        self.trainer = self.config.setup()
        self.trainer.setup()

    def __call__(self, input: dict) -> None:
        if self.step == self.config.max_num_iterations:
            self.shut_down()
        else:
            if input is not None:
                self.update(input=input)
            if self.is_initialized:
                self.train()

    def update(self, input: dict) -> None:
        self.trainer.pipeline.datamanager.train_dataset.update(input=input)
        if not self.is_initialized:
            self.trainer._init_viewer_state()

        if (self.config.is_viewer_enabled()
                or self.config.is_viewer_beta_enabled()):
            image_indices = self.trainer.viewer_state._pick_drawn_image_idxs(
                self.trainer.pipeline.datamanager.train_dataset.
                num_active_frames)
            for index in image_indices:
                image = self.trainer.pipeline.datamanager.train_dataset[index][
                    'image']
                bgr = image[..., [2, 1, 0]]
                camera_json = self.trainer.pipeline.datamanager.train_dataset.cameras.to_json(
                    camera_idx=int(index), image=bgr, max_size=128)
                self.trainer.viewer_state.viser_server.add_dataset_image(
                    idx=f'{index:06d}', json=camera_json)

        self.is_initialized = True
        torch.cuda.empty_cache()

    def train(self) -> None:
        with self.trainer.train_lock:
            self.trainer.pipeline.train()

            for callback in self.trainer.callbacks:
                callback.run_callback_at_location(
                    self.step,
                    location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)

            loss, loss_dict, metrics_dict = self.trainer.train_iteration(
                self.step)

            for callback in self.trainer.callbacks:
                callback.run_callback_at_location(
                    self.step,
                    location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION)

        self.trainer._update_viewer_state(self.step)

        if step_check(self.step,
                      self.config.logging.steps_per_log,
                      run_at_zero=True):
            loss_dict = set_logging_prefix(metrics=loss_dict, prefix='loss/')
            metrics_dict = set_logging_prefix(metrics=metrics_dict,
                                              prefix='metrics/')
            # wandb.log(loss_dict, step=self.step)
            # wandb.log(metrics_dict, step=self.step)

        if step_check(self.step, self.config.steps_per_save):
            self.save_snapshot()

        self.step += 1

    def shut_down(self) -> None:
        self.save_snapshot()
        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row('Config File',
                      str(self.config.get_base_dir() / 'config.yml'))
        table.add_row('Checkpoint Directory', str(self.trainer.checkpoint_dir))
        CONSOLE.print(
            Panel(table,
                  title='[bold][green]:tada: Training Finished :tada:[/bold]',
                  expand=False))

        for callback in self.trainer.callbacks:
            callback.run_callback_at_location(
                step=self.step, location=TrainingCallbackLocation.AFTER_TRAIN)

        profiler.flush_profiler(self.config.logging)
        self.is_shut_down = True

    def save_snapshot(self) -> None:
        self.trainer.save_checkpoint(self.step)
        self.trainer.pipeline.datamanager.train_dataset.save_dataset(
            dir_prediction=self.args.dir_prediction)
        with open(
                self.args.dir_prediction +
                '/matrices/matrices_origin2frame_training.json', 'w') as file:
            matrices_origin2frame_training = np.tile(
                np.eye(4), (self.trainer.pipeline.datamanager.train_dataset.
                            num_active_frames, 1, 1))
            matrices_origin2frame_training[:, :3] = multiply(
                self.trainer.pipeline.model.camera_optimizer(
                    torch.arange(self.trainer.pipeline.datamanager.
                                 train_dataset.num_active_frames).to(
                                     self.device)),
                self.trainer.pipeline.datamanager.train_dataset.cameras.
                camera_to_worlds[:self.trainer.pipeline.datamanager.
                                 train_dataset.num_active_frames].to(
                                     self.device)).detach().cpu().numpy()
            json.dump(matrices_origin2frame_training.tolist(), file)
