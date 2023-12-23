import os
import json
import yaml
import numpy as np
import torch
# import wandb
import random
import argparse
import datetime

from nerf_vo.execute import execute
from evaluation.renderer import Renderer
from evaluation.evaluator import Evaluator
from evaluation.nerf_renderer import NerfstudioRenderer, NeRFSLAMNGPRenderer
from evaluation.evaluation_utils import set_logging_prefix
from evaluation.datasets.eth3d_dataset import ETH3DDataset
from evaluation.datasets.replica_dataset import ReplicaDataset
from evaluation.datasets.scannet_dataset import ScanNetDataset
from evaluation.datasets.seven_scenes_dataset import SevenScenesDataset
from evaluation.datasets.tum_rgbd_dataset import TUMRGBDDataset


def _set_random_seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def execute_render_and_evaluate(args: argparse.Namespace,
                                config: dict) -> None:
    mapping_model = execute(args)
    torch.cuda.empty_cache()

    if config['mapping_module'] == 'instant-ngp':
        nerf = NeRFSLAMNGPRenderer(mapping_model=mapping_model)
    elif config['mapping_module'] == 'nerfstudio':
        nerf = NerfstudioRenderer(mapping_model=mapping_model)
    else:
        raise NotImplementedError

    with open(config['dir_prediction'] +
              '/mapping_keyframe2frame.json') as file:
        keyframes = json.load(file)

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
    if 'evaluation_frame_height' in config and 'evaluation_frame_width' in config:
        dataset = dataset_class(
            dir_dataset=config['dir_dataset'],
            num_evaluation_frames=config['num_evaluation_frames'],
            frame_height=config['evaluation_frame_height'],
            frame_width=config['evaluation_frame_width'])
    else:
        dataset = dataset_class(
            dir_dataset=config['dir_dataset'],
            num_evaluation_frames=config['num_evaluation_frames'])

    renderer = Renderer(config=config, dataset=dataset, nerf=nerf)
    renderer.render_camera_extrinsics_keyframes()
    renderer.render_frames()
    renderer.render_mesh()

    evaluator = Evaluator(config=config, dataset=dataset)
    metrics_trajectory = evaluator.calculate_metrics_trajectory()
    print(metrics_trajectory)
    metrics_2d = evaluator.calculate_metrics_2d()
    print(metrics_2d)
    metrics_3d = evaluator.calculate_metrics_3d()
    print(metrics_3d)

    metrics = {}
    metrics['iteration'] = config['mapping_iterations']
    metrics_trajectory = set_logging_prefix(metrics=metrics_trajectory,
                                            prefix='trajectory/')
    metrics.update(metrics_trajectory)
    metrics_2d = set_logging_prefix(metrics=metrics_2d, prefix='2d/')
    metrics.update(metrics_2d)
    if metrics_3d is not None:
        metrics_3d = set_logging_prefix(metrics=metrics_3d, prefix='3d/')
        metrics.update(metrics_3d)
    # wandb.log(metrics)


def main() -> None:

    parser = argparse.ArgumentParser(
        description='Execute NeRF-VO, render results and calculate metrics.')
    parser.add_argument('--config',
                        type=str,
                        default='nerf_vo_replica',
                        help='name of the config')
    parser.add_argument('--experiment',
                        type=str,
                        default='1st_commit',
                        help='experiment name to identify execution')
    parser.add_argument('--first_scene',
                        type=int,
                        default=0,
                        help='first scene to process')
    parser.add_argument('--last_scene',
                        type=int,
                        default=7,
                        help='last scene to process')
    args = parser.parse_args()

    first_scene = args.first_scene
    last_scene = args.last_scene
    experiment = args.experiment

    file_config = f'configs/{args.config}.yaml'
    with open(file_config, 'r') as file:
        config = yaml.safe_load(file)

    dir_dataset = config['dir_dataset']
    current_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    dir_prediction = config['dir_prediction'] + \
        f'/{args.config}_{current_date}{"" if args.experiment == "" else f"_{args.experiment}"}'
    experiment = f'/{args.config}_{current_date}' if experiment == '' else experiment
    os.makedirs(dir_prediction) if not os.path.exists(dir_prediction) else None
    dir_result = dir_prediction + '/results'
    os.makedirs(dir_result) if not os.path.exists(dir_result) else None

    with open(dir_prediction + '/config.json', "w") as file:
        json.dump(config, file)

    torch.cuda.empty_cache()
    _set_random_seed(42)
    torch.multiprocessing.set_start_method('spawn')
    torch.backends.cudnn.benchmark = True

    for index, scene_name in enumerate(config['scene_names']):
        if index < first_scene:
            continue
        if index > last_scene:
            break

        config['dir_dataset'] = dir_dataset + f'/{scene_name}'
        config['dir_prediction'] = dir_prediction + f'/{scene_name}'
        os.makedirs(config['dir_prediction']) if not os.path.exists(
            config['dir_prediction']) else None
        config['dir_result'] = dir_result + f'/{scene_name}'
        os.makedirs(config['dir_result']) if not os.path.exists(
            config['dir_result']) else None
        config['experiment'] = experiment
        config['scene_name'] = scene_name
        if 'depth_supervision_lambdas' in config:
            config['depth_supervision_lambda'] = config[
                'depth_supervision_lambdas'][index]
        if 'extrinsic_learning_rates' in config:
            config['extrinsic_learning_rate'] = config[
                'extrinsic_learning_rates'][index]
        args = argparse.Namespace(**config)

        # wandb.init(
        #     project=experiment,
        #     config=config,
        #     name=scene_name,
        #     dir=config['dir_prediction']
        # )

        execute_render_and_evaluate(args=args, config=config)
        torch.cuda.empty_cache()

        # wandb.finish()


if __name__ == '__main__':
    main()
