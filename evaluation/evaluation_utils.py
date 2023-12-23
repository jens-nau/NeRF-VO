import os
import cv2
import json
import tqdm
import lpips
import scipy
import numpy as np
import torch
import open3d as o3d
import lietorch


def set_logging_prefix(metrics, prefix):
    metrics_with_prefix = {}
    for key, value in metrics.items():
        new_key = prefix + str(key)
        metrics_with_prefix[new_key] = value
    return metrics_with_prefix


def load_camera_intrinsics(dir_dataset: str, dataset_name: str) -> dict:
    with open(os.path.dirname(dir_dataset) + '/camera_parameters.json',
              'r') as file:
        camera_parameters = json.load(file)[dataset_name]
    camera_intrinsics = {
        'height': camera_parameters['h'],
        'width': camera_parameters['w'],
        'fx': camera_parameters['fx'],
        'fy': camera_parameters['fy'],
        'cx': camera_parameters['cx'],
        'cy': camera_parameters['cy'],
        'depth_scale': camera_parameters['depth_scale'],
    }
    for key in ["k1", "k2", "k3", "p1", "p2"]:
        if key in camera_parameters:
            camera_intrinsics[key] = camera_parameters[key]
    return camera_intrinsics


def scale_camera_intrinsics(camera_intrinsics: dict, height: int,
                            width: int) -> dict:
    scale_factor_x = width / camera_intrinsics['width']
    scale_factor_y = height / camera_intrinsics['height']
    camera_intrinsics['height'] = height
    camera_intrinsics['width'] = width
    camera_intrinsics['fx'] *= scale_factor_x
    camera_intrinsics['fy'] *= scale_factor_y
    camera_intrinsics['cx'] *= scale_factor_x
    camera_intrinsics['cy'] *= scale_factor_y
    return camera_intrinsics


def read_timestamp_data(dir_dataset: str, mode: str = "color") -> dict:
    if mode == "color":
        file_association_data = dir_dataset + "/rgb.txt"
    elif mode == "depth":
        file_association_data = dir_dataset + "/depth.txt"
    elif mode == "camera_extrinsics":
        file_association_data = dir_dataset + "/groundtruth.txt"
    else:
        raise NotImplementedError

    with open(file_association_data) as file:
        data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    lines = [[
        element.strip() for element in line.split(" ") if element.strip() != ""
    ] for line in lines if len(line) > 0 and line[0] != "#"]
    return dict([(float(line[0]), line[1:]) for line in lines
                 if len(line) > 1])


def associate_timestamp_data(source_timestamps: list,
                             target_timestamps: list) -> list:
    MAX_DIFFERENCE = 0.02
    potential_matches = [
        (abs(source_timestamp - target_timestamp), source_timestamp,
         target_timestamp) for source_timestamp in source_timestamps
        for target_timestamp in target_timestamps
        if abs(source_timestamp - target_timestamp) < MAX_DIFFERENCE
    ]
    potential_matches.sort()
    matches = []
    for difference, source_timestamp, target_timestamp in potential_matches:
        if (source_timestamp in source_timestamps
                and target_timestamp in target_timestamps):
            source_timestamps.remove(source_timestamp)
            target_timestamps.remove(target_timestamp)
            matches.append((source_timestamp, target_timestamp))
    return matches


def interpolate_invalid_camera_extrinsics(
        camera_extrinsics: np.array) -> np.array:
    if np.isinf(camera_extrinsics).any(axis=(1, 2)).sum() > 0:
        invalid_camera_extrinsics = np.isinf(camera_extrinsics).any(axis=(1,
                                                                          2))
        valid_camera_extrinsics = np.tile(np.eye(4),
                                          (camera_extrinsics.shape[0], 1, 1))

        for index in range(camera_extrinsics.shape[0]):
            if invalid_camera_extrinsics[index]:
                previous_valid_camera_extrinsics = index - 1
                next_valid_camera_extrinsics = index + 1

                while previous_valid_camera_extrinsics >= 0 and invalid_camera_extrinsics[
                        previous_valid_camera_extrinsics]:
                    previous_valid_camera_extrinsics -= 1

                while next_valid_camera_extrinsics < camera_extrinsics.shape[
                        0] and invalid_camera_extrinsics[
                            next_valid_camera_extrinsics]:
                    next_valid_camera_extrinsics += 1

                if previous_valid_camera_extrinsics >= 0 and next_valid_camera_extrinsics < camera_extrinsics.shape[
                        0]:
                    tquads_frame2origin_previous_valid_camera_extrinsics = lietorch.SE3(
                        torch.tensor(
                            np.concatenate(
                                (camera_extrinsics[
                                    previous_valid_camera_extrinsics, :3, 3],
                                 scipy.spatial.transform.Rotation.from_matrix(
                                     camera_extrinsics[
                                         previous_valid_camera_extrinsics, :
                                         3, :3]).as_quat()))[None])).inv()
                    tquads_frame2origin_next_valid_camera_extrinsics = lietorch.SE3(
                        torch.tensor(
                            np.concatenate(
                                (camera_extrinsics[
                                    next_valid_camera_extrinsics, :3, 3],
                                 scipy.spatial.transform.Rotation.from_matrix(
                                     camera_extrinsics[
                                         next_valid_camera_extrinsics, :3, :3]
                                 ).as_quat()))[None])).inv()

                    pose_difference = tquads_frame2origin_next_valid_camera_extrinsics * tquads_frame2origin_previous_valid_camera_extrinsics.inv(
                    )
                    tangent_velocity = pose_difference.log() / torch.tensor(
                        next_valid_camera_extrinsics -
                        previous_valid_camera_extrinsics)[None]
                    tangent_motion = tangent_velocity * torch.tensor(
                        index - previous_valid_camera_extrinsics)[None]
                    valid_camera_extrinsics[index] = (
                        lietorch.SE3.exp(tangent_motion) *
                        tquads_frame2origin_previous_valid_camera_extrinsics
                    ).inv().matrix()[0].numpy()
                elif previous_valid_camera_extrinsics >= 0:
                    valid_camera_extrinsics[index] = camera_extrinsics[
                        previous_valid_camera_extrinsics]
                elif next_valid_camera_extrinsics < camera_extrinsics.shape[0]:
                    valid_camera_extrinsics[index] = camera_extrinsics[
                        next_valid_camera_extrinsics]
            else:
                valid_camera_extrinsics[index] = camera_extrinsics[index]
        return valid_camera_extrinsics
    else:
        return camera_extrinsics


def integrate_mesh(file_mesh: str, camera_intrinsics: dict,
                   camera_extrinsics: np.array, frames_color: list,
                   frames_depth: list) -> None:
    VOXEL_SIZE = 1 / 64
    BLOCK_RESOLUTION = 8
    BLOCK_COUNT = 100000
    DEPTH_TRUNCATION = 5.0

    device = o3d.core.Device('CPU:0')
    depth_scale = camera_intrinsics['depth_scale']

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=camera_intrinsics['width'],
        height=camera_intrinsics['height'],
        fx=camera_intrinsics['fx'],
        fy=camera_intrinsics['fy'],
        cx=camera_intrinsics['cx'],
        cy=camera_intrinsics['cy'])

    frames_color = [
        o3d.t.geometry.Image(np.ascontiguousarray(frame_color)).to(device)
        for frame_color in frames_color
    ]
    frames_depth = [
        o3d.t.geometry.Image(
            np.ascontiguousarray(
                (frame_depth * depth_scale).astype(np.uint16))).to(device)
        for frame_depth in frames_depth
    ]

    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight', 'color'),
                                        attr_dtypes=(o3d.core.float32,
                                                     o3d.core.float32,
                                                     o3d.core.float32),
                                        attr_channels=((1), (1), (3)),
                                        voxel_size=VOXEL_SIZE,
                                        block_resolution=BLOCK_RESOLUTION,
                                        block_count=BLOCK_COUNT,
                                        device=device)

    try:
        for camera_matrix_origin2frame, frame_color, frame_depth in tqdm.tqdm(
                zip(camera_extrinsics, frames_color, frames_depth)):
            camera_matrix_frame2origin = np.linalg.inv(
                camera_matrix_origin2frame)

            frustum_block_coords = vbg.compute_unique_block_coordinates(
                depth=frame_depth,
                intrinsic=o3d.core.Tensor(camera_intrinsics.intrinsic_matrix),
                extrinsic=o3d.core.Tensor(camera_matrix_frame2origin),
                depth_scale=depth_scale,
                depth_max=DEPTH_TRUNCATION)

            vbg.integrate(
                block_coords=frustum_block_coords,
                depth=frame_depth,
                color=frame_color,
                intrinsic=o3d.core.Tensor(camera_intrinsics.intrinsic_matrix),
                extrinsic=o3d.core.Tensor(camera_matrix_frame2origin),
                depth_scale=depth_scale,
                depth_max=DEPTH_TRUNCATION)

        mesh = vbg.extract_triangle_mesh()
        mesh = mesh.to_legacy()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(file_mesh, mesh)
    except FileExistsError:
        print('Error: integrate_mesh')


def kabsch_umeyama_alignment(points_target: np.array,
                             points_source: np.array,
                             with_scale: bool = True) -> tuple:
    assert points_target.shape == points_source.shape
    num_points, num_dimensions = points_target.shape

    mean_points_target = points_target.mean(axis=0)
    mean_points_source = points_source.mean(axis=0)
    variance_points_target = np.mean(
        np.linalg.norm(points_target - mean_points_target, axis=1)**2)

    covariance_matrix = ((points_target - mean_points_target).T
                         @ (points_source - mean_points_source)) / num_points
    U, D, VT = np.linalg.svd(covariance_matrix)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (num_dimensions - 1) + [d])

    rotation = U @ S @ VT
    scale = variance_points_target / \
        np.trace(np.diag(D) @ S) if with_scale else 1.0
    translation = mean_points_target - scale * rotation @ mean_points_source

    return rotation, translation.reshape(3, 1), scale


def calculate_absolute_trajectory_error(matrices_origin2frame_gt: np.array,
                                        matrices_origin2frame_pred: np.array,
                                        with_scale: bool = True) -> dict:
    trajectory_gt = matrices_origin2frame_gt[:, :3, 3]
    trajectory_pred = matrices_origin2frame_pred[:, :3, 3]
    rotation, translation, scale = kabsch_umeyama_alignment(
        points_target=trajectory_gt,
        points_source=trajectory_pred,
        with_scale=with_scale)
    trajectory_pred_aligned = (scale * rotation @ trajectory_pred.T +
                               translation).T
    alignment_error = trajectory_pred_aligned - trajectory_gt
    trajectory_error = np.sqrt(
        np.sum(np.multiply(alignment_error, alignment_error), axis=1))
    absolute_trajectory_error = {
        'absolute_trajectory_error_rmse':
        np.sqrt(
            np.dot(trajectory_error, trajectory_error) /
            len(trajectory_error)),
        'absolute_trajectory_error_mean':
        np.mean(trajectory_error),
        'absolute_trajectory_error_median':
        np.median(trajectory_error),
        'absolute_trajectory_error_std':
        np.std(trajectory_error),
        'absolute_trajectory_error_max':
        np.max(trajectory_error),
        'absolute_trajectory_error_min':
        np.min(trajectory_error),
    }

    return absolute_trajectory_error


def calculate_psnr(image1: np.array, image2: np.array) -> float:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        image1: numpy array representing the first image
        image2: numpy array representing the second image

    Returns:
        The PSNR value between the two images
    """
    PIXEL_MAX = 255.0

    mse = np.mean((image1 - image2)**2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    return psnr


def calculate_psnr_color(image1: np.array, image2: np.array) -> float:
    red1, green1, blue1 = cv2.split(image1)
    red2, green2, blue2 = cv2.split(image2)
    psnr_red = calculate_psnr(red1, red2)
    psnr_green = calculate_psnr(green1, green2)
    psnr_blue = calculate_psnr(blue1, blue2)
    psnr = (psnr_red + psnr_green + psnr_blue) / 3.0

    return psnr


# NOTE: Adapted from https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e
def calculate_mssim(image1: torch.Tensor, image2: torch.Tensor) -> float:
    KERNEL_SIZE = 11
    PADDING = 5
    STANDARD_DEVIATION = 1.5
    IMAGE_CHANNELS = 3
    C1 = (0.01)**2
    C2 = (0.03)**2

    with torch.no_grad():
        gaussian_kernel_1d = torch.Tensor([
            np.exp(-(x - KERNEL_SIZE // 2)**2 /
                   float(2 * STANDARD_DEVIATION**2))
            for x in range(KERNEL_SIZE)
        ])
        gaussian_kernel_1d = (gaussian_kernel_1d /
                              gaussian_kernel_1d.sum()).unsqueeze(1)
        gaussian_kernel_2d = gaussian_kernel_1d.mm(
            gaussian_kernel_1d.t()).float()
        gaussian_kernel = gaussian_kernel_2d.expand(IMAGE_CHANNELS, 1,
                                                    KERNEL_SIZE,
                                                    KERNEL_SIZE).contiguous()

        luminance1 = torch.nn.functional.conv2d(image1,
                                                gaussian_kernel,
                                                padding=PADDING,
                                                groups=IMAGE_CHANNELS)
        luminance2 = torch.nn.functional.conv2d(image2,
                                                gaussian_kernel,
                                                padding=PADDING,
                                                groups=IMAGE_CHANNELS)
        square_luminance1 = luminance1**2
        square_luminance2 = luminance2**2
        luminance12 = luminance1 * luminance2

        square_contrast1 = torch.nn.functional.conv2d(
            image1 * image1,
            gaussian_kernel,
            padding=PADDING,
            groups=IMAGE_CHANNELS) - square_luminance1
        square_contrast2 = torch.nn.functional.conv2d(
            image2 * image2,
            gaussian_kernel,
            padding=PADDING,
            groups=IMAGE_CHANNELS) - square_luminance2
        contrast12 = torch.nn.functional.conv2d(
            image1 * image2,
            gaussian_kernel,
            padding=PADDING,
            groups=IMAGE_CHANNELS) - luminance12

        ssim_map = ((2 * luminance12 + C1) * (2 * contrast12 + C2)) / (
            (square_luminance1 + square_luminance2 + C1) *
            (square_contrast1 + square_contrast2 + C2))
        ssim = ssim_map.mean()

    return ssim.item()


def calculate_depth_metrics_2d(frame_depth_gt: np.array,
                               frame_depth_pred: np.array,
                               with_scale: bool = True) -> dict:
    mask = (frame_depth_gt > 0) * (frame_depth_pred > 0) * \
        (frame_depth_gt < 5) * (frame_depth_pred < 5)
    frame_depth_gt = frame_depth_gt[mask]
    frame_depth_pred = frame_depth_pred[mask]

    if with_scale:
        depth_scale_pred2gt = frame_depth_gt.mean() / frame_depth_pred.mean()
        frame_depth_pred *= depth_scale_pred2gt

    absolute_difference = np.abs(frame_depth_pred - frame_depth_gt)
    absolute_relative = absolute_difference / frame_depth_pred
    square_difference = absolute_difference**2
    square_relative = square_difference / frame_depth_pred
    square_log_difference = (np.log(frame_depth_pred) -
                             np.log(frame_depth_gt))**2
    threshold = np.maximum((frame_depth_gt / frame_depth_pred),
                           (frame_depth_pred / frame_depth_gt))
    delta1 = (threshold < 1.25).astype('float')
    delta2 = (threshold < 1.25**2).astype('float')
    delta3 = (threshold < 1.25**3).astype('float')

    depth_metrics_2d = {
        'absolute_relative': np.mean(absolute_relative),
        'absolute_difference': np.mean(absolute_difference),
        'square_relative': np.mean(square_relative),
        'square_difference': np.sqrt(np.mean(square_difference)),
        'square_log_difference': np.sqrt(np.mean(square_log_difference)),
        'delta1': np.mean(delta1),
        'delta2': np.mean(delta2),
        'delta3': np.mean(delta3),
    }

    return depth_metrics_2d


def calculate_color_metrics_2d(frame_color_gt: np.array,
                               frame_color_pred: np.array,
                               lpips_loss: lpips.LPIPS) -> dict:
    PIXEL_MAX = 255.0

    psnr = calculate_psnr_color(image1=frame_color_gt, image2=frame_color_pred)

    tensor_color_gt = torch.Tensor(frame_color_gt.transpose(
        (2, 0, 1))).unsqueeze(0).float().div(PIXEL_MAX) * 2 - 1
    tensor_color_pred = torch.Tensor(frame_color_pred.transpose(
        (2, 0, 1))).unsqueeze(0).float().div(PIXEL_MAX) * 2 - 1

    mssim = calculate_mssim(image1=tensor_color_gt.clone(),
                            image2=tensor_color_pred.clone())

    tensor_color_gt = tensor_color_gt * 2 - 1
    tensor_color_pred = tensor_color_pred * 2 - 1

    lpips = lpips_loss(tensor_color_gt, tensor_color_pred).item()

    color_metrics_2d = {
        'psnr': psnr,
        'mssim': mssim,
        'lpips': lpips,
    }

    return color_metrics_2d


def get_pcd_alignment_transformation(source: o3d.t.geometry.PointCloud,
                                     target: o3d.t.geometry.PointCloud):

    max_correspondence_distance = 0.02
    estimation_method = o3d.t.pipelines.registration.TransformationEstimationPointToPoint(
    )
    criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=0.0000001, relative_rmse=0.0000001, max_iteration=30)

    registration_icp = o3d.t.pipelines.registration.icp(
        source=source,
        target=target,
        max_correspondence_distance=max_correspondence_distance,
        estimation_method=estimation_method,
        criteria=criteria)

    return registration_icp.transformation


def calculate_metrics_3d(mesh_gt: o3d.geometry.TriangleMesh,
                         mesh_pred: o3d.geometry.TriangleMesh) -> dict:
    METRIC_THRESHOLD_3D = 0.05
    NUM_SAMPLE_POINTS = 200000
    VOXEL_SIZE = 1 / 64

    pcd_gt = mesh_gt.sample_points_uniformly(
        number_of_points=NUM_SAMPLE_POINTS)
    pcd_pred = mesh_pred.sample_points_uniformly(
        number_of_points=NUM_SAMPLE_POINTS)

    pcd_gt = o3d.t.geometry.PointCloud.from_legacy(pcd_gt)
    pcd_pred = o3d.t.geometry.PointCloud.from_legacy(pcd_pred)

    pcd_gt = pcd_gt.voxel_down_sample(VOXEL_SIZE)
    pcd_pred = pcd_pred.voxel_down_sample(VOXEL_SIZE)

    transformation = get_pcd_alignment_transformation(source=pcd_pred,
                                                      target=pcd_gt)
    pcd_pred = pcd_pred.transform(transformation)

    points_gt = pcd_gt.point.positions.numpy()
    points_pred = pcd_pred.point.positions.numpy()

    gt_points_kd_tree = scipy.spatial.cKDTree(points_pred)
    distances_pred_to_gt, _ = gt_points_kd_tree.query(points_gt)

    gt_points_kd_tree = scipy.spatial.cKDTree(points_gt)
    distances_gt_to_pred, _ = gt_points_kd_tree.query(points_pred)

    accuracy = np.mean(distances_gt_to_pred)
    completion = np.mean(distances_pred_to_gt)
    precision = np.mean((distances_gt_to_pred
                         < METRIC_THRESHOLD_3D).astype(float))
    recall = np.mean((distances_pred_to_gt
                      < METRIC_THRESHOLD_3D).astype(float))
    f1score = 2 * precision * recall / (precision + recall)

    metrics_3d = {
        'accuracy': float(accuracy),
        'completion': float(completion),
        'precision': float(precision),
        'recall': float(recall),
        'f1score': float(f1score),
    }

    return metrics_3d
