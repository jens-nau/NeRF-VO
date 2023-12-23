import os
import json


def load_camera_intrinsics(dir_dataset: str, dataset_name: str) -> dict:
    with open(os.path.dirname(dir_dataset) + "/camera_parameters.json",
              "r") as file:
        camera_parameters = json.load(file)[dataset_name]
    camera_intrinsics = {
        "height": camera_parameters["h"],
        "width": camera_parameters["w"],
        "fx": camera_parameters["fx"],
        "fy": camera_parameters["fy"],
        "cx": camera_parameters["cx"],
        "cy": camera_parameters["cy"],
        "depth_scale": camera_parameters["depth_scale"],
    }
    for key in ["k1", "k2", "k3", "p1", "p2"]:
        if key in camera_parameters:
            camera_intrinsics[key] = camera_parameters[key]
    return camera_intrinsics


def scale_camera_intrinsics(camera_intrinsics: dict, height: int,
                            width: int) -> dict:
    scale_factor_x = width / camera_intrinsics["width"]
    scale_factor_y = height / camera_intrinsics["height"]
    camera_intrinsics["height"] = height
    camera_intrinsics["width"] = width
    camera_intrinsics["fx"] *= scale_factor_x
    camera_intrinsics["fy"] *= scale_factor_y
    camera_intrinsics["cx"] *= scale_factor_x
    camera_intrinsics["cy"] *= scale_factor_y
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
