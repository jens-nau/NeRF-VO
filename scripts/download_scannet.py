import os
import png
import zlib
import numpy as np
import struct
import urllib.request
import imageio

from tqdm import tqdm

COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}
COMPRESSION_TYPE_DEPTH = {
    -1: 'unknown',
    0: 'raw_ushort',
    1: 'zlib_ushort',
    2: 'occi_ushort'
}
BASE_URL = 'http://kaldir.vc.in.tum.de/scannet/v1/scans/'


class RGBDFrame():

    def load(self, file_handle) -> None:
        self.camera_to_world = np.asarray(struct.unpack(
            'f' * 16, file_handle.read(16 * 4)),
                                          dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = b''.join(
            struct.unpack('c' * self.color_size_bytes,
                          file_handle.read(self.color_size_bytes)))
        self.depth_data = b''.join(
            struct.unpack('c' * self.depth_size_bytes,
                          file_handle.read(self.depth_size_bytes)))

    def decompress_depth(self, compression_type):
        if compression_type == 'zlib_ushort':
            return self.decompress_depth_zlib()
        else:
            raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == 'jpeg':
            return self.decompress_color_jpeg()
        else:
            raise

    def decompress_color_jpeg(self):
        return imageio.v2.imread(self.color_data)


class SensorData:

    def __init__(self, filename: str) -> None:
        self.version = 4
        self.load(filename=filename)

    def load(self, filename: str) -> None:
        with open(filename, 'rb') as file:
            version = struct.unpack('I', file.read(4))[0]
            assert self.version == version
            strlen = struct.unpack('Q', file.read(8))[0]
            self.sensor_name = b''.join(
                struct.unpack('c' * strlen, file.read(strlen)))
            self.intrinsic_color = np.asarray(struct.unpack(
                'f' * 16, file.read(16 * 4)),
                                              dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack(
                'f' * 16, file.read(16 * 4)),
                                              dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack(
                'f' * 16, file.read(16 * 4)),
                                              dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack(
                'f' * 16, file.read(16 * 4)),
                                              dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack(
                'i', file.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack(
                'i', file.read(4))[0]]
            self.color_width = struct.unpack('I', file.read(4))[0]
            self.color_height = struct.unpack('I', file.read(4))[0]
            self.depth_width = struct.unpack('I', file.read(4))[0]
            self.depth_height = struct.unpack('I', file.read(4))[0]
            self.depth_shift = struct.unpack('f', file.read(4))[0]
            num_frames = struct.unpack('Q', file.read(8))[0]
            self.frames = []
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(file)
                self.frames.append(frame)

    def export_frames_color(self, dir_output: str) -> None:
        os.makedirs(dir_output) if not os.path.exists(dir_output) else None
        print(f'Exporting {len(self.frames)} color franes to {dir_output}...')
        for index in range(0, len(self.frames)):
            frame_color = self.frames[index].decompress_color(
                self.color_compression_type)
            imageio.imwrite(f'{dir_output}/{index:06d}.jpg', frame_color)

    def export_frames_depth(self, dir_output: str) -> None:
        os.makedirs(dir_output) if not os.path.exists(dir_output) else None
        print(f'Exporting {len(self.frames)} depth franes to {dir_output}...')
        for index in range(0, len(self.frames)):
            depth_data = self.frames[index].decompress_depth(
                self.depth_compression_type)
            frame_depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
                self.depth_height, self.depth_width)
            with open(f'{dir_output}/{index:06d}.png', 'wb') as file:
                writer = png.Writer(width=frame_depth.shape[1],
                                    height=frame_depth.shape[0],
                                    bitdepth=16)
                frame_depth = frame_depth.reshape(
                    -1, frame_depth.shape[1]).tolist()
                writer.write(file, frame_depth)

    def save_matrix_to_file(self, matrix: np.array, filename: str) -> None:
        with open(filename, 'w') as file:
            for line in matrix:
                np.savetxt(file, line[np.newaxis], fmt='%f')

    def export_extrinsics(self, dir_output: str) -> None:
        os.makedirs(dir_output) if not os.path.exists(dir_output) else None
        print(
            f'Exporting {len(self.frames)} camera extrinsics to {dir_output}...'
        )
        for index in range(0, len(self.frames)):
            self.save_matrix_to_file(self.frames[index].camera_to_world,
                                     f'{dir_output}/{index:06d}.txt')

    def export_intrinsics(self, dir_output: str) -> None:
        os.makedirs(dir_output) if not os.path.exists(dir_output) else None
        print(f'Exporting camera intrinsics to {dir_output}...')
        self.save_matrix_to_file(self.intrinsic_color,
                                 f'{dir_output}/intrinsic_color.txt')
        self.save_matrix_to_file(self.extrinsic_color,
                                 f'{dir_output}/extrinsic_color.txt')
        self.save_matrix_to_file(self.intrinsic_depth,
                                 f'{dir_output}/intrinsic_depth.txt')
        self.save_matrix_to_file(self.extrinsic_depth,
                                 f'{dir_output}/extrinsic_depth.txt')


def main() -> None:
    dir_dataset = 'datasets/ScanNet'
    for scene_name in tqdm([
            'scene0000_00', 'scene0059_00', 'scene0106_00', 'scene0169_00',
            'scene0181_00', 'scene0207_00'
    ]):
        print(f'Processing {scene_name}...')
        os.makedirs(f'{dir_dataset}/{scene_name}') if not os.path.exists(
            f'{dir_dataset}/{scene_name}') else None
        print(f'Downloading sensor data...')
        urllib.request.urlretrieve(
            f'{BASE_URL}/{scene_name}/{scene_name}.sens',
            f'{dir_dataset}/{scene_name}/{scene_name}.sens')
        print(f'Reading sensor data...')
        sensor_data = SensorData(
            f'{dir_dataset}/{scene_name}/{scene_name}.sens')
        sensor_data.export_frames_color(
            dir_output=f'{dir_dataset}/{scene_name}/color')
        sensor_data.export_frames_depth(
            dir_output=f'{dir_dataset}/{scene_name}/depth')
        sensor_data.export_intrinsics(
            dir_output=f'{dir_dataset}/{scene_name}/intrinsics')
        sensor_data.export_extrinsics(
            dir_output=f'{dir_dataset}/{scene_name}/extrinsics')


if __name__ == '__main__':
    main()
