import cv2
import argparse

from abc import abstractmethod
from tqdm import tqdm
from torch.utils.data.dataset import Dataset

from nerf_vo.data.data_utils import load_camera_intrinsics, scale_camera_intrinsics


class BaseDataset(Dataset):

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.dir_dataset = args.dir_dataset
        self.first_frame_index = args.first_frame_index
        self.last_frame_index = args.last_frame_index
        self.stride = args.frame_stride
        self.height = args.frame_height
        self.width = args.frame_width
        self._load_dataset()
        self.cache = self._cache_dataset() if args.cache_dataset else None
        self.tqdm = tqdm(total=self.__len__())
        self.is_initialized = True

    def _load_dataset(self) -> None:
        self.files_color = self._load_files_color(
        )[self.first_frame_index:self.last_frame_index:self.stride]
        self.camera_intrinsics = scale_camera_intrinsics(
            self._load_camera_intrinsics(),
            height=self.height,
            width=self.width)

    @abstractmethod
    def _load_files_color(self) -> list:
        raise NotImplementedError

    def _load_camera_intrinsics(self) -> dict:
        return load_camera_intrinsics(dir_dataset=self.dir_dataset,
                                      dataset_name=self.args.dataset_name)

    def _cache_dataset(self) -> None:
        return [
            self._get_frame(frame_index=frame_index)
            for frame_index in tqdm(range(len(self)))
        ]

    def _get_frame(self, frame_index: int) -> dict:
        file_color = self.files_color[frame_index]
        frame_color = cv2.resize(
            cv2.cvtColor(cv2.imread(file_color), cv2.COLOR_BGR2RGB),
            (self.width, self.height))

        return {
            'frame_index': frame_index,
            'camera_intrinsics': self.camera_intrinsics,
            'frame_color': frame_color,
            'last_frame': (frame_index >= self.__len__() - 1),
        }

    def __len__(self) -> int:
        return len(self.files_color)

    def __getitem__(self, frame_index) -> dict:
        self.tqdm.update(1)
        return (self._get_frame(frame_index=frame_index)
                if self.cache is None else self.cache[frame_index])
