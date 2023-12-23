import os

from nerf_vo.data.base_dataset import BaseDataset


class SevenScenesDataset(BaseDataset):

    def _load_files_color(self) -> list:
        return sorted([
            os.path.join(self.dir_dataset + '/seq-01', file)
            for file in os.listdir(self.dir_dataset + '/seq-01')
            if file.endswith('color.png')
        ])
