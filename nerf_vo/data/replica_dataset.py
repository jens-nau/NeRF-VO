import os

from nerf_vo.data.base_dataset import BaseDataset


class ReplicaDataset(BaseDataset):

    def _load_files_color(self) -> list:
        return sorted([
            os.path.join(self.dir_dataset + '/results', file)
            for file in os.listdir(self.dir_dataset + '/results')
            if file.endswith('.jpg')
        ])
