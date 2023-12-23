from nerf_vo.data.eth3d_dataset import ETH3DDataset
from nerf_vo.data.replica_dataset import ReplicaDataset
from nerf_vo.data.scannet_dataset import ScanNetDataset
from nerf_vo.data.seven_scenes_dataset import SevenScenesDataset
from nerf_vo.data.tum_rgbd_dataset import TUMRGBDDataset
from nerf_vo.multiprocessing.process_module import ProcessModule


class DataModule(ProcessModule):

    def initialize_module(self) -> None:
        self.process_name = "data"
        self.frame_index = 0
        if self.name == "eth3d":
            self.method = ETH3DDataset(args=self.args)
        elif self.name == "replica":
            self.method = ReplicaDataset(args=self.args)
        elif self.name == "scannet":
            self.method = ScanNetDataset(args=self.args)
        elif self.name == "7-scenes":
            self.method = SevenScenesDataset(args=self.args)
        elif self.name == "tum-rgbd":
            self.method = TUMRGBDDataset(args=self.args)
        else:
            raise NotImplementedError
        super().initialize_module()

    def get_input(self) -> dict:
        return None

    def step(self, input: dict) -> tuple:
        super().step(input=None)
        output = self.method[self.frame_index]
        self.frame_index += 1
        if self.frame_index == len(self.method):
            self.shutdown = True
        return output, False
