from nerf_vo.multiprocessing.process_module import ProcessModule


class TrackingModule(ProcessModule):

    def initialize_module(self) -> None:
        self.process_name = 'tracking'
        if self.name == 'droid-slam':
            from nerf_vo.tracking.droid_slam import DROIDSLAM
            self.method = DROIDSLAM(args=self.args, device=self.device)
        elif self.name == 'dpvo':
            from nerf_vo.tracking.dpvo import DPVOHandler
            self.method = DPVOHandler(args=self.args, device=self.device)
        else:
            raise NotImplementedError
        super().initialize_module()

    def step(self, input: dict) -> tuple:
        super().step(input=None)
        output = self.method(input=input)
        if input['last_frame']:
            with self.shared_variables['status_lock']:
                self.shared_variables['status']['data'] = 'shutdown'
        if self.method.is_shut_down:
            self.shutdown = True
        return output, False
