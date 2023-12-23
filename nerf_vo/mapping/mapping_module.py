from nerf_vo.multiprocessing.process_module import ProcessModule


class MappingModule(ProcessModule):
    def initialize_module(self) -> None:
        self.process_name = 'mapping'
        self.is_receving_data = True
        self.last_received_data = 0
        if self.name == 'instant-ngp':
            from nerf_vo.mapping.instant_ngp import InstantNGP
            self.method = InstantNGP(args=self.args, device=self.device)
        elif self.name == 'nerfstudio':
            from nerf_vo.mapping.nerfstudio import Nerfstudio
            self.method = Nerfstudio(args=self.args, device=self.device)
        else:
            raise NotImplementedError
        super().initialize_module()

    def shut_down_module(self) -> None:
        if self.logging_queue is not None:
            self.logging_queue.put((self.process_name, 'shutdown', 0, 0))
        else:
            with self.shared_variables['status_lock']:
                self.shared_variables['status'][self.process_name] = 'shutdown'
        super().shut_down_module()

    def get_input(self) -> dict:
        try:
            return self.input_queue.get(block=False)
        except Exception as e:
            return None

    def push_output(self, output: dict) -> None:
        pass

    def step(self, input: dict) -> tuple:
        super().step(input=None)
        skip_step = False
        if input is None:
            if self.last_received_data < self.args.mapping_iterations / self.args.num_keyframes or not self.is_receving_data:
                self.method(input=input)
            else:
                skip_step = True
            self.last_received_data += 1
        else:
            self.method(input=input)
            self.last_received_data = 0

        if input is not None and input['last_frame']:
            with self.shared_variables['status_lock']:
                self.shared_variables['status']['enhancement'] = 'shutdown'
            self.is_receving_data = False
        if self.method.is_shut_down:
            self.shutdown = True
        return None, skip_step
