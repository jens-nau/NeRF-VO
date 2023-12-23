# import wandb
import pandas as pd

from nerf_vo.multiprocessing.process_module import ProcessModule


class LoggingModule(ProcessModule):

    def initialize_module(self) -> None:
        self.process_name = 'logging'
        self.shutdown = False
        self.logs = {}
        # if self.multithreading:
        #     wandb.init(
        #         project=self.args.experiment,
        #         name=f'{(self.args.scene_name if self.args.comment == "" else self.args.comment)}_logging',
        #         dir=self.args.dir_prediction,
        #     )
        super().initialize_module()

    def shut_down_module(self) -> None:
        for key, value in self.logs.items():
            logs = pd.DataFrame(value)
            logs.to_csv(f'{self.args.dir_result}/runtime_{key}.csv',
                        index=False)
        # if self.multithreading:
        #     wandb.finish()
        with self.shared_variables['status_lock']:
            self.shared_variables['status']['mapping'] = 'shutdown'
            self.shared_variables['status']['logging'] = 'shutdown'
        super().shut_down_module()

    def get_input(self) -> list:
        events = []
        while True:
            try:
                events.append(self.input_queue.get(block=False))
            except Exception as e:
                break
        return events

    def push_output(self, output: dict) -> None:
        pass

    def step(self, input: list) -> tuple:
        for process_name, field, step, runtime in input:
            if field == 'shutdown':
                self.shutdown = True
            elif field == 'runtime':
                if process_name not in self.logs:
                    self.logs[process_name] = []
                self.logs[process_name].append({
                    'step': step,
                    'runtime': runtime
                })
            # else:
            #     wandb.log({f'{field}/{process_name}': runtime}, step=step)
        return None, False
