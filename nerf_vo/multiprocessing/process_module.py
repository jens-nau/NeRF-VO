import torch
import argparse

from time import sleep
from queue import Queue

from nerf_vo.multiprocessing.performance_tracker import PerformanceTracker


class ProcessModule:

    def __init__(self,
                 name: str,
                 args: argparse.Namespace,
                 multithreading: bool = True,
                 gradient_tracing: bool = False,
                 device: torch.device = torch.device('cuda:0'),
                 logging_queue: Queue = None,
                 shared_variables: dict = None) -> None:
        self.name = name
        self.args = args
        self.multithreading = multithreading
        self.gradient_tracing = gradient_tracing
        self.device = device
        self.logging_queue = logging_queue
        self.shared_variables = shared_variables

        self.process_name = 'generic'
        self.method = None
        self.is_initialized = False
        self.shutdown = False
        self.is_shut_down = False

        self.step_counter = 0
        self.total_runtime = 0.0

        self.input_queue = None
        self.output_queue = None

    def initialize_module(self) -> None:
        self.is_initialized = True
        with self.shared_variables['status_lock']:
            self.shared_variables['status'][
                self.process_name] = 'is_initialized'

    def shut_down_module(self) -> None:
        if self.logging_queue is not None:
            self.logging_queue.put(
                (self.process_name, 'runtime_total', None, self.total_runtime))
            self.logging_queue.put(
                (self.process_name, 'runtime_average', None,
                 self.total_runtime / (self.step_counter - 2)))
        if self.multithreading:
            while True:
                with self.shared_variables['status_lock']:
                    if self.shared_variables['status'][
                            self.process_name] == 'shutdown':
                        break
                sleep(1)
        with self.shared_variables['status_lock']:
            self.shared_variables['status'][self.process_name] = 'is_shut_down'
        self.is_shut_down = True
        torch.cuda.empty_cache()

    def register_input_queue(self, input_queue) -> None:
        self.input_queue = input_queue

    def register_output_queue(self, output_queue) -> None:
        self.output_queue = output_queue

    def get_input(self) -> dict:
        return self.input_queue.get()

    def push_output(self, output: dict) -> None:
        self.output_queue.put(output)

    def step(self, input: dict) -> None:
        if (self.method.is_initialized if self.method is not None else True):
            self.step_counter += 1

    def run(self) -> bool:
        if not self.is_initialized:
            self.initialize_module()
        while not self.is_shut_down:
            input = self.get_input()

            with PerformanceTracker(
                    process_name=self.process_name,
                    logging_queue=(self.logging_queue if
                                   (self.method.is_initialized if self.method
                                    is not None else True) else None),
                    step=self.step_counter) as performance_tracker:
                if not self.gradient_tracing:
                    with torch.no_grad():
                        output, skip_step = self.step(input=input)
                else:
                    output, skip_step = self.step(input=input)

            self.push_output(output=output)

            if skip_step:
                if (self.method.is_initialized
                        if self.method is not None else True):
                    self.step_counter -= 1
            else:
                performance_tracker.submit()
                if self.step_counter != 1 and self.step_counter != (int(
                    ((self.args.last_frame_index if self.args.last_frame_index
                      != -1 else 2000) - self.args.first_frame_index) /
                        self.args.frame_stride) if self.process_name in [
                            'data', 'tracking', 'estimation'
                        ] else self.args.mapping_iterations):
                    self.total_runtime += performance_tracker.runtime

            if self.shutdown:
                self.shut_down_module()

            if not self.multithreading:
                return True
        return False
