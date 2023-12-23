from time import time
from queue import Queue


class PerformanceTracker:

    def __init__(self, process_name: str, logging_queue: Queue,
                 step: int) -> None:
        self.process_name = process_name
        self.logging_queue = logging_queue
        self.step = step
        self.start_time = 0.0
        self.runtime = 0.0

    def __enter__(self):
        self.start_time = time()
        return self

    def __exit__(self, *args) -> None:
        self.runtime = time() - self.start_time

    def submit(self):
        if self.logging_queue is not None:
            self.logging_queue.put(
                (self.process_name, 'runtime', self.step, self.runtime))
