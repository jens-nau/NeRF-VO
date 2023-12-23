import os
import torch
import argparse
import torch.multiprocessing as mp

from time import sleep

from nerf_vo.data.data_module import DataModule
from nerf_vo.tracking.tracking_module import TrackingModule
from nerf_vo.enhancement.enhancement_module import EnhancementModule
from nerf_vo.mapping.mapping_module import MappingModule
from nerf_vo.multiprocessing.logging_module import LoggingModule


def execute(args: argparse.Namespace):
    os.makedirs(args.dir_prediction +
                '/snapshots') if not os.path.exists(args.dir_prediction +
                                                    '/snapshots') else None
    os.makedirs(args.dir_prediction +
                '/matrices') if not os.path.exists(args.dir_prediction +
                                                   '/matrices') else None

    device = torch.device('cuda:0')

    if args.multithreading:
        from torch.multiprocessing import Queue
    else:
        from queue import Queue

    queue_data2tracking = Queue()
    queue_tracking2estimation = Queue()
    queue_estimation2mapping = Queue()
    if args.performance_tracking:
        logging_queue = Queue()
    else:
        logging_queue = None

    status_manager = mp.Manager()
    shared_variables = {
        'status': status_manager.dict(),
        'status_lock': status_manager.Lock(),
    }

    data_module = DataModule(name=args.dataset_name,
                             args=args,
                             multithreading=args.multithreading,
                             gradient_tracing=False,
                             device=device,
                             logging_queue=logging_queue,
                             shared_variables=shared_variables)
    tracking_module = TrackingModule(name=args.tracking_module,
                                     args=args,
                                     multithreading=args.multithreading,
                                     gradient_tracing=False,
                                     device=device,
                                     logging_queue=logging_queue,
                                     shared_variables=shared_variables)
    estimation_module = EnhancementModule(name=args.enhancement_module,
                                          args=args,
                                          multithreading=args.multithreading,
                                          gradient_tracing=False,
                                          device=device,
                                          logging_queue=logging_queue,
                                          shared_variables=shared_variables)
    mapping_module = MappingModule(name=args.mapping_module,
                                   args=args,
                                   multithreading=args.multithreading,
                                   gradient_tracing=True,
                                   device=device,
                                   logging_queue=logging_queue,
                                   shared_variables=shared_variables)
    logging_module = LoggingModule(name='logging',
                                   args=args,
                                   multithreading=args.multithreading,
                                   gradient_tracing=True,
                                   device=device,
                                   logging_queue=None,
                                   shared_variables=shared_variables)

    data_module.register_output_queue(output_queue=queue_data2tracking)
    tracking_module.register_input_queue(input_queue=queue_data2tracking)
    tracking_module.register_output_queue(
        output_queue=queue_tracking2estimation)
    estimation_module.register_input_queue(
        input_queue=queue_tracking2estimation)
    estimation_module.register_output_queue(
        output_queue=queue_estimation2mapping)
    mapping_module.register_input_queue(input_queue=queue_estimation2mapping)
    logging_module.register_input_queue(input_queue=logging_queue)

    if args.multithreading:
        data_thread = mp.Process(target=data_module.run, args=())
        tracking_thread = mp.Process(target=tracking_module.run, args=())
        estimation_thread = mp.Process(target=estimation_module.run, args=())
        logging_thread = mp.Process(target=logging_module.run, args=())

        data_thread.start()
        tracking_thread.start()
        estimation_thread.start()
        logging_thread.start()

        mapping_module.run()

        data_thread.terminate()
        tracking_thread.terminate()
        estimation_thread.terminate()
        if not logging_queue.empty():
            sleep(5)
        logging_thread.terminate()

    else:
        data_module.run()
        tracking_module.run()
        estimation_module.run()
        mapping_module.run()
        logging_module.run()

        while data_module.run() and tracking_module.run(
        ) and estimation_module.run() and mapping_module.run(
        ) and logging_module.run():
            continue

        while mapping_module.run() and logging_module.run():
            continue

    return mapping_module.method
