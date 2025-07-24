# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import sys

from dinov2.logging import setup_logging
from dinov2.train import get_args_parser as get_train_args_parser
from dinov2.run.submit import get_args_parser, submit_jobs


logger = logging.getLogger("dinov2")


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        from dinov2.train import main as train_main

        self._setup_args()
        train_main(self.args)

    def checkpoint(self):
        import submitit

        logger.info(f"Requeuing {self.args}")
        empty = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty)

    def _setup_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        self.args.output_dir = self.args.output_dir.replace("%j", str(job_env.job_id))
        logger.info(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        logger.info(f"Args: {self.args}")


def main():
    description = "DINOv2 training (single server mode)"
    train_args_parser = get_train_args_parser(add_help=False)
    parents = [train_args_parser]
    args_parser = get_args_parser(description=description, parents=parents)
    args = args_parser.parse_args()

    setup_logging()

    # Check if we want to use single server mode
    if hasattr(args, 'single_server') and args.single_server:
        # Run directly without Submitit but with proper multi-GPU setup
        import torch
        from dinov2.train import main as train_main
        
        logger.info("Running in single server mode (no SLURM)")
        
        # Set up distributed training for multi-GPU
        if torch.cuda.device_count() > 1:
            logger.info(f"Found {torch.cuda.device_count()} GPUs, setting up distributed training")
            
            # Set environment variables for distributed training
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
            
            # Launch distributed training using torch.multiprocessing
            import torch.multiprocessing as mp
            mp.spawn(run_distributed_training, args=(args,), nprocs=torch.cuda.device_count(), join=True)
        else:
            logger.info("Single GPU detected, running on GPU 0")
            train_main(args)
        return 0
    else:
        # Original Submitit behavior
        assert os.path.exists(args.config_file), "Configuration file does not exist!"
        submit_jobs(Trainer, args, name="dinov2:train")
        return 0


def run_distributed_training(rank, args):
    """Function to run distributed training on each GPU"""
    import torch
    from dinov2.train import main as train_main
    
    # Set the rank and local rank
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)

    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(rank)

    logger.info(f"Starting training on GPU {rank}")
    
    # Call the main training function
    train_main(args)


if __name__ == "__main__":
    sys.exit(main())
