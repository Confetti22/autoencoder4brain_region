import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import  DataLoader
from torch.utils.data.distributed import DistributedSampler

import argparse
from lib.utils.distributed import init_dist_node, init_dist_gpu, get_shared_folder


import submitit, random, sys
from pathlib import Path


def parse_args():

    parser = argparse.ArgumentParser(description='Template')

    # === GENERAL === #
    parser.add_argument('-reset', action='store_true',
                                            help='Reset saved model logs and weights')
    parser.add_argument('-tb', action='store_true',
                                            help='Start TensorBoard')
    parser.add_argument('-gpus', type=str, default="0",
                                            help='GPUs list, only works if not on slurm')
    parser.add_argument('-cfg', type =str,help='Configuration file',
                        default='config/brain_region_unet.yaml')

    # === Trainer === #



    # === SLURM === #
    parser.add_argument('-slurm', action='store_true',
                                            help='Submit with slurm')
    parser.add_argument('-slurm_ngpus', type=int, default = 8,
                                            help='num of gpus per node')
    parser.add_argument('-slurm_nnodes', type=int, default = 2,
                                            help='number of nodes')
    parser.add_argument('-slurm_nodelist', default = None,
                                            help='slurm nodeslist. i.e. "GPU17,GPU18"')
    parser.add_argument('-slurm_partition', type=str, default = "general",
                                            help='slurm partition')
    parser.add_argument('-slurm_timeout', type=int, default = 2800,
                                            help='slurm timeout minimum, reduce if running on the "Quick" partition')


    args = parser.parse_args()

    # cmdline parameters will overwrite the CFG parameters

    # === Read CFG File === #
    if args.cfg:



        with open(args.cfg, 'r') as f:
            import yaml
            yml = yaml.safe_load(f)

        # update values from cfg file only if not passed in cmdline
        cmd = [c[1:] for c in sys.argv if c[0]=='-']
        for k,v in yml.items():
            if k not in cmd:
                args.__dict__[k] = v

    return args


class SLURM_Trainer(object):
    def __init__(self, args,cfg):
        self.args = args
        self.cfg=cfg

    def __call__(self):

        init_dist_node(self.args)
        train(None, self.args, self.cfg)


from config import load_cfg
import shutil
import os
def main():

    args = parse_args()
    args.port = random.randint(49152,65535)

    cfg=load_cfg(args)

    cfg_save_path = f"{cfg.OUT}/logs/{cfg.EXP_NAME}"
    os.makedirs(cfg_save_path,exist_ok=True)
    shutil.copy2(args.cfg,f"{cfg_save_path}/cfg.yaml")
    print(f"config has been saved")


    args.model = cfg.EXP_NAME
    args.out = cfg.OUT
    args.optimizer = cfg.SOLVER.NAME
    args.lr_start = cfg.SOLVER.LR_START
    args.lr_end = cfg.SOLVER.LR_END
    args.lr_warmup = cfg.SOLVER.LR_WARMUP
    
    if args.slurm:

        # Almost copy-paste from https://github.com/facebookresearch/deit/blob/main/run_with_submitit.py
        args.output_dir = get_shared_folder(args) / "%j"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

        executor.update_parameters(
            mem_gb=128*args.slurm_ngpus,
            gpus_per_node=args.slurm_ngpus,
            tasks_per_node=args.slurm_ngpus,
            cpus_per_task=2,
            nodes=args.slurm_nnodes,
            timeout_min=2800,
            slurm_partition=args.slurm_partition
        )

        if args.slurm_nodelist:
            executor.update_parameters(slurm_additional_parameters = {"nodelist": f'{args.slurm_nodelist}' })

        executor.update_parameters(name=cfg.EXP_NAME)
        trainer = SLURM_Trainer(args,cfg)
        job = executor.submit(trainer)
        print(f"Submitted job_id: {job.job_id}")


    else:
        init_dist_node(args)
        mp.spawn(train, args=(args,cfg), nprocs=args.ngpus_per_node)
	

def train(gpu, args, cfg):


    # === SET ENV === #
    init_dist_gpu(gpu, args)
    
    # === DATA === #
    get_train_dataset = getattr(__import__("lib.datasets.{}".format(cfg.DATASET.name), fromlist=["get_dataset"]), "get_dataset")
    # from lib.datasets.visor_3d_dataset import get_dataset
    train_dataset = get_train_dataset(cfg)

    train_sampler = DistributedSampler(train_dataset, shuffle=cfg.DATASET.shuffle, num_replicas = args.world_size, rank = args.rank, seed = 31)
    train_loader = DataLoader(dataset=train_dataset, 
                            sampler = train_sampler,
                            batch_size=cfg.DATASET.batch_per_gpu, 
                            num_workers= cfg.DATASET.num_workers,
                            pin_memory = True,
                            drop_last = True
                            )
    get_valid_dataset = getattr(__import__("lib.datasets.{}".format(cfg.DATASET.name), fromlist=["get_valid_dataset"]), "get_valid_dataset")
    valid_dataset  = get_valid_dataset(cfg)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False, num_replicas = args.world_size, rank = args.rank, seed = 31)
    valid_loader = DataLoader(dataset=valid_dataset, 
                            sampler = valid_sampler,
                            batch_size= 6, 
                            num_workers= cfg.DATASET.num_workers,
                            pin_memory = True,
                            drop_last = True
                            )
    
    print(f"Data loaded")

    # === MODEL === #
    from lib.arch.autoencoder import build_autoencoder_model
    from torchsummary import summary

    model = build_autoencoder_model(cfg)
    model=model.cuda(args.gpu)
    model.train()

    
    #print out model info
    print(model)
    summary(model,(1,128,128,128))

    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model) #group_norm did not require to sync, group_norm is preferred when batch_size is small
    model = nn.parallel.DistributedDataParallel(model, device_ids= [args.gpu],find_unused_parameters=True)

    # === LOSS === #
    get_loss = getattr(__import__("lib.loss.{}".format(cfg.LOSS.name), fromlist=["get_loss"]), "get_loss")
    loss = get_loss(cfg).cuda(args.gpu)
    
    #reconsturct the preprocess image in range [0-1] with bce loss

    # === OPTIMIZER === #
    from lib.core.optimizer import get_optimizer
    optimizer = get_optimizer(model, cfg.SOLVER)

    # === TRAINING === #
    Trainer = getattr(__import__("lib.trainers.{}".format(cfg.TRAINER.name), fromlist=["Trainer"]), "Trainer")
    Trainer(args, cfg,train_loader,valid_loader, model, loss, optimizer).fit()


if __name__ == "__main__":
    main()
