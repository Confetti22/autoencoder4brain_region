# Copyright (c) Ramy Mounir.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from lib.utils.file import checkdir
from lib.utils.tensorboard import get_writer, TBWriter
from lib.core.scheduler import cosine_scheduler
from lib.utils.distributed import MetricLogger
#for draw_boder_test
from lib.utils.augmentations import center_crop_3d
from lib.utils.compute_feature_map import contour_plt,TraverseDataset3d,get_feature_list
from torch.utils.data import Dataset,DataLoader

from glob import glob
import math
import numpy as np
import tifffile as tif
import sys
import re
from tqdm.auto import tqdm

import os

import torch

def get_three_slice(x):
    radius =int(x.shape[-1]//2)
    x_x = x[:,:,radius]
    x_y = x[:,radius,:]
    x_z = x[radius,:,:]
    return x_x, x_y, x_z

def unnormalize(img):
    clip_low = 96
    clip_high = 2672
    return img *(clip_high - clip_low) + clip_low

class Trainer:

    def __init__(self, args, cfg, train_loader,model, loss, optimizer,valid_loader = None):

        self.args = args
        self.cfg = cfg
        self.train_gen = train_loader
        self.valid_gen = valid_loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.fp16_scaler = torch.GradScaler('cuda') if cfg.TRAINER.fp16 else None
        self.best_acc = 0


        # === TB writers === #
        if self.args.main:	

            self.writer = get_writer(args)

            self.lr_sched_writer = TBWriter(self.writer, 'scalar', 'Schedules/Learning Rate')			
            self.train_loss_writer = TBWriter(self.writer, 'scalar', 'Train:Loss')
            self.train_acc_writer = TBWriter(self.writer, 'scalar', 'Train:acc')
            self.valid_loss_writer = TBWriter(self.writer, 'scalar', 'valid:Loss')
            self.valid_acc_writer = TBWriter(self.writer, 'scalar', 'valid:acc')


            checkdir("{}/weights/{}/".format(args.out, self.args.model), args.reset)


    def train_one_epoch(self, epoch, lr_schedule, save_recon_img_flag, MSE_loss:True):
        self.model.train()


        metric_logger = MetricLogger(delimiter="  ")
        header = 'Epoch: [{}/{}]'.format(epoch, self.cfg.TRAINER.epoch)

        for it, (input_data, labels) in enumerate(metric_logger.log_every(self.train_gen, 8, header)):

            # === Global Iteration === #
            it = len(self.train_gen) * epoch + it

            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]

            # === Inputs === #
            input_data, labels = input_data.cuda(non_blocking=True), labels.cuda(non_blocking=True)

            # === Forward pass === #
            if self.cfg.TRAINER.fp16:
                train_type=torch.float16
            else:
                train_type = torch.float32
            with torch.autocast('cuda',dtype=train_type):
                logits = self.model(input_data)
                loss = self.loss(logits, labels)
                acc = (logits.argmax(dim=-1) == labels).float().mean() 

            # Sanity Check
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)
            
            # === Backward pass === #
            self.model.zero_grad()
            # for mix precision backward propogation
            if self.cfg.TRAINER.fp16:
                self.fp16_scaler.scale(loss).backward()
                self.fp16_scaler.step(self.optimizer)
                self.fp16_scaler.update()
            
            loss.backward()
            
            self.optimizer.step()


            # === Logging === #
            torch.cuda.synchronize()
            metric_logger.update(train_loss=loss.item())
            metric_logger.update(train_acc = acc )

            if self.args.main:
                self.train_loss_writer(metric_logger.meters['train_loss'].value, it)
                self.train_acc_writer(metric_logger.meters['train_acc'].value, it)
                self.lr_sched_writer(self.optimizer.param_groups[0]["lr"], it)


        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)


    def fit(self):

        # === Resume === #
        self.load_if_available()

        # === Schedules === #
        if self.cfg.SOLVER.LR_SCHEDULER_NAME =='cosine':
            lr_schedule = cosine_scheduler(
                        base_value = self.args.lr_start * (self.cfg.DATASET.batch_per_gpu * self.args.world_size) / 256.,
                        final_value = self.args.lr_end,
                        epochs = self.cfg.TRAINER.epoch,
                        niter_per_ep = len(self.train_gen),
                        warmup_epochs= self.args.lr_warmup,
                        )           

        # === training loop === #
        for epoch in range(self.start_epoch, self.cfg.TRAINER.epoch):

            self.train_gen.sampler.set_epoch(epoch)


            save_recon_img_flag = ( (epoch+1) %self.cfg.TRAINER.save_every==0)
            self.train_one_epoch(epoch, lr_schedule,save_recon_img_flag,MSE_loss=True)

            # === eval and save model === #
            if self.args.main and  ( (epoch+1)% self.cfg.TRAINER.save_every == 0  or epoch == 0 ):
                if self.valid_gen :
                    self.valid( it = epoch*len(self.train_gen),epoch=epoch )
                self.save(epoch)
            

    def load_if_available(self):

        ckpts = sorted(glob(f'{self.args.out}/weights/{self.args.model}/Epoch_*.pth'))

        if len(ckpts) >0:
            ckpts = sorted(
                    ckpts,
                    key=lambda x: int(re.search(r'Epoch_(\d+)', os.path.basename(x)).group(1))
                    )
            ckpt = torch.load(ckpts[-1], map_location='cpu')
            self.start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            if self.cfg.TRAINER.fp16: self.fp16_scaler.load_state_dict(ckpt['fp16_scaler'])
            print("Loaded ckpt: ", ckpts[-1])

        else:
            self.start_epoch = 0
            print("Starting from scratch")
    
    def valid(self,it,epoch):
        self.model.eval()
        valid_loss = []
        valid_accs = []
        for batch in tqdm (self.valid_gen):
            imgs, labels= batch
            imgs = imgs.to('cuda')
            labels = labels.to('cuda')

            with torch.no_grad():
                logits = self.model(imgs)
            loss = self.loss(logits,labels)
            valid_loss.append(loss.item())
            
            acc = (logits.argmax(dim = -1) == labels).float().mean()
            valid_accs.append(acc)
        
        valid_loss = sum(valid_loss)/len(valid_loss)
        valid_acc = sum(valid_accs)/len(valid_accs)

        
        torch.cuda.synchronize()

        self.valid_loss_writer(valid_loss, it)
        self.valid_acc_writer(valid_acc, it)

                # update logs
        if valid_acc > self.best_acc:
            print(f"[ Valid | {epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            print(f"[ Valid | {epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        
        #probability_border drawing test


        #test on probability map of class one

        if self.cfg.DATASET.e5:
            draw_img_pth = "/share/home/shiqiz/data/t1779/draw_boder_test/0002.tif"
        else:
            draw_img_pth = "/home/confetti/mnt/data/processed/t1779/draw_boder_test/0002.tif"
        win_size =self.cfg.DATASET.input_size[0]

        draw_border_img = tif.imread(draw_img_pth)
        draw_border_img = center_crop_3d(draw_border_img,crop_shape=(win_size,*draw_border_img.shape[1:]))
        draw_border_dataset = TraverseDataset3d(draw_border_img,stride=4,win_size=win_size,net_input_shape=win_size)  
        print(f"len of dataset is {len(draw_border_dataset)}")
        border_draw_loader = DataLoader(draw_border_dataset,24,shuffle=False,drop_last=False)

        feats_lst = get_feature_list('cuda',self.model,border_draw_loader,'fc',save_path=None,apply_softmax=True)
        sample_shape = draw_border_dataset.sample_shape[1:]
        feats_map = feats_lst.reshape(*sample_shape,-1)
        pro_map1=feats_map[:,:,0]
        contour_plt(pro_map1, 'P1', writer=self.writer, step=it)


        # save models
        if valid_acc > self.best_acc:
            print(f"Best model found at it {it}, saving model")
            self.save(epoch,comment='best')
            self.best_acc = valid_acc

        


    def save(self, epoch,comment =''):

        if self.cfg.TRAINER.fp16:
            state = dict(epoch=epoch+1, 
                            model=self.model.state_dict(), 
                            optimizer=self.optimizer.state_dict(), 
                            fp16_scaler = self.fp16_scaler.state_dict(),
                            args = self.args
                        )
        else:
            state = dict(epoch=epoch+1, 
                            model=self.model.state_dict(), 
                            optimizer=self.optimizer.state_dict(),
                            args = self.args
                        )

        torch.save(state, f"{self.args.out}/weights/{self.args.model}/Epoch_{str(epoch +1 ).zfill(3)}{comment}.pth".format(self.args.out, self.args.model, str(epoch+1).zfill(3) ))
