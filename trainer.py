#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
from tqdm import tqdm

import numpy as np
import torch

import wandb
from utils.logger import print_log


class TrainConfig:
    def __init__(self, args, save_dir, lr, max_epoch, metric_min_better=True, early_stop=False, patience=3, grad_clip=None, anneal_base=1):
        self.save_dir = save_dir
        self.lr = lr
        self.max_epoch = max_epoch
        self.metric_min_better = metric_min_better
        self.patience = patience if early_stop else max_epoch + 1
        self.grad_clip = grad_clip
        self.anneal_base = anneal_base

        # record args
        self.args = str(args)

    def __str__(self):
        return str(self.__class__) + ': ' + str(self.__dict__)


class Trainer:
    def __init__(self, model, train_loader, valid_loader, config, cdr=None, fold=None, wandb=0):
        self.model = model
        self.config = config
        self.optimizer = self.get_optimizer()
        sched_config = self.get_scheduler(self.optimizer)
        if sched_config is None:
            sched_config = {
                'scheduler': None,
                'frequency': None
            }
        self.scheduler = sched_config['scheduler']
        self.sched_freq = sched_config['frequency']
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # distributed training
        self.local_rank = -1

        # log
        self.model_dir = os.path.join(self.config.save_dir, 'checkpoint')
        self.writer_buffer = {}

        # training process recording
        self.global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.last_valid_metric = None
        self.patience = config.patience

        self.init_step = (cdr - 1) * 10 * config.max_epoch + fold * config.max_epoch
        self.ex_name = 'CDR{0}_fold{1}'.format(cdr, fold)
        self.wandb = wandb


    @classmethod
    def to_device(cls, data, device):
        if isinstance(data, dict):
            for key in data:
                data[key] = cls.to_device(data[key], device)
        elif isinstance(data, list) or isinstance(data, tuple):
            res = [cls.to_device(item, device) for item in data]
            data = type(data)(res)
        elif hasattr(data, 'to'):
            data = data.to(device)
        return data

    def _is_main_proc(self):
        return self.local_rank == 0 or self.local_rank == -1

    def _train_epoch(self, device):
        if self.train_loader.sampler is not None and self.local_rank != -1:  # distributed
            self.train_loader.sampler.set_epoch(self.epoch)
        t_iter = tqdm(self.train_loader) if self._is_main_proc() else self.train_loader
        for batch in t_iter:
            batch = self.to_device(batch, device)
            loss = self.train_step(batch, self.global_step)
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            # try_catch_oom(self.optimizer.step)
            self.optimizer.step()
            if hasattr(t_iter, 'set_postfix'):
                t_iter.set_postfix(loss=loss.item())
            self.global_step += 1
            if self.sched_freq == 'batch':
                self.scheduler.step()
        if self.sched_freq == 'epoch':
            self.scheduler.step()
        # write training log
        mean_writer = {}
        for name in self.writer_buffer:
            value = np.mean(self.writer_buffer[name])
            mean_writer[name] = value
        if self.wandb:
            wandb.log(mean_writer, step=self.init_step + self.epoch)
        self.writer_buffer = {}

    def _valid_epoch(self, device):
        metric_arr = []
        self.model.eval()
        with torch.no_grad():
            t_iter = tqdm(self.valid_loader) if self._is_main_proc() else self.valid_loader
            for batch in t_iter:
                batch = self.to_device(batch, device)
                metric = self.valid_step(batch, self.valid_global_step)
                metric_arr.append(metric.cpu().item())
                self.valid_global_step += 1
        self.model.train()
        # judge
        ckpt_saved, save_path = False, None
        valid_metric = np.mean(metric_arr)
        if self._metric_better(valid_metric):
            self.patience = self.config.patience
            if self._is_main_proc():
                save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
                module_to_save = self.model.module if self.local_rank == 0 else self.model
                torch.save(module_to_save, save_path)
                torch.save(module_to_save, os.path.join(self.config.save_dir, f'best.ckpt'))
                ckpt_saved = True
            self.last_valid_metric = valid_metric
        else:
            self.patience -= 1
        # write validation log
        mean_writer = {}
        for name in self.writer_buffer:
            value = np.mean(self.writer_buffer[name])
            mean_writer[name] = value
        if self.wandb:
            wandb.log(mean_writer, step=self.init_step + self.epoch)
        self.writer_buffer = {}
        return ckpt_saved, save_path
    
    def _metric_better(self, new):
        old = self.last_valid_metric
        if old is None:
            return True
        if self.config.metric_min_better:
            return new < old
        else:
            return old < new

    def train(self, device_ids, local_rank):
        # set local rank
        self.local_rank = local_rank
        # init writer
        if self._is_main_proc():
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            with open(os.path.join(self.config.save_dir, 'train_config.json'), 'w') as fout:
                json.dump(self.config.__dict__, fout)
        # main device
        main_device_id = local_rank if local_rank != -1 else device_ids[0]
        device = torch.device('cpu' if main_device_id == -1 else f'cuda:{main_device_id}')
        self.model.to(device)
        if local_rank != -1:
            print_log(f'Using data parallel, local rank {local_rank}, all {device_ids}')
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank)
        else:
            print_log(f'training on {device_ids}')
        for _ in range(self.config.max_epoch):
            print_log(f'epoch{self.epoch} starts') if self._is_main_proc() else 1
            self._train_epoch(device)
            print_log(f'validating ...') if self._is_main_proc() else 1
            ckpt_saved, save_path = self._valid_epoch(device)
            if ckpt_saved:
                print_log(f'checkpoint saved to {save_path}')
            self.epoch += 1
            if self.patience <= 0:
                print(f'early stopping' + ('' if local_rank == -1 else f', local rank {local_rank}'))
                break
        print_log(f'finished training' + ('' if local_rank == -1 else f', local rank {local_rank}'))

    def log(self, name, value, step, val=False):
        if self._is_main_proc():
            if isinstance(value, torch.Tensor):
                value = value.cpu().item()
            if name not in self.writer_buffer:
                self.writer_buffer[name] = []
            self.writer_buffer[name].append(value)

    ########## Overload these functions below ##########
    # define optimizer
    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        return optimizer

    # scheduler example: linear. Return None if no scheduler is needed.
    def get_scheduler(self, optimizer):
        lam = lambda epoch: self.config.anneal_base ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)
        return {
            'scheduler': scheduler,
            'frequency': 'epoch' # or batch
        }

    # train step, note that batch should be dict/list/tuple or other objects with .to(device) attribute
    def train_step(self, batch, batch_idx):
        return self.share_forward(batch, batch_idx, 'train/' + self.ex_name)

    # validation step
    def valid_step(self, batch, batch_idx):
        return self.share_forward(batch, batch_idx, 'validation/' + self.ex_name, val=True)
    
    def share_forward(self, batch, batch_idx, _type, val=False):
        loss, snll, closs = self.model(
            batch['X'], batch['S'], batch['L'], batch['offsets']
        )
        ppl = snll.exp().item()
        self.log(f'Loss/{_type}', loss, batch_idx, val=val)
        self.log(f'SNLL/{_type}', snll, batch_idx, val=val)
        self.log(f'Closs/{_type}', closs, batch_idx, val=val)
        self.log(f'PPL/{_type}', ppl, batch_idx, val=val)
        return loss