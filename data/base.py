
import os, sys
import numpy as np
import torch
import six
import json
import random
import time
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from utils import *
from itertools import combinations 
from tqdm import tqdm


class Trainer:
    
    def __iter__(self, *args, **kargs):
        return self.train.__iter__(*args, **kargs)
    
    @must_override
    def evaluate_model(self):
        pass
        
        
    @warn_not_override
    def _evaluate_during_train(self, model=None, trainer_target=None, args=None):
        pass
        
    def train_model(self, model=None, trainer_target=None, args=None):
        
        if model is None:
            model = self.model
        
        if model is not None:
            config = model.config
        else:
            config = None
            
        if config is not None and hasattr(config, 'warm_steps'):
            warm_steps = config.warm_steps
        else:
            warm_steps = 0
        
        if args is None:
            raise Exception('require args')
            
        trainer_source = self
        if trainer_target is None:
            trainer_target = self
        
        losses = []
        times = []
        decay_rate = args.decay_rate
        learning_rate = args.lr
        for i_epoch in range(args.max_epoches):
            print(f'epoch: {i_epoch} / {args.max_epoches}')
            global_steps = int(model.global_steps.data)

            if global_steps > args.max_steps:
                print(f"reach max_steps, stop training")
                break

            tic = time.time()

            if len(trainer_source.train) % args.batch_size == 0:
                total = len(trainer_source.train) / args.batch_size
            else:
                total = int(len(trainer_source.train) / args.batch_size) + 1
            with tqdm(total=total) as tbar:
                t_losses = [0]
                for i, batch in enumerate(trainer_source):
                    
                    # warm up
                    if global_steps < warm_steps:
                        _lr = learning_rate * (global_steps+1) / warm_steps
                        adjust_learning_rate(model.optimizer, lr=_lr)
                        if global_steps % 10 == 0:
                            print(f"warm up: learning rate was adjusted to {_lr}")
                    
                    loss = model.train_step(batch)['loss'].detach().cpu().numpy()
                    losses.append(loss)
                    toc = time.time()
                    times.append(toc - tic)

                    global_steps = int(model.global_steps.data)
                    if global_steps % 100 == 0:
                        print(f"g_step {global_steps}, step {i+1}, "
                            f"avg_time {sum(times)/len(times):.3f}, "
                            f"loss:{sum(losses)/len(losses):.4f}")
                        t_losses = losses
                        losses = []
                        times = []

                    tic = time.time()

                    if global_steps % 1000 == 0:
                        _lr = learning_rate/(1+decay_rate*global_steps/1000)
                        print(f"learning rate was adjusted to {_lr}")
                        adjust_learning_rate(model.optimizer, lr=_lr)

                    if global_steps % args.evaluate_interval == 0:
                        with open(args.log_path, 'a') as f:
                            f.write(f'epoch: {i_epoch} / {args.max_epoches}\tglobal_steps: {global_steps} loss: {sum(t_losses)}\n')
                        self._evaluate_during_train(model=model, trainer_target=trainer_target, args=args)

                    if global_steps >= args.max_steps:
                        break

                    tbar.set_postfix(loss="%.4lf"%(loss))
                    tbar.update(1)