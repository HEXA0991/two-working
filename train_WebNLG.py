#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from utils import *
from data import *
# from models.joint_models_original import JointModel
from models.joint_models import JointModel, JointModelMacroF1
from models.base import *

# torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(4)

cuda_device = 'cuda:0'

# In[3]:


import argparse

def none_or_str(value):
    if value == 'None':
        return None
    return value

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

parser = argparse.ArgumentParser(description='Arguments for training.')

#### Train
parser.add_argument('--model_class',
                    default='JointModel',
                    action='store',)

parser.add_argument('--model_read_ckpt',
                    default=None, type=none_or_str,
                    action='store',)

parser.add_argument('--model_write_ckpt',
                    default='./ckpts/micro_webnlg', type=none_or_str,
                    action='store',)

parser.add_argument('--pretrained_wv',
                    default='./wv/glove.6B.100d.webnlg.txt', type=none_or_str,
                    action='store',)

parser.add_argument('--dataset',
                    default='WebNLGfront',
                    action='store',)

parser.add_argument('--label_config',
                    default=None, type=none_or_str,
                    action='store',)

parser.add_argument('--batch_size',
                    default=32, type=int, # 12
                    action='store',)

parser.add_argument('--evaluate_interval',
                    default=500, type=int,    # 500
                    action='store',)

parser.add_argument('--max_steps',
                    default=int(1e9), type=int,
                    action='store')

parser.add_argument('--max_epoches',
                    default=299, type=int,
                    action='store')

parser.add_argument('--decay_rate',
                    default=0.05, type=float,
                    action='store')


#### Model Config
parser.add_argument('--token_emb_dim',
                    default=100, type=int,
                    action='store',)

parser.add_argument('--char_encoder',
                    default='lstm',
                    action='store',)

parser.add_argument('--char_emb_dim',
                    default=30, type=int,
                    action='store',)

parser.add_argument('--cased',
                    default=0, type=int,
                    action='store',)

parser.add_argument('--hidden_dim',
                    default=200, type=int,
                    action='store',)

parser.add_argument('--num_layers',
                    default=3, type=int,
                    action='store',)

parser.add_argument('--crf',
                    default=None, type=none_or_str,
                    action='store',)

parser.add_argument('--loss_reduction',
                    default='sum',
                    action='store',)

parser.add_argument('--maxlen',
                    default=100, type=int,
                    action='store',)

parser.add_argument('--dropout',
                    default=0.5, type=float,
                    action='store',)

parser.add_argument('--optimizer',
                    default='adam',
                    action='store',)

parser.add_argument('--lr',
                    default=0.001, type=float,
                    action='store',)

parser.add_argument('--vocab_size',
                    default=500000, type=int,
                    action='store',)

parser.add_argument('--vocab_file',
                    default=None, type=none_or_str,
                    action='store',)

parser.add_argument('--re_vocab_file',
                    default='/home/Bio/zhangshiqi/codes/two-working/datasets/WebNLG/rel2id.json', type=none_or_str,
                    action='store',)

parser.add_argument('--ner_tag_vocab_size',
                    default=3, type=int,    # (label * BI + O) 9conll04
                    action='store',)

parser.add_argument('--re_tag_vocab_size',
                    default=435, type=int,    # ((label + Dep_To) * fw\bw + O) 11conll04
                    action='store',)

parser.add_argument('--lm_emb_dim',
                    default=4096, type=int,
                    action='store',)

parser.add_argument('--lm_emb_path',
                    default='/home/Bio/zhangshiqi/codes/two-working/wv/webnlg_emb/', type=str,
                    action='store',)

parser.add_argument('--head_emb_dim',
                    default=768, type=int,
                    action='store',)

parser.add_argument('--tag_form',
                    default='iob2',
                    action='store',)

parser.add_argument('--warm_steps',
                    default=1000, type=int,
                    action='store',)

parser.add_argument('--grad_period',
                    default=1, type=int,
                    action='store',)

parser.add_argument('--device',
                    default=cuda_device, type=none_or_str,
                    action='store',)

parser.add_argument('--log_path',
                    default='./log_deprel_all_webnlg_micro.txt', type=none_or_str,
                    action='store',)


# In[4]:

args = parser.parse_args()

with open(args.log_path, 'w') as f:
    f.write(time.asctime(time.localtime(time.time())) + '\n')
    f.write(f'{args.model_class} \n')


# In[5]:


if args.device is not None and args.device != 'cpu':
    torch.cuda.set_device(args.device)
elif args.device is None:
    if torch.cuda.is_available():
        gpu_idx, gpu_mem = set_max_available_gpu()
        args.device = cuda_device
    else:
        args.device = "cpu"


# In[6]:


config = Config(**args.__dict__)
ModelClass = eval(args.model_class)
model = ModelClass(config)


# In[7]:


if args.model_read_ckpt:
    print(f"reading params from {args.model_read_ckpt}")
    model = model.load(args.model_read_ckpt)
    model.token_embedding.token_indexing.update_vocab = False
elif args.token_emb_dim > 0 and args.pretrained_wv:
    print(f"reading pretrained wv from {args.pretrained_wv}")
    model.token_embedding.load_pretrained(args.pretrained_wv, freeze=True)
    model.token_embedding.token_indexing.update_vocab = False


# In[8]:


print("reading data..")
Trainer = model.get_default_trainer_class()
flag = args.dataset
trainer = Trainer(
    model=model,
    train_path=f'./datasets/WebNLG/deprel_all/train.{flag}.deprel_all.json',
    test_path=f'./datasets/WebNLG/deprel_all/test.{flag}.deprel_all.json',
    valid_path=f'./datasets/WebNLG/deprel_all/valid.{flag}.deprel_all.json',
    label_config=args.label_config,
    batch_size=int(args.batch_size),
    tag_form=args.tag_form, num_workers=0,
)


# In[ ]:

# trainer.evaluate_model()


# %%capture cap
print("=== start training ===")
trainer.train_model(args=args)




