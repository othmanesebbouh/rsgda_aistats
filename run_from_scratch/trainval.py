import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import math
import itertools
import os, sys
import exp_configs
import shutil
import numpy as np
import torch.nn as nn
from src import models
from src import datasets
from src import optimizers
# from src import utils as ut
from src import metrics
from haven import haven_wizard as hw
import argparse

from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import default_collate

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
import shutil

import pprint


def trainval(exp_dict, savedir, args):
    # Set seed and device
    # ===================
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    seed = 42 + exp_dict['runs']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        device = 'cuda'
        torch.cuda.manual_seed_all(seed)
        assert torch.cuda.is_available(), 'cuda is not available please run with "-c 0"'
    else:
        device = 'cpu'

    print('Running on device: %s' % device)

    # Load Datasets
    # ==================
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     split='train',
                                     datadir=args.datadir)

    train_loader = DataLoader(train_set,
                              drop_last=True,
                              shuffle=True,
                              sampler=None,
                              batch_size=exp_dict["batch_size"])

    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                   split='val',
                                   datadir=args.datadir)

    # load the entire dataset as init of the adv input.
    # ==================
    adv_input_init = torch.load(args.datadir + "/MNIST/processed/training.pt")[0].float()
    adv_input_init = torchvision.transforms.Normalize((0.5,), (0.5,))(adv_input_init).reshape(-1, 1, 28, 28)

    # load the model
    # ==================
    model = models.get_model_robust(train_loader, exp_dict, device=device, adv_input_init=adv_input_init)
    del adv_input_init

    score_list_path = os.path.join(savedir, "score_list.pkl")

    # restart experiment
    score_list = []
    s_epoch = 0

    # Train and Val
    # ==============
    for epoch in range(s_epoch, exp_dict["max_epoch"]):
        # Set seed
        seed = epoch + exp_dict.get('runs', 0)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        train_loss_dict = model.val_on_dataset(train_set,
                                               metric=exp_dict["loss_func"], name='loss', train=True)
        val_acc_dict = model.val_on_dataset(val_set, metric=exp_dict["acc_func"], name='score')

        score_dict = {"epoch": epoch}
        score_dict.update(train_loss_dict)
        score_dict.update(val_acc_dict)
        if epoch % exp_dict["grad_phi_every"] == 0:
            grad_phi_dict = model.grad_phi(dataset=train_set, tol=exp_dict["tol_compute_phi"], 
                            check_every=exp_dict["phi_epochs_before_check"], max_epoch_phi=exp_dict["max_epoch_phi"])
        score_dict.update(grad_phi_dict)

        # Train one epoch
        model.train_on_loader(train_loader, epoch)

        # Add score_dict to score_list
        score_list += [score_dict]

        # Report and save
        print(pd.DataFrame(score_list).tail())
        save_pkl(score_list_path, score_list)
        print("Saved: %s" % savedir)


def save_pkl(fname, data):
    """Save data in pkl format."""
    # Save file
    with open(fname, "wb") as f:
        pickle.dump(data, f)
    os.rename(fname, fname)

if __name__ == "__main__":
    CUDA_LAUNCH_BLOCKING=1
    import exp_configs

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', required=True)
    parser.add_argument("-c", "--cuda", default=1, type=int)
    parser.add_argument("-j", "--job_scheduler", default=None)
    parser.add_argument("-p", "--python_binary_path", default=None)
    parser.add_argument("-v", "--view_results", default=None)
    args, others = parser.parse_known_args()

    # Get job configuration to launch experiments in the cluster
    job_config = None
    if os.path.exists('job_configs.py'):
        import job_configs

        job_config = job_configs.JOB_CONFIG

    # Run experiments either sequentially or in the cluster
    hw.run_wizard(func=trainval,
                  exp_groups=exp_configs.EXP_GROUPS,
                  job_config=job_config,
                  job_scheduler=args.job_scheduler,
                  python_binary_path=args.python_binary_path,
                  savedir_base=args.savedir_base,
                  use_threads=True,
                  results_fname=None,
                  args=args)
