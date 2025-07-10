#!/usr/bin/env python-3
"""
@author: sunkg
"""

import os
import os.path
import random
import statistics
import sys
import time
import torch
import numpy as np
import torch.utils.tensorboard
import torch.utils.data
import torchvision
import torchvision.utils
import tqdm
from model import ReconModel
from paired_dataset import get_paired_volume_datasets, center_crop
from basemodel import Config
from augment import augment, flip
import utils
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.parallel

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


class Prefetch(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = [i for i in tqdm.tqdm(dataset, leave=False)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        return self.dataset[ind]


def augment_none(batch):
    return batch


def augment_rigid(batch):
    return [augment(x, rigid=True, bspline=False) for x in batch]


def augment_b_spline(batch):
    return [augment(x, rigid=True, bspline=True) for x in batch]


augment_funcs = {'None': augment_none,
                 'Rigid': augment_rigid,
                 'BSpline': augment_b_spline}


def main(main_args):
    # setup
    cfg = Config()
    cfg.sparsity_ref = main_args.sparsity_ref
    cfg.sparsity_tar = main_args.sparsity_tar
    cfg.lr = main_args.lr
    cfg.shape = main_args.shape
    cfg.img_size = tuple([cfg.shape, cfg.shape])
    cfg.coils = main_args.coils
    cfg.mask = main_args.mask
    cfg.use_amp = main_args.use_amp
    cfg.num_heads = main_args.num_heads
    cfg.window_size = main_args.window_size
    cfg.mlp_ratio = main_args.mlp_ratio
    cfg.n_SC = main_args.n_SC
    cfg.num_recurrent = main_args.num_recurrent
    cfg.sens_chans = main_args.sens_chans
    cfg.sens_steps = main_args.sens_steps
    cfg.embed_dim = main_args.embed_dim
    cfg.lambda0 = main_args.lambda0
    cfg.lambda1 = main_args.lambda1
    cfg.lambda2 = main_args.lambda2
    cfg.lambda3 = main_args.lambda3
    cfg.GT = main_args.GT
    cfg.ds_ref = main_args.ds_ref
    cfg.protocals = main_args.protocals

    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    if len(main_args.patch_size) == 1:
        cfg.patch_size = tuple([main_args.patch_size[0] for _ in range(main_args.num_recurrent)])
    else:
        cfg.patch_size = tuple(
            [main_args.patch_size[i % len(main_args.patch_size)] if i != main_args.num_recurrent - 1 else 1 for i in
             range(main_args.num_recurrent)]
        )  # Patch size is set as alternating 4,2,1.

    print(main_args)

    for path in [main_args.logdir, main_args.logdir + '/res', main_args.logdir + '/ckpt']:
        if not os.path.exists(path):
            os.mkdir(path)
            print('mkdir:', path)
    writer = torch.utils.tensorboard.SummaryWriter(main_args.logdir)

    print('loading model...')
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'
    # torch.use_deterministic_algorithms(True)
    seed = 14982321
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    net = ReconModel(cfg=cfg)

#PARTE MOFIFICATA!!!!!!!!!!!!!
    if torch.cuda.device_count() >= 2:
        print("Using", torch.cuda.device_count(), "GPUs!")
        cfg.GPUs = 2
        device_ids = [0, 1]  # specifica gli ID delle GPU da utilizzare
        net = torch.nn.DataParallel(net, device_ids=device_ids)
    else:
        print("Two GPUs are required but not available. Using CPU!")
        cfg.GPUs = 0





  #FINE PARTE MODIFICATA!

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.device = device.type
    if cfg.device == 'cpu':
        cfg.GPUs = 0
    else:
        cfg.GPUs = 1

    batchsize_train = main_args.batch_size #NON MODIFICATO
    iter_cnt = 0 #NON MODIFICATO

    print('training from scratch...')  #NON MODIFICATO

    net = ReconModel(cfg=cfg) #NON MODIFICATO
    epoch_start = 0 #NON MODIFICATO

    #SEGUENTI DUE RIGHE AGGIUNTE IO!!
    if torch.cuda.device_count() >= 2:
        net = nn.DataParallel(net, device_ids=device_ids)
    #net = net.to(device) RIMETTIIIIIIIII



    writer.add_text('date', repr(time.ctime()))
    writer.add_text('working dir', repr(os.getcwd()))
    writer.add_text('__file__', repr(os.path.abspath(__file__)))
    writer.add_text('commands', repr(sys.argv))
    writer.add_text('arguments', repr(main_args))
    writer.add_text('actual config', repr(cfg))

    print('loading data...')

    volumes_train = get_paired_volume_datasets(
        main_args.train, crop=int(main_args.shape * 1.1), q=0, protocals=main_args.protocals,
        basepath=main_args.basepath,
        exchange_modal=main_args.exchange_Modal
    )

    volumes_val = get_paired_volume_datasets(
        main_args.val, crop=cfg.shape, protocals=main_args.protocals, basepath=main_args.basepath,
        exchange_modal=main_args.exchange_Modal
    )

    slices_val = torch.utils.data.ConcatDataset(volumes_val)
    # For visualization during training ####
    len_vis = 16  # era 16
    col_vis = 4
    batch_vis = next(iter(torch.utils.data.DataLoader(slices_val, batch_size=len_vis, shuffle=True)))
    batch_vis = [x.to(device, non_blocking=True) for x in batch_vis]
    batch_vis = [utils.complex_to_chan_dim(x) for x in batch_vis]

    print('training...')
    last_loss, last_ckpt, last_disp = 0, 0, 0
    # time_data, time_vis = 0, 0
    signal_earlystop = False
    iter_best = iter_cnt
    eval_psnr_best = None
    eval_ssim_best = None

    optim_r = torch.optim.AdamW(net.module.parameters(), lr=cfg.lr, weight_decay=0)

    scalar = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    eval_psnr, eval_ssim = [], []

    time_start = time.time()

    for index_epoch in tqdm.trange(epoch_start, main_args.epoch, desc='epoch', leave=True):

        slices_train = torch.utils.data.ConcatDataset(volumes_train)

        if main_args.prefetch:
            # load all data to RAM
            slices_train = Prefetch(slices_train)
            slices_val = Prefetch(slices_val)

        slices_train_len = str(len(slices_train))
        volumes_train_len = str(len(volumes_train))
        slices_val_len = str(len(slices_val))
        volumes_val_len = str(len(volumes_val))
        message = 'dataset: ' + slices_train_len + ' / '
        message = message + volumes_train_len + ' for training, '
        message = message + slices_val_len + ' / ' + volumes_val_len + ' for validation'
        print(message)

        loader_train = torch.utils.data.DataLoader(slices_train, batch_size=batchsize_train, shuffle=True,
                                                   num_workers=main_args.num_workers, pin_memory=True, drop_last=True
                                                   )
        loader_val = torch.utils.data.DataLoader(
            slices_val, batch_size=main_args.batch_size, shuffle=True,
            num_workers=main_args.num_workers, pin_memory=True, drop_last=True
        )

        # ##################  training ########################
        postfix_m = str(batchsize_train) + ': {n_fmt}/{total_fmt}' + '[{elapsed}<{remaining},{rate_fmt}]' + '{postfix}'
        tqdm_iter = tqdm.tqdm(loader_train, desc='iter',
                              bar_format=postfix_m, leave=False
                              )

        # ### learning rate decay ####
        if index_epoch % (main_args.epoch // 3) == 0:
            for param_group in optim_r.param_groups:
                param_group['lr'] = param_group['lr'] * (0.5 ** (index_epoch // 100))

        if signal_earlystop:
            break
        for batch in tqdm_iter:

            net.module.train()
            time_data = time.time() - time_start

            iter_cnt += 1
            with torch.no_grad():
                batch = [x.to(device, non_blocking=True) for x in batch]
                batch = augment_funcs[main_args.aux_aug](batch)
                batch = flip(batch, len(main_args.protocals))
                # batch = rotate(batch)
                batch = [center_crop(x, (cfg.shape, cfg.shape)) for x in batch]
                batch = [utils.complex_to_chan_dim(x) for x in batch]

            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                local_fidelities, loss_fidelity, loss_consistency, loss_ssim, loss_all = net(*batch)

            optim_r.zero_grad()
            scalar.scale(loss_all.mean()).backward()

            scalar.step(optim_r)
            scalar.update()

            del batch

            time_start = time.time()

            if iter_cnt % 5000 == 0:  # every 5000 iterations save recon images
                last_disp = iter_cnt
                net.module.eval()
                with torch.no_grad():
                    net.module.test(*batch_vis)
                    vis = net.module.get_vis('images')
                for name, val in vis['images'].items():
                    torchvision.utils.save_image(val,
                                                 main_args.logdir + '/res/' + '%010d_' % iter_cnt + name + '.jpg',
                                                 nrow=len_vis // col_vis, padding=10,
                                                 range=(0, 1), pad_value=0.5
                                                 )
                del vis, name, val

            if iter_cnt % 30000 == 0:  # every 30000 iterations save model param.
                last_ckpt = iter_cnt
                torch.save({'state_dict': net.module.state_dict(),
                            'config': cfg,
                            'epoch': index_epoch},
                           main_args.logdir + '/ckpt/ckpt_%010d.pt' % iter_cnt)

            time_vis = time.time() - time_start
            time_start = time.time()
            postfix = '[%d/%d/%d/%d]' % (iter_cnt, last_loss, last_disp, last_ckpt)
            if time_data >= 0.1:
                postfix += ' data %.1f' % time_data
            if time_vis >= 0.1:
                postfix += ' vis %.1f' % time_vis
            tqdm_iter.set_postfix_str(postfix)

        # ##################  validation  ########################
        net.module.eval()

        if torch.cuda.device_count() >= 2:
            net = torch.nn.DataParallel(net, device_ids=device_ids)

        postfix_n = '(val) {n_fmt}/{total_fmt}' + '[{elapsed}<{remaining},{rate_fmt}]' + '{postfix}'
        tqdm_iter = tqdm.tqdm(loader_val, desc='iter',
                              bar_format=str(main_args.batch_size) + postfix_n, leave=False)
        stat_eval = []
        stat_loss = []
        time_start = time.time()
        with torch.no_grad():
            for batch in tqdm_iter:
                time_data = time.time() - time_start
                batch = [x.to(device, non_blocking=True) for x in batch]
                batch = [utils.complex_to_chan_dim(x) for x in batch]

                #net.test(*batch) #RIG ORIGINALE
                net.module.test(*batch) #RIGA MODIFICATA:

                #stat_loss.append(net.Eval) RIGA ORIGINALE
                stat_loss.append(net.module.Eval)  #riga modificata
                vis = net.module.get_vis('scalars')
                #vis = net.get_vis('scalars') riga origjnale
                stat_eval.append(vis['scalars'])
                del batch

                time_start = time.time()
                postfix = None
                if time_data >= 0.1:
                    postfix = str(postfix) + ' data %.1f' % time_data
            vis = {key: statistics.mean([x[key] for x in stat_eval]) for key in stat_eval[0]}
            for name, val in vis.items():
                writer.add_scalar('val/' + name, val, iter_cnt)
            eval_psnr_current, eval_ssim_current = [(sum(i) / len(loader_val)) for i in zip(*stat_loss)]
            eval_psnr.append(eval_psnr_current)
            eval_ssim.append(eval_ssim_current)
            del vis

            np.save(main_args.logdir + '/PSNR', np.array(eval_psnr))
            np.save(main_args.logdir + '/SSIM', np.array(eval_ssim))

            if (eval_psnr_best is None) or (
                    (eval_psnr_current > eval_psnr_best) & (eval_ssim_current > eval_ssim_best)):
                eval_psnr_best = eval_psnr_current
                eval_ssim_best = eval_ssim_current
                iter_best = iter_cnt
                print('Current best iteration %d/%d:' % (iter_best, len(loader_train) * main_args.epoch),
                      f' PSNR: {eval_psnr_best:.2f}', f', SSIM: {eval_ssim_best:.4f}')
                torch.save({'state_dict': net.module.state_dict(),
                            'config': cfg,
                            'epoch': index_epoch},
                           main_args.logdir + '/ckpt/best.pt')  # save best model variant

            else:
                if iter_cnt >= main_args.early_stop + iter_best:
                    signal_earlystop = True
                    print('signal_earlystop set due to early_stop')

    print('reached end of training loop, and signal_earlystop is ' + str(signal_earlystop))
    writer.flush()
    writer.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description='CS with adaptive mask')
    parser.add_argument('--logdir', metavar='logdir',
                        type=str, default='D:/progetto python/log',
                        help='log directory')
    parser.add_argument('--epoch', type=int, default=300, help='epochs to train')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of threads for parallel preprocessing')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--early_stop', type=int, default=1000000, metavar='N',
                        help='stop training after val loss not going down for N iters')
    parser.add_argument('--n_SC', type=int, default=1, help='number of self-cross attention')
    parser.add_argument('--patch_size', type=tuple, default=tuple([4, 2, 1]), help='patch size in ViT')
    parser.add_argument('--lambda0', type=float, default=10, help='weight of the kspace loss')
    parser.add_argument('--lambda1', type=float, default=10,
                        help='weight of the consistency loss in K-space')
    parser.add_argument('--lambda2', type=float, default=1, help='weight of the SSIM loss')
    parser.add_argument('--lambda3', type=float, default=1e2, help='weight of the TV Loss')
    parser.add_argument('--embed_dim', type=int, default=32, help='dimension of embeddings in ViT')
    parser.add_argument('--shape', type=int, default=320, help='image shape')
    parser.add_argument('--num_heads', type=int, default=8, help='number of multiheads in ViT')
    parser.add_argument('--window_size', type=int, default=16, help='window size of the SwinTransformer')
    parser.add_argument('--num_recurrent', type=int, default=25, help='number of DCRBs')
    parser.add_argument('--mlp_ratio', type=int, default=32, help='ratio in MLP')
    parser.add_argument('--sens_chans', type=int, default=8,
                        help='number of channels in sensitivity network')
    parser.add_argument('--sens_steps', type=int, default=4,
                        help='number of steps in initial sensitivity network')
    parser.add_argument('--GT', type=bool, default=True, help='if there is GT, default is True')
    parser.add_argument('--exchange_Modal', type=bool, default=False,
                        help='exchange order of protocals for augmentation, default is False')
    parser.add_argument('--ds_ref', type=bool, default=True,
                        help='if use gradient map of reference image as input, default is True')
    parser.add_argument('--mask', metavar='type',
                        choices=['mask', 'taylor', 'lowpass', 'equispaced', 'loupe', 'random'],
                        type=str, default='equispaced', help='types of mask')
    parser.add_argument('--sparsity_ref', metavar='0-1', type=float, default=1,
                        help='sparsity of masks for reference modality')
    parser.add_argument('--sparsity_tar', metavar='0-1', type=float, default=0.25,
                        help='sparisity of masks for target modality')
    parser.add_argument('--train', type=str,
                        default='D:/progetto python/h5paths/train/percorsi_dei_file.csv',
                        help='path to csv file of training data')
    parser.add_argument('--val', default='D:/progetto python/h5paths/val/percorsi_dei_file.csv',
                        type=str, help='path to csv file of validation data')
    parser.add_argument('--basepath',
                        default='D:/progetto python/', type=str,
                        help='path to basepath of data')
    parser.add_argument('--coils', type=int, default=16, help='number of coils')
    parser.add_argument('--protocals', metavar='NAME', type=str,
                        default=['AXT2', 'AXFLAIR'], nargs='*',
                        help='input modalities, first element is target, second is reference')
    parser.add_argument('--aux_aug', type=str, default='Rigid', choices=augment_funcs.keys(),
                        help='data augmentation aux image')
    parser.add_argument('--prefetch', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    import argparse

    parse_arguments()
