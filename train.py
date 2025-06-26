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
from paired_dataset import get_paired_volume_datasets, center_crop
from basemodel import Config
from model import ReconModel
from augment import augment, flip
import utils
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torch.nn.parallel

torch.cuda.empty_cache()

#configurazione delle variabili di ambiente
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUD_ALLOC_CONF"] = "max_split_size_mb:512"

class Prefetch(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = [i for i in tqdm.tqdm(dataset, leave=False)]  #il costruttore crea una lista che contiene tutti
                                                                     #gli elementi del dataset precaricati in memoria

    def __len__(self):  #ritorna la lunghezza del dataset pre caricato che corrisponde al n° di elementi del dataset originale
        return len(self.dataset)

    def __getitem__(self, ind):  #usato per ottenere un elemento dal dataset precaricato in base dell'indice ind
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
    #CONFIGURAZIONE
    cfg = Config()
    cfg.device = "cuda"  # Se hai una GPU
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



    #AMBIENTE
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    #IMPOSTAZIONE DEL patch_size
    if len(main_args.patch_size) == 1:
        cfg.patch_size = tuple([main_args.patch_size[0] for _ in range(main_args.num_recurrent)])
    else:
        cfg.patch_size = tuple(
            [main_args.patch_size[i % len(main_args.patch_size)] if i != main_args.num_recurrent - 1 else 1 for i in
             range(main_args.num_recurrent)]
        )  # Patch size is set as alternating 4,2,1.

    print(main_args)

    #CREAZIONE DELLE DIRECTORY DI LOG
    for path in [main_args.logdir, main_args.logdir + '/res', main_args.logdir + '/ckpt']:
        if not os.path.exists(path):
            os.mkdir(path)
            print('mkdir:', path)
    writer = torch.utils.tensorboard.SummaryWriter(main_args.logdir)

    #SETUP PER LA RIPRODUCIBILITA'
    print('loading model...')
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'
    # torch.use_deterministic_algorithms(True)
    seed = 14982321
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # SELEZIONE DEL DISPOSITIVO (GPU/CPU)
    # Verifica se ci sono due GPU disponibili
    if torch.cuda.device_count() >= 2:
        print("Using", torch.cuda.device_count(), "GPUs!")
        cfg.GPUs = 1
        device_ids = [0, 1]  # Specifica gli ID delle GPU da utilizzare
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Usa la prima GPU
        net = ReconModel(cfg=cfg).to(device)  # Inizializza e sposta il modello sulla GPU
        #net = torch.nn.DataParallel(net, device_ids=device_ids)  # Wrappa il modello con DataParallel per più GPU
    else:
        print("Two GPUs are required but not available. Using CPU!")
        cfg.GPUs = 0
        device = torch.device("cpu")  # Se non ci sono due GPU, usa la CPU
        net = ReconModel(cfg=cfg).to(device)  # Inizializza e sposta il modello sulla CPU

    cfg.device = device.type  # Salva il tipo di dispositivo (cuda o cpu)

    # Impostazioni aggiuntive (se necessario)
    batchsize_train = main_args.batch_size // 2
    iter_cnt = 0

    print('Training from scratch...')

    epoch_start = 0


    writer.add_text('date', repr(time.ctime()))
    writer.add_text('working dir', repr(os.getcwd()))
    writer.add_text('__file__', repr(os.path.abspath(__file__)))
    writer.add_text('commands', repr(sys.argv))
    writer.add_text('arguments', repr(main_args))
    writer.add_text('actual config', repr(cfg))

    print('loading data...')

    #CARICAMENTO DEI DATI: su CPU per poi essere trasferiti su GPU
    volumes_train = get_paired_volume_datasets(
        main_args.train, crop=int(main_args.shape * 1.1), q=0, protocals=main_args.protocals,
        basepath=main_args.basepath,
        exchange_modal=main_args.exchange_Modal
    )

    print("Test loaded")

    volumes_val = get_paired_volume_datasets(
        main_args.val, crop=cfg.shape, protocals=main_args.protocals, basepath=main_args.basepath,
        exchange_modal=main_args.exchange_Modal
    )


    # Concatenazione dei dataset
    slices_val = torch.utils.data.ConcatDataset(volumes_val)
    #print(f"La dimensione del dataset concatenato: {len(slices_val)}")


    #VISUALIZZAZIONE DURANTE IL TRAINING
    len_vis = 16  #16?
    col_vis = 4  #n° colonne per visualizzare le img o dati in un layout a griglia
    batch_vis = next(iter(torch.utils.data.DataLoader(slices_val, batch_size=8, shuffle=True)))
    batch_vis = [x.to(device, non_blocking=True) for x in batch_vis]
    batch_vis = [utils.complex_to_chan_dim(x) for x in batch_vis]

    print('training...')
    last_loss, last_ckpt, last_disp = 0, 0, 0
    # time_data, time_vis = 0, 0
    signal_earlystop = False
    iter_best = iter_cnt
    eval_psnr_best = None
    eval_ssim_best = None

    #CONFIGURA OTTIMIZZATORE E SCALER PER AMP
    optim_r = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=0)

    scalar = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    eval_psnr, eval_ssim = [], []

    time_start = time.time()

    #CICLO DI TRAINING
    for index_epoch in tqdm.trange(epoch_start, main_args.epoch, desc='epoch', leave=True):

        slices_train = torch.utils.data.ConcatDataset(volumes_train)  #concatena i dati in unico dataset

        if main_args.prefetch:  #se prefetch è abilitata...
            # carico i dati in RAM per velocizzare il processo
            slices_train = Prefetch(slices_train)
            slices_val = Prefetch(slices_val)

        #informazioni suil dataset: fornisce un'idea delle dimensioni del dataset che si sta utilizzando
        slices_train_len = str(len(slices_train))
        volumes_train_len = str(len(volumes_train))
        slices_val_len = str(len(slices_val))
        volumes_val_len = str(len(volumes_val))
        message = 'dataset: ' + slices_train_len + ' / '
        message = message + volumes_train_len + ' for training, '
        message = message + slices_val_len + ' / ' + volumes_val_len + ' for validation'
        print("Informazioni sul dataset: ", message)

        #creazione dei DataLoader: iteratori per caricare i dati in batch dal dataset durante addestramento e validazione
        loader_train = torch.utils.data.DataLoader(slices_train, batch_size=batchsize_train, shuffle=True,
                                                   num_workers=main_args.num_workers, pin_memory=True, drop_last=True,
                                                   )
        loader_val = torch.utils.data.DataLoader(
            slices_val, batch_size=8, shuffle=True,
            num_workers=main_args.num_workers, pin_memory=True, drop_last=True
        )

        scalar = GradScaler()
        #ADDESTRAMENTO DEL MODELLO
        print("Inizio addestramento")
        postfix_m = str(batchsize_train) + ': {n_fmt}/{total_fmt}' + '[{elapsed}<{remaining},{rate_fmt}]' + '{postfix}'
        tqdm_iter = tqdm.tqdm(loader_train, desc='iter',
                              bar_format=postfix_m, leave=False
                              )

        #decay del lr: a ogni terzo dell'addestramento il lr viene ridotto. aiuta il modello a convergere
        if index_epoch % (main_args.epoch // 3) == 0:
            for param_group in optim_r.param_groups:
                param_group['lr'] = param_group['lr'] * (0.5 ** (index_epoch // 100))

        if signal_earlystop:  #parametro usato per arrestrare il processo quando il modello smette di migliorare
            break
        #cicli nei batch di addestramento
        for batch in tqdm_iter:
            net.train()  #.model  imposta il modello in modalità di addestramento
            time_data = time.time() - time_start  #misura il tempo impiegato per caricare e preparare i dati

            iter_cnt += 1  #incremento il contatore delle iterazioni

            with torch.no_grad():  #disabilita il calcolo del gradiente
                batch = [x.to(device, non_blocking=True) for x in batch]  #sposta i dati del batch su GPU (o CPU) per
                #l'elaborazione
                batch = augment_funcs[main_args.aux_aug](batch)  #applica le trasformazioni di augmentazione
                batch = flip(batch, len(main_args.protocals))  #applica il flipping
                # batch = rotate(batch)
                batch = [center_crop(x, (cfg.shape, cfg.shape)) for x in batch] #eseguo il crop centrale per
                #ritagliare i dati di una dimensione specificata
                batch = [utils.complex_to_chan_dim(x) for x in batch]  #converte le dim complesse dei dati in canali
                print("Qui")
                #separati, rendendoli compatibili con la rete
            #abilita la precisione mista per ridurre l'uso della memoria e accelerare il training
            with autocast(enabled=cfg.use_amp):
                local_fidelities, loss_fidelity, loss_consistency, loss_ssim, loss_all = net(*batch)

                #esegue il forward pass del batch attraverso il modello (net) e calcola le componenti del loss
                #print(f"Loss value: {loss_all.mean().item()}, requires_grad: {loss_all.requires_grad}")

            optim_r.zero_grad()  #Resetta i gradienti dei parametri del modello prima della backpropagation per evitare
            #accumuli.
            scalar.scale(loss_all.mean()).backward()

            scalar.step(optim_r)
            scalar.update()
            torch.cuda.empty_cache()

            del batch

            print("batch elaborato")

            time_start = time.time()
            #SALVATAGGIO IMMAGINI DI RICOSTRUZIONE
            if iter_cnt % 500 == 0:  #ogni 500 iterazioni...
                last_disp = iter_cnt
                net.eval()  #...il codice passa in modalità valutazione
                with torch.no_grad():
                    net.test(*batch_vis)  #esegue il test sul batch di visualizzazione..
                    vis = net.get_vis('images')
                for name, val in vis['images'].items():
                    torchvision.utils.save_image(val,
                                                 main_args.logdir + '/res/' + '%010d_' % iter_cnt + name + '.jpg',
                                                 nrow=len_vis // col_vis, padding=10,
                                                 range=(0, 1), pad_value=0.5
                                                 )  #..per poi salvare
                del vis, name, val

            if iter_cnt % 30000 == 0:  #ogni 30000 iterazioni...
                last_ckpt = iter_cnt
                torch.save({'state_dict': net.state_dict(),
                            'config': cfg,
                            'epoch': index_epoch},
                           main_args.logdir + '/ckpt/ckpt_%010d.pt' % iter_cnt) #salva i parametri del modello, la
                #configurazione e l'epoca corrente di
                #un checkpoint

            time_vis = time.time() - time_start
            time_start = time.time()
            postfix = '[%d/%d/%d/%d]' % (iter_cnt, last_loss, last_disp, last_ckpt)
            if time_data >= 0.1:
                postfix += ' data %.1f' % time_data
            if time_vis >= 0.1:
                postfix += ' vis %.1f' % time_vis
            tqdm_iter.set_postfix_str(postfix)

        # ##################  validation  ########################
        net.eval()

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

            net.test(*batch)

            stat_loss.append(net.Eval)
            vis = net.get_vis('scalars')
            stat_eval.append(vis['scalars'])
            del batch

            time_start = time.time()
            # TODO: ho inizializzato postfix come una stringa vuota per evitare l'errore
            postfix = ""
            if time_data >= 0.1:
                postfix += ' data %.1f' % time_data
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
            torch.save({'state_dict': net.state_dict(),
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
                        type=str, default='D:/tesi rebecca/jCAN-main/ckpt',
                        help='log directory')
    parser.add_argument('--epoch', type=int, default=200, help='epochs to train')
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
                        default='D:/tesi rebecca/jCAN-main/single_data_TRAIN/data_TRAIN.csv',
                        help='path to csv file of training data')
    parser.add_argument('--val', default='D:/tesi rebecca/jCAN-main/single_data_VAL/data_VAL.csv',
                        type=str, help='path to csv file of validation data')
    parser.add_argument('--basepath',
                        default='D:/tesi rebecca/jCAN-main', type=str,
                        help='path to basepath of data')
    parser.add_argument('--coils', type=int, default=1, help='number of coils')
    parser.add_argument('--protocals', metavar='NAME', type=str, default=['CORPDFS_FBK'], nargs='*',
                        help='input modalities, first element is target, second is reference')
    parser.add_argument('--aux_aug', type=str, default='Rigid', choices=augment_funcs.keys(),
                        help='data augmentation aux image')
    parser.add_argument('--prefetch', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    import argparse
    import faulthandler

    parse_arguments()
    #faulthandler.enable()