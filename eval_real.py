#!/usr/bin/env python3
from tqdm import tqdm
import json
import os
import os.path
import statistics

import nibabel as nib
import numpy as np
import torch
import torchvision.utils
from torchio.transforms import (
    RandomAffine,
    OneOf,
    Compose,
)

from augment import augment_eval
from model import ReconModel
from paired_dataset import get_paired_volume_datasets, center_crop


def augment_aux(batch, factor=1):
    assert factor > 0  #controlla l'entità della trasformazione geometrica
    img_full, img_aux = batch
    _, grid = augment_eval(img_aux, rigid=True, bspline=True)  #genera una griglia di deformazione che rappresenta una
                                                               #mappa di spostamento dei pixel dell'immagine
    identity = np.array([[[1, 0, 0], [0, 1, 0]]])  #matrice I che rappresenta una trasformazione affine standard (senza spostamento)
    identity = identity * np.ones((img_aux.shape[0], 1, 1))
    identity = torch.as_tensor(identity, dtype=img_aux.abs().dtype).to(img_aux.device, non_blocking=True)
    identity = torch.nn.functional.affine_grid(identity, size=img_aux.shape, align_corners=False)
    offset = grid - identity
    grid = identity + offset * factor  #in questo modo si crea una griglia trasformata
    img_aux, _ = augment_eval(img_aux, rigid=False, bspline=False, grid=grid)  #la griglia modificata viene usata per
                                                                               #applicare la trasformazione all'immagine
                                                                               #ausiliaria producendo una nuova img
    return img_full, img_aux   #viene restituita la sua immagine e l'immagine modificata


def augment3_d(coil_img,                #immagine delle coil
               degree=(0, 0.01, 0.01),  #angolo di rotazione in radianti
               translation=(0.05, 0, 0)): #traslazione in mm
    #converto i due parametri forniti come tuple in liste
    degree = list(degree)
    translation = list(translation)
    img = torch.linalg.vector_norm(coil_img, ord=2, dim=1, keepdim=True)  #normalizzazione L2 dell'immagine lungo il
                                                                          #canale delle coil
    sm = coil_img / img  #L'immagine viene normalizzata dividendo ogni voxel per la sua norma L2, creando così una
                         #rappresentazione dell'immagine con intensità normalizzate
    img = img.permute(1, 0, 2, 3)  # [1, slices, 320,320]  permutazione dell'ordine delle dimensioni dell'immagine
    transform = Compose([OneOf({RandomAffine(degrees=degree, translation=translation, scales=0), })])

    img_rigid = transform(img)  #trasformo l'immagine permutata (rotazione e traslazione)
    img_rigid = img_rigid.permute(1, 0, 2, 3)  # [1, slices, 320,320]  riporto l'immagine alla forma originale

    coil_img_rigid = sm * img_rigid  #l'immagine trasformata viene combinata con l'immagine normalizzata per ricostruire
                                     #l'immagine con le coil mantenendo le proprietà con la normalizzaziona, ma con
                                     #trasformazione affine applicata

    return coil_img_rigid, img_rigid


def main(args):
    affine = np.eye(4) * [0.7, -0.7, -5, 1]  #matrice di trasformazione affine per salvare le immagini in formato NIfTI.
                                             #è usata per la trasformazione geometrica delle immagini salvate

    print(args)  #stampa gli argomenti passati alla funzione "main" utili per debug e verifica

    #creazione della directory di salvataggio
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    #confugrazione del dispositivo: usa la GPU se disponibile, altrimenti la CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #caricamento del modello
    if os.path.isfile(args.model_path) or os.path.isdir(args.model_path):
        ckpt = torch.load(args.model_path)
        cfg = ckpt['config']
        net = ReconModel(cfg=cfg)
        net.load_state_dict(ckpt['state_dict'])
        print('load ckpt from:', args.model_path)
    else:
        raise FileNotFoundError

    #configurazione del modello
    net.use_amp = False
    cfg = net.cfg
    net.GT = args.GT

    #caricamento del dataset
    volumes = get_paired_volume_datasets(args.val, crop=cfg.shape, protocals=args.protocals, basepath=args.basepath)

    #preparazione per la valutazione
    net.eval()  #imposta il modello in modalità di valutazione, disabilitando il dropout e le altre operazioni di addestramento

    stat_eval = []  #lista per memorizzare le statistiche di valutazione
    #liste per memorizzare valori di PSNR e SSIM per le immagini ricostruite e per quelle originali
    psnr, ssim = [], []
    psnr_raw, ssim_raw = [], []
    col_vis = 4  #n° colonne per la visualizzazione delle immagini
    total = sum([param.nelement() for param in net.parameters()]) #Calcola e stampa la dimensione totale del modello in
                                                                  #milioni di parametri
    print('Network size is %.2fM' % (total / 1e6))

    #loop di valutazione per ogni volume
    for i, volume in enumerate(tqdm(volumes, desc="Validazione", unit="volume")):

        with torch.no_grad():
            range_list = range(len(volume))
            if (args.aux_aug > 0) & (len(volume) > 1):
                volume_ref = torch.from_numpy(np.array(volume)[:, 1, ...])
                volume_ref_rigid, volume_ref_rss_rigid = augment3_d(volume_ref,
                                                                    degree=([0,
                                                                             args.aux_aug * args.rotate,
                                                                             args.aux_aug * args.rotate]),
                                                                    translation=([args.aux_aug * args.translate,
                                                                                  0,
                                                                                  0]))  # through-plane motion
                # volume_ref_rigid, volume_ref_rss_rigid = augment3D(volume_ref,
                # degree =([args.aux_aug*args.rotate 0, 0]),
                # translation = ([0, args.aux_aug*args.translate, args.aux_aug*args.translate])) #in-plane motion
                volume_new = []
                for idx in range_list:
                    volume_new.append([volume[idx][0], volume_ref_rigid[idx].numpy()])

                ran_new = range(len(volume_new))
                batch = [torch.tensor(np.stack(s, axis=0)).to(device) for s in zip(*[volume_new[j] for j in ran_new])]
            else:
                batch = [torch.tensor(np.stack(s, axis=0)).to(device) for s in zip(*[volume[j] for j in range_list])]
            #applicazione del ritaglio
            batch = [center_crop(i, (cfg.shape, cfg.shape)) for i in batch]

            #eseguo il test del modello sulle img di batch
            net.test(*batch)

            vis = net.get_vis('scalars')
            stat_eval.append(vis['scalars'])
            psnr.append(vis['scalars']['metric_PSNR'])
            ssim.append(vis['scalars']['metric_SSIM'])
            psnr_raw.append(vis['scalars']['metric_PSNR_raw'])
            ssim_raw.append(vis['scalars']['metric_SSIM_raw'])
            print('Raw volume:', i + 1, f', psnr: {psnr_raw[-1]:.2f}', f', ssim: {ssim_raw[-1]:.4f}')
            print('Recon volume:', i + 1, f', psnr: {psnr[-1]:.2f}', f', ssim: {ssim[-1]:.4f}')

            vis = net.get_vis('images')
            for name, val in vis['images'].items():
                torchvision.utils.save_image(val,
                                             args.save_path + '/' + '%010d_' % i + name + '.jpg',
                                             nrow=batch[0].shape[0] // col_vis, padding=10,
                                             range=(0, 1), pad_value=0.5)

            del batch

        #salvataggio dei risultati
        if args.save_img is False:
            continue
        image, sampled, aux, rec = net.Target_f_rss, net.Target_sampled_rss, net.Ref_f_rss, net.rec_rss
        image, sampled, aux, rec = [nib.Nifti1Image(x.cpu().squeeze(1).numpy().T, affine) for x in
                                    (image, sampled, aux, rec)]
        nib.save(image, args.save_path + '/' + str(i) + '_image.nii')
        nib.save(aux, args.save_path + '/' + str(i) + '_aux.nii')
        nib.save(sampled, args.save_path + '/' + str(i) + '_sampled.nii')
        nib.save(rec, args.save_path + '/' + str(i) + '_rec.nii')

    with open(args.save_path + '/' + os.path.split(args.model_path)[1][:-3] + '.txt', 'w') as f:
        json.dump(stat_eval, f)
    vis_mean = {key: statistics.mean([x[key] for x in stat_eval]) for key in stat_eval[0]}
    vis_std = {key: statistics.stdev([x[key] for x in stat_eval]) for key in stat_eval[0]}

    print(vis_mean)
    print(vis_std)


def parse():
    import argparse
    #creazione del parser
    parser = argparse.ArgumentParser(description='jCAN for MRI reconstruction')  #oggetto che andrà ad analizzare gli
                                                                                 #argomenti passati alla riga di comando
    #configurazione del parser: definisco i parametri che lo script dovrà accettare
    parser.add_argument('--model_path', type=str, default='D:/tesi rebecca/mylog/ckpt/best.pt',
                        help='with ckpt path, set empty str to load latest ckpt')  #pecifica il percorso al checkpoint del
                                                                                   #modello da caricare.
    parser.add_argument('--save_path', type=str, default='D:/tesi rebecca/output',
                        help='path to save evaluated data')  #definisce il percorso dove verranno salvati i dati valutati
    parser.add_argument('--save_img', default=True,
                        type=bool, help='save images or not')  #indica se le immagini dovrebbero essere salvate o meno durante l'esecuzione
    parser.add_argument('--val', default='D:/tesi rebecca/jCAN-main/single_data_VAL/data_VAL.csv',
                        type=str, help='path to csv of test data')  #percorso al file CSV contenente i dati di test
    parser.add_argument('--basepath',
                        default='D:/tesi rebecca/jCAN-main',
                        type=str, help='path to test data')  #specifica il percorso base dove sono memorizzati i dati di test.
    parser.add_argument('--shape', type=tuple, default=320,
                        help='mask and image shape, images will be cropped to match')  #definisce la dim delle img e maschere
    parser.add_argument('--protocals', metavar='NAME',
                        type=str, default=['CORPDFS_FBK'], nargs='*',
                        help='input modalities')  #specifica le modalità di acquisizione delle immagini (protocolli)
    parser.add_argument('--aux_aug', type=float, default=-1,
                        help='data augmentation aux image, set to -1 to ignore')  #definisce il livello di augmentazione
                                                                                  #dei dati per le immagini ausiliarie.
    parser.add_argument('--rotate', type=float, default=0.01 * 180,
                        help='rotation augmentation in degree')  #Specifica l'angolo di rotazione in gradi per l'augmentazione
                                                                 #delle immagini.
    parser.add_argument('--translate', type=float, default=0.05,
                        help='translation augmentation in pixel')  #Specifica l'ampiezza della traslazione in pixel per
                                                                   #l'augmentazione delle immagini.
    parser.add_argument('--GT', type=bool, default=True,
                        help='if there is GT, default is True')  #Indica se esistono o meno le verità di terreno per i dati test
    #ANALISI degli argomenti
    args = parser.parse_args()  #con questa chiamata il parser può essere usato per analizzare gli argomenti effettivamente
                                #forniti dall'utente quando esegue lo script
    main(args)


if __name__ == '__main__':
    parse()
#Questo assicura che la funzione main venga eseguita solo se lo script viene eseguito direttamente, non se viene
#importato come modulo in un altro script.