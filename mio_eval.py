import os
import torch
import numpy as np
import nibabel as nib
from paired_dataset import get_paired_volume_datasets, center_crop
from model import ReconModel
import torchvision.utils
from tqdm import tqdm

def validate(args):
    affine = np.eye(4) * [0.7, -0.7, -5, 1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Caricamento del modello
    if os.path.isfile(args.model_path):
        ckpt = torch.load(args.model_path)
        cfg = ckpt['config']
        net = ReconModel(cfg=cfg)
        net.load_state_dict(ckpt['state_dict'])
        print('Modello caricato da:', args.model_path)
    else:
        raise FileNotFoundError("Checkpoint del modello non trovato.")

    net.to(device)
    net.eval()

    volumes = get_paired_volume_datasets(
        args.val, crop=cfg.shape, protocals=args.protocals, basepath=args.basepath
    )

    for i, volume in enumerate(volumes):
        print("Volume: ", volume)
        with torch.no_grad():
            # batch = [torch.tensor(np.stack(s, axis=0)).to(device) for s in zip(*volume)]
            # batch = [center_crop(i, (cfg.shape, cfg.shape)) for i in batch]

            # net.test(*batch)
            # vis = net.get_vis('scalars')

            # print(f'Volume {i + 1}: PSNR {vis["scalars"]["metric_PSNR"]:.2f}, '
            #       f'SSIM {vis["scalars"]["metric_SSIM"]:.4f}')

            # del batch
            # torch.cuda.empty_cache()


            print("Salvando")
            image, rec = net.Target_f_rss, net.rec_rss
            image, rec = [nib.Nifti1Image(x.cpu().squeeze(1).numpy().T, affine) for x in (image, rec)]
            nib.save(image, f'{args.save_path}/{i}_image.nii')
            nib.save(rec, f'{args.save_path}/{i}_rec.nii')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='jCAN for MRI reconstruction')  # oggetto che andrà ad analizzare gli
    # argomenti passati alla riga di comando
    # configurazione del parser: definisco i parametri che lo script dovrà accettare
    parser.add_argument('--model_path', type=str, default='D:/tesi rebecca/jCAN-main/ckpt4_1/ckpt/best.pt',
                        help='with ckpt path, set empty str to load latest ckpt')  # pecifica il percorso al checkpoint del
    # modello da caricare.
    parser.add_argument('--save_path', type=str, default='D:/tesi rebecca/output1',
                        help='path to save evaluated data')  # definisce il percorso dove verranno salvati i dati valutati
    parser.add_argument('--save_img', default=True,
                        type=bool,
                        help='save images or not')  # indica se le immagini dovrebbero essere salvate o meno durante l'esecuzione
    parser.add_argument('--val', default='D:/tesi rebecca/jCAN-main/single_data_TEST/data_TEST.csv',
                        type=str, help='path to csv of test data')  # percorso al file CSV contenente i dati di test
    parser.add_argument('--basepath',
                        default='D:/tesi rebecca/jCAN-main',
                        type=str,
                        help='path to test data')  # specifica il percorso base dove sono memorizzati i dati di test.
    parser.add_argument('--shape', type=tuple, default=320,
                        help='mask and image shape, images will be cropped to match')  # definisce la dim delle img e maschere
    parser.add_argument('--protocals', metavar='NAME',
                        type=str, default=['CORPDFS_FBK'], nargs='*',
                        help='input modalities')  # specifica le modalità di acquisizione delle immagini (protocolli)
    parser.add_argument('--aux_aug', type=float, default=-1,
                        help='data augmentation aux image, set to -1 to ignore')  # definisce il livello di augmentazione
    # dei dati per le immagini ausiliarie.
    parser.add_argument('--rotate', type=float, default=0.01 * 180,
                        help='rotation augmentation in degree')  # Specifica l'angolo di rotazione in gradi per l'augmentazione
    # delle immagini.
    parser.add_argument('--translate', type=float, default=0.05,
                        help='translation augmentation in pixel')  # Specifica l'ampiezza della traslazione in pixel per
    # l'augmentazione delle immagini.
    parser.add_argument('--GT', type=bool, default=True,
                        help='if there is GT, default is True')
    args = parser.parse_args()
    validate(args)