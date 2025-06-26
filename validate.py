import os
import torch
import numpy as np
import nibabel as nib
from paired_dataset import get_paired_volume_datasets, center_crop
from model import ReconModel
import torchvision.utils
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def compute_metrics(gt, pred):
    """
    Calcola PSNR e SSIM su ciascuna slice (2D) e restituisce la media sul volume (3D).
    """
    psnr_total, ssim_total = 0.0, 0.0
    n_slices = gt.shape[0]

    for i in range(n_slices):
        gt_slice = gt[i]
        pred_slice = pred[i]

        psnr = compare_psnr(gt_slice, pred_slice, data_range=gt_slice.max() - gt_slice.min())
        ssim = compare_ssim(gt_slice, pred_slice, data_range=gt_slice.max() - gt_slice.min())

        psnr_total += psnr
        ssim_total += ssim

    return psnr_total / n_slices, ssim_total / n_slices


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

    all_psnr, all_ssim = [], []

    for i, volume in enumerate(tqdm(volumes, desc="Validazione dei volumi")):
        with torch.no_grad():
            batch = [torch.tensor(np.stack(s, axis=0)).to(device) for s in zip(*volume)]
            batch = [center_crop(i, (cfg.shape, cfg.shape)) for i in batch]

            net.test(*batch)

            # Output GT e ricostruzione
            target, recon = net.Target_f_rss, net.rec_rss
            target_np = target.cpu().squeeze(1).numpy()
            recon_np = recon.cpu().squeeze(1).numpy()

            # Calcolo metriche
            psnr, ssim = compute_metrics(target_np, recon_np)
            all_psnr.append(psnr)
            all_ssim.append(ssim)

            print(f'Volume {i + 1}: PSNR {psnr:.2f}, SSIM {ssim:.4f}')

            if args.save_img:
                image_nifti, rec_nifti = [nib.Nifti1Image(x.transpose(1, 2, 0), affine) for x in (target_np, recon_np)]
                nib.save(image_nifti, f'{args.save_path}/{i}_image.nii')
                nib.save(rec_nifti, f'{args.save_path}/{i}_rec.nii')

            del batch
            torch.cuda.empty_cache()

    # Stampa delle metriche complessive
    print(f'\nValutazione completata su {len(volumes)} volumi:')
    print(f'Media PSNR: {np.mean(all_psnr):.2f}, Std: {np.std(all_psnr):.2f}')
    print(f'Media SSIM: {np.mean(all_ssim):.4f}, Std: {np.std(all_ssim):.4f}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='jCAN for MRI reconstruction')
    parser.add_argument('--model_path', type=str, default='D:/tesi rebecca/jCAN-main/ckpt4_1/ckpt/best.pt',
                        help='with ckpt path, set empty str to load latest ckpt')
    parser.add_argument('--save_path', type=str, default='D:/tesi rebecca/output',
                        help='path to save evaluated data')
    parser.add_argument('--save_img', default=True, type=bool,
                        help='save images or not')
    parser.add_argument('--val', default='D:/tesi rebecca/jCAN-main/single_data_TEST/data_TEST.csv',
                        type=str, help='path to csv of test data')
    parser.add_argument('--basepath',
                        default='D:/tesi rebecca/jCAN-main',
                        type=str,
                        help='path to test data')
    parser.add_argument('--shape', type=tuple, default=320,
                        help='mask and image shape, images will be cropped to match')
    parser.add_argument('--protocals', metavar='NAME',
                        type=str, default=['CORPDFS_FBK'], nargs='*',
                        help='input modalities')
    parser.add_argument('--aux_aug', type=float, default=-1,
                        help='data augmentation aux image, set to -1 to ignore')
    parser.add_argument('--rotate', type=float, default=0.01 * 180,
                        help='rotation augmentation in degree')
    parser.add_argument('--translate', type=float, default=0.05,
                        help='translation augmentation in pixel')
    parser.add_argument('--GT', type=bool, default=True,
                        help='if there is GT, default is True')

    args = parser.parse_args()
    validate(args)
