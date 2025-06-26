import os
import torch
import torchvision
import numpy as np
import nibabel as nib
from tqdm import tqdm
from PIL import Image
import argparse  # Aggiungi l'importazione di argparse
from paired_dataset import get_paired_volume_datasets, center_crop
from model import ReconModel

# Funzione principale
def main(args):
    affine = np.eye(4) * [0.7, -0.7, -5, 1]  # Matrice di trasformazione affine

    print("Args: ", args)

    # Creazione della directory di salvataggio
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # Configurazione del dispositivo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Caricamento del modello
    if os.path.isfile(args.model_path) or os.path.isdir(args.model_path):
        ckpt = torch.load(args.model_path)
        cfg = ckpt['config']
        net = ReconModel(cfg=cfg)
        net.load_state_dict(ckpt['state_dict'])
        print('Caricato il modello da:', args.model_path)
    else:
        raise FileNotFoundError

    # Configurazione del modello
    net.use_amp = False
    cfg = net.cfg
    net.GT = args.GT

    # Caricamento del dataset
    volumes = get_paired_volume_datasets(args.val, crop=cfg.shape, protocals=args.protocals, basepath=args.basepath)

    # Loop per generare e salvare le immagini senza validazione
    for i, volume in enumerate(tqdm(volumes, desc="Generazione immagini", unit="volume")):
        range_list = range(len(volume))
        batch = [torch.tensor(np.stack(s, axis=0)).to(device) for s in zip(*[volume[j] for j in range_list])]

        # Estrazione del volume 3D
        volume_3d = batch[0].cpu().numpy()  # Forma: [C, H, W]

        # Selezioniamo una slice centrale (ad esempio la metà del volume)
        central_slice = volume_3d[0, volume_3d.shape[1] // 2, :]  # Seleziona il piano centrale

        # Normalizzazione della slice per la visualizzazione
        central_slice = (central_slice - central_slice.min()) / (central_slice.max() - central_slice.min()) * 255
        central_slice = central_slice.astype(np.uint8)

        # Salvataggio della slice centrale come immagine
        img = Image.fromarray(central_slice)
        img.show()  # Mostra la slice centrale
        img.save(args.save_path + '/' + f'volume_{i}_central_slice.jpg')  # Salva la slice come immagine

        print(f'Salvate le immagini per il volume {i + 1}')

        # Salvataggio in formato NIfTI
        if args.save_img:
            image = nib.Nifti1Image(batch[0].cpu().squeeze(1).numpy().T, affine)
            nib.save(image, args.save_path + '/' + f'{i}_image.nii')

# Blocco di configurazione per argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='jCAN for MRI reconstruction')  # oggetto che andrà ad analizzare gli
    # argomenti passati alla riga di comando
    # configurazione del parser: definisco i parametri che lo script dovrà accettare
    parser.add_argument('--model_path', type=str, default='D:/tesi rebecca/jCAN-main/ckpt10/ckpt/best.pt',
                        help='with ckpt path, set empty str to load latest ckpt')  # pecifica il percorso al checkpoint del
    # modello da caricare.
    parser.add_argument('--save_path', type=str, default='D:/tesi rebecca/output10',
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

    # Parsing degli argomenti passati
    args = parser.parse_args()

    # Avvia il processo di validazione o generazione immagini
    main(args)