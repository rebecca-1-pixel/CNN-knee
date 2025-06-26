#!/usr/bin/env python3
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
from tqdm import tqdm
from PIL import Image

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

    #todo: funzione aggiunta per il test
    # Funzione per rendere serializzabile qualsiasi tipo di valore
    def make_serializable(obj):
        # Se è un tipo np.float32 o np.float64, lo convertiamo in float
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        # Se è un dizionario, lo processiamo ricorsivamente
        elif isinstance(obj, dict):
            return {key: make_serializable(value) for key, value in obj.items()}
        # Se è una lista, lo processiamo ricorsivamente
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        # Se è una tupla, lo processiamo ricorsivamente
        elif isinstance(obj, tuple):
            return tuple(make_serializable(item) for item in obj)
        # Se è un set, lo processiamo ricorsivamente
        elif isinstance(obj, set):
            return {make_serializable(item) for item in obj}
        # Restituiamo l'oggetto così com'è se non è un tipo da convertire
        else:
            return obj

    # Applicazione della funzione di serializzazione a stat_eval
    #stat_eval_serializable = [make_serializable(item) for item in stat_eval]

    # Ora possiamo salvare il dizionario in formato JSON
    #with open(args.save_path + '/' + os.path.split(args.model_path)[1][:-3] + '.txt', 'w') as f:
        json.dump(stat_eval_serializable, f)

    #with open(args.save_path + '/' + os.path.split(args.model_path)[1][:-3] + '.txt', 'w') as f:
        #json.dump(stat_eval, f)

    #todo: funzione aggiunta per il test:
    # Funzione per calcolare la deviazione standard solo se ci sono almeno due dati
    # Funzione sicura per la deviazione standard (safe_stdev)
    def safe_stdev(data):
        # Se ci sono meno di 2 dati, non possiamo calcolare la deviazione standard, quindi ritorniamo 0
        if len(data) < 2:
            return 0.0
        return statistics.stdev(data)

    # Calcolare la media
    #vis_mean = {key: statistics.mean([x[key] for x in stat_eval]) for key in stat_eval[0]}

    # Calcolare la deviazione standard in modo sicuro, usando safe_stdev
    #vis_std = {key: safe_stdev([x[key] for x in stat_eval]) for key in stat_eval[0]}

    #print(vis_mean)
    #print(vis_std)




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
    parser.add_argument('--val', default='D:/tesi rebecca/jCAN-main/single_data_TEST/data_TEST.csv',
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