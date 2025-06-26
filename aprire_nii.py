import nibabel as nib
import numpy as np
from PIL import Image


def apri_e_visualizza_nii(percorso_file):
    # Carica il file .nii o .nii.gz
    img_nii = nib.load(percorso_file)

    # Estrai i dati come array NumPy
    dati = img_nii.get_fdata()
    print(f"Forma del volume: {dati.shape}")  # Mostra la forma del volume

    # Se il volume è 3D, seleziona uno slice intermedio lungo l'asse z
    if len(dati.shape) == 3:
        slice_index = dati.shape[2] // 2
        slice_2d = dati[:, :, slice_index]
    elif len(dati.shape) == 4:
        # Nel caso in cui il volume sia 4D (es. tempo o altre dimensioni)
        # Prendiamo uno slice di uno dei volumi temporali
        slice_index = dati.shape[2] // 2
        slice_2d = dati[0, :, :, slice_index]  # Primo volume, slice
    else:
        raise ValueError(f"Forma non supportata: {dati.shape}")

    # Normalizza l'immagine per la visualizzazione
    slice_norm = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d)) * 255
    slice_img = slice_norm.astype(np.uint8)  # Conversione in uint8

    # Crea e mostra l'immagine
    img = Image.fromarray(slice_img)
    img.show(title=f"Slice z={slice_index}")
    img.save("slice_visualizzata.jpg")  # Salva l'immagine se necessario


# Esempio d'uso
percorso_file_nii = "D:/tesi rebecca/output1.1/1_image.nii"
apri_e_visualizza_nii(percorso_file_nii)
