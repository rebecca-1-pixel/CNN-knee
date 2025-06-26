import h5py
import os


def explore_h5_file(file_path):
    """Esplora un file HDF5 e stampa le sue chiavi e la forma dei dataset."""
    if not os.path.exists(file_path):
        print(f"Il file {file_path} non esiste. Verifica il percorso.")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"File {file_path} aperto correttamente.")
            print("Contenuto del file:")
            explore_h5_group(f)

    except OSError as e:
        print(f"Errore durante l'apertura del file {file_path}: {e}")


def explore_h5_group(group):
    """Esplora un gruppo HDF5 e stampa chiavi e forme dei dataset."""
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            print(f"Dataset: {key}, Forma: {item.shape}, Tipo: {item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"Gruppo: {key}")
            explore_h5_group(item)


if __name__ == "__main__":
    # Sostituisci questo con il percorso corretto del tuo file HDF5
    file_path = r'C:\Users\rebic\PycharmProjects\test_torch\JCAN-main\paired\knee_singlecoil_train\singlecoil_train\file1000001.h5'

    explore_h5_file(file_path)

