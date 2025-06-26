import h5py
import os
import torch
torch.set_default_dtype(torch.float32)

def extract_acquisition_param(h5_file_path):
    """Estrae il parametro 'acquisition' da un file H5."""
    with h5py.File(h5_file_path, 'r') as h5_file:
        if 'acquisition' in h5_file.attrs:
            acquisition_method = h5_file.attrs['acquisition']
            return acquisition_method
        else:
            return None

def process_h5_files_in_directory(directory_path):
    """Processa tutti i file H5 in una cartella ed estrae il parametro 'acquisition'."""
    results = []

    # Attraversa tutti i file nella cartella specificata
    for filename in os.listdir(directory_path):
        if filename.endswith(".h5"):
            file_path = os.path.join(directory_path, filename)
            acquisition_method = extract_acquisition_param(file_path)
            if acquisition_method:
                results.append(f"{filename}: {acquisition_method}")
            else:
                results.append(f"{filename}: 'acquisition' attribute not found")

    # Stampa i risultati
    for result in results:
        print(result)

    # Salva i risultati in un file di testo
    output_file = os.path.join(directory_path, "acquisition_methods.txt")
    with open(output_file, 'w') as f:
        for result in results:
            f.write(result + '\n')

# Specifica il percorso della cartella contenente i file H5
directory_path = r"C:\Users\rebic\PycharmProjects\test_torch\estrazione\acquisition_param\paired\percorsi_corpd_fbk_TRAIN"

# Processa i file nella cartella
process_h5_files_in_directory(directory_path)