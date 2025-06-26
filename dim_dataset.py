import pandas as pd
import h5py

def estrai_dimensioni(file_csv):
    # Leggi il CSV senza intestazioni
    df = pd.read_csv(file_csv, header=None)

    # Il file CSV ha solo una colonna, quindi usiamo la colonna 0
    file_paths = df[0].tolist()

    # Per ogni file H5, estrai le dimensioni dei dataset
    for file_path in file_paths:
        print(f"File: {file_path}")

        # Apri il file H5
        try:
            with h5py.File(file_path, 'r') as f:
                # Per ogni dataset nel file, estrai e stampa le dimensioni
                for name, dataset in f.items():
                    dimensions = dataset.shape
                    print(f"Dataset: {name}, Dimensioni: {dimensions}")
                    # Memorizza e stampa la dimensione con il nome del file
                    print(f"File {file_path} contiene dataset '{name}' con dimensioni {dimensions}")

        except Exception as e:
            print(f"Impossibile aprire il file {file_path}: {e}")

# Esegui la funzione con il percorso del tuo file CSV
csv_file_path = r'C:\Users\rebic\PycharmProjects\test_torch\JCAN-main\paired\knee_singlecoil_val\singlecoil_val\file1000000.h5'
estrai_dimensioni(csv_file_path)




