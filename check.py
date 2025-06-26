import h5py
import os

def check_attributes_in_h5_files(directory):
    # Cicla attraverso i file nella cartella specificata
    for filename in os.listdir(directory):
        if filename.endswith('.h5'):
            file_path = os.path.join(directory, filename)
            try:
                with h5py.File(file_path, 'r') as f:
                    # Esamina gli attributi del file
                    if 'CORPDFS_FBK' in f.attrs.values():
                        print(f"File '{filename}' contiene l'attributo CORPDFS_FBK")
                    else:
                        print(f"File '{filename}' NON contiene l'attributo CORPDFS_FBK")
            except Exception as e:
                print(f"Errore nel leggere il file {filename}: {e}")

# Esegui la funzione, passando la cartella contenente i file H5
directory_path = 'D:/tesi rebecca/jCAN-main/paired/singlecoil_test/'  # Modifica questo percorso con il percorso della tua cartella
check_attributes_in_h5_files(directory_path)