import os
import h5py
import csv


def leggi_metodo_acquisizione(file_h5):
    """
    Estrai il metodo di acquisizione dal file H5.
    Args:
        file_h5 (str): Il percorso del file H5.
    Returns:
        str: Il metodo di acquisizione.
    """
    try:
        with h5py.File(file_h5, 'r') as f:
            # Supponiamo che il metodo di acquisizione sia memorizzato in un attributo chiamato 'protocol'
            if 'acquisition' in f.attrs:
                return f.attrs['acquisition']
            else:
                return 'Unknown'
    except Exception as e:
        print(f"Errore nella lettura del file {file_h5}: {e}")
        return 'Unknown'


def crea_csv_per_metodi(cartella_origine, file_csv_corpd, file_csv_corpdfs):
    """
    Crea due file CSV separati per i file H5 con diversi metodi di acquisizione.
    Args:
        cartella_origine (str): Il percorso della cartella contenente i file H5.
        file_csv_corpd (str): Il percorso del file CSV per i file con CORPD_FBK.
        file_csv_corpdfs (str): Il percorso del file CSV per i file con CORPDFS_FBK.
    """
    # Apre i file CSV per la scrittura
    with open(file_csv_corpd, 'w', newline='') as csvfile_corpd, \
            open(file_csv_corpdfs, 'w', newline='') as csvfile_corpdfs:

        csv_writer_corpd = csv.writer(csvfile_corpd)
        csv_writer_corpdfs = csv.writer(csvfile_corpdfs)

        # Cicla attraverso tutti i file nella cartella di origine
        for nome_file in os.listdir(cartella_origine):
            if nome_file.endswith('.h5'):
                percorso_file = os.path.join(cartella_origine, nome_file)

                # Estrai il metodo di acquisizione
                metodo_acquisizione = leggi_metodo_acquisizione(percorso_file)

                # Aggiungi il file al CSV corrispondente
                if metodo_acquisizione == 'CORPD_FBK':
                    csv_writer_corpd.writerow([percorso_file])
                elif metodo_acquisizione == 'CORPDFS_FBK':
                    csv_writer_corpdfs.writerow([percorso_file])

    print(f"File CSV creato con successo: {file_csv_corpd}")
    print(f"File CSV creato con successo: {file_csv_corpdfs}")

# Percorsi della cartella contenente i file H5 e dei file CSV da creare
cartella_origine = r'C:\Users\rebic\PycharmProjects\test_torch\JCAN-main\paired\knee_singlecoil_train\singlecoil_train'
file_csv_corpd = r'C:\Users\rebic\PycharmProjects\test_torch\JCAN-main\paired\percorsi_corpd_fbk_TRAIN.csv'
file_csv_corpdfs = r'C:\Users\rebic\PycharmProjects\test_torch\JCAN-main\paired\percorsi_corpdfs_fbk_TRAIN.csv'

# Chiama la funzione per creare i file CSV
crea_csv_per_metodi(cartella_origine, file_csv_corpd, file_csv_corpdfs)



def get_paired_volume_datasets(csv_path, protocals=None, crop=None, q=0, flatten_channels=False, basepath=None,
                               exchange_modal=False, output_csv_path=None):
    datasets = []
    valid_files = []
    print(csv_path)
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            print(line)
            dataset_paths = [os.path.join(basepath, filepath) for filepath in line]

            try:
                # Costruzione del dataset con il percorso corretto
                dataset = AlignedVolumesDataset(*dataset_paths, protocals=protocals, crop=crop, q=q,
                                                flatten_channels=flatten_channels, exchange_modal=exchange_modal)

                # Verifica che il dataset non sia vuoto
                if len(dataset) > 0:
                    datasets.append(dataset)
                    valid_files.append(','.join(dataset_paths))
                else:
                    print(f"Dataset vuoto trovato per i file: {dataset_paths}")

            except Exception as e:
                print(f"Errore nel caricamento dei file: {dataset_paths}. Errore: {e}")

    # Salva i file validi in un altro file CSV
    if output_csv_path:
        with open(output_csv_path, 'w', newline='') as output_csv_file:
            writer = csv.writer(output_csv_file)
            writer.writerows([path.split(',')] for path in valid_files)
        print(f"File CSV con i percorsi dei dataset validi salvato in: {output_csv_path}")

    return datasets