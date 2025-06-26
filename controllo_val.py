import csv
import h5py
import numpy as np
import os

# Funzione per leggere un file .h5 e restituire i dati
def read_h5_file(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            data = {key: np.array(f[key]) for key in f.keys()}  # Estrarre tutti i dataset
        return data
    except Exception as e:
        print(f"Errore nella lettura del file {filepath}: {e}")
        return None

# Funzione per analizzare il contenuto e segnalare problemi di concatenazione
def analyze_data(data_list):
    shape_list = [d.shape for d in data_list]  # Estrai le forme dei dati
    dtype_list = [d.dtype for d in data_list]  # Estrai i tipi di dati

    print("Forme dei dati estratti:", shape_list)
    print("Tipi di dati estratti:", dtype_list)

    # Verifica problemi di concatenazione
    if len(set(shape_list)) > 1:
        print("Attenzione: I dati hanno forme diverse, non è possibile concatenarli direttamente.")
        return False
    if len(set(dtype_list)) > 1:
        print("Attenzione: I dati hanno tipi diversi, non è possibile concatenarli direttamente.")
        return False
    return True

# Funzione principale per estrarre informazioni dai file .h5
def extract_info_from_csv(csv_path, basepath=''):
    extracted_data = []

    # Leggi il file CSV
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Ogni riga del CSV contiene il percorso di un file .h5
            h5_filepath = os.path.join(basepath, row[0])
            print(f"Leggendo il file {h5_filepath}...")

            # Estrai i dati dal file .h5
            data = read_h5_file(h5_filepath)
            if data:
                extracted_data.append(data)

    # Se non ci sono dati, esci
    if not extracted_data:
        print("Nessun dato estratto dai file.")
        return

    # Analizza i dati per problemi di concatenazione
    for key in extracted_data[0].keys():  # Assumi che tutti i file abbiano le stesse chiavi
        print(f"\nAnalizzando la chiave '{key}'...")
        data_list = [d[key] for d in extracted_data]

        # Analizza le forme e i tipi dei dati per questa chiave
        if analyze_data(data_list):
            print(f"È possibile concatenare i dati per la chiave '{key}'")
        else:
            print(f"Non è possibile concatenare i dati per la chiave '{key}'")

# Percorso al CSV contenente i file .h5
csv_path ='C:/Users/rebic/PycharmProjects/test_torch/jCAN-main/dati_val_loaded/valid_dataset.csv'
basepath = 'C:/Users/rebic/PycharmProjects/test_torch/jCAN-main' # Se i percorsi nel CSV sono relativi, specifica il basepath

# Estrai informazioni e analizza i file .h5
extract_info_from_csv(csv_path, basepath)
