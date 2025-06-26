import os
import csv


def crea_csv_con_file_h5(cartella_origine, output_csv):
    """
    Crea un file CSV con i percorsi di tutti i file .h5 presenti nella cartella specificata.

    Args:
        cartella_origine (str): Percorso della cartella che contiene i file .h5.
        output_csv (str): Percorso e nome del file CSV di output.
    """
    # Verifica che la cartella di origine esista
    if not os.path.exists(cartella_origine):
        raise FileNotFoundError(f"La cartella specificata non esiste: {cartella_origine}")

    # Ottiene una lista di file con estensione .h5 nella cartella specificata
    file_h5 = [os.path.join(cartella_origine, file) for file in os.listdir(cartella_origine) if file.endswith('.h5')]

    # Crea il file CSV e scrive i percorsi dei file .h5
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for file in file_h5:
            csv_writer.writerow([file])

    print(f"File CSV creato con successo: {output_csv}")


# Specifica il percorso della cartella contenente i file .h5
cartella_origine = r'C:\Users\rebic\PycharmProjects\test_torch\JCAN-main\paired\knee_singlecoil_val\singlecoil_val'

# Specifica il nome del file CSV di output
output_csv = r'C:\Users\rebic\PycharmProjects\test_torch\JCAN-main\paired\file_percorsi_val.csv'

# Esegui la funzione per creare il file CSV
crea_csv_con_file_h5(cartella_origine, output_csv)
