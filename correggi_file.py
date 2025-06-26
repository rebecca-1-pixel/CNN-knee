import csv
import os


def fix_file_paths_in_place(csv_path):
    # Crea un file temporaneo per la scrittura
    temp_csv_path = csv_path + '.temp'

    with open(csv_path, 'r') as input_csv_file, open(temp_csv_path, 'w', newline='') as output_csv_file:
        reader = csv.reader(input_csv_file)
        writer = csv.writer(output_csv_file)

        for row in reader:
            # Correggi i percorsi nella riga corrente
            corrected_row = []
            for path in row:
                # Rimuovi virgolette e correggi i separatori
                path = path.strip("[]").strip().replace("//", "/")
                # Rimuovi eventuali virgolette aggiuntive
                path = path.replace("'", "").replace('"', '')
                corrected_row.append(path)
            writer.writerow(corrected_row)

    # Sostituisci il file originale con il file temporaneo
    os.replace(temp_csv_path, csv_path)
    print(f"Percorsi corretti salvati nel file CSV: {csv_path}")


# Esempio di utilizzo
csv_path = 'C:/Users/rebic/PycharmProjects/test_torch/jCAN-main/dati_val_loaded/valid_dataset.csv'
fix_file_paths_in_place(csv_path)



