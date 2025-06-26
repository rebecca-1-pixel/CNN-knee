import os

file_path = 'C:/Users/rebic/PycharmProjects/test_torch/jCAN-main/paired/fig1.csv'

# Verifica se il file esiste
if os.path.isfile(file_path):
    print(f"Il file esiste: {file_path}")
else:
    print(f"Il file non esiste: {file_path}")
