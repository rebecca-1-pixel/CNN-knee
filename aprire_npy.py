import numpy as np

# Carica il file .npy (modifica il percorso con il tuo file)
file_path = 'D:/tesi rebecca/jCAN-main/ckpt10/SSIM.npy'
data = np.load(file_path)

# Visualizza il contenuto del file (se necessario)
print("Contenuto del file:")
print(data)

# Calcola la media del vettore
mean_value = np.mean(data)

# Stampa il valore medio
print(f"Il valore medio del vettore è: {mean_value}")