import numpy as np
import skimage
try:
    import skimage.metrics
    from skimage.metrics import structural_similarity as compare_ssim
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
except ImportError:
    import skimage.measure
    from skimage.measure import compare_psnr, compare_ssim
import torch
torch.set_default_dtype(torch.float32)

#DA TENSORE AD ARRAY NUMPY
def to_numpy(*args):
    outputs = []
    for arg in args:
        if hasattr(arg, 'cpu') and callable(arg.cpu):
            arg = arg.detach().cpu()
        if hasattr(arg, 'numpy') and callable(arg.numpy):
            arg = arg.detach().numpy()
        assert len(arg.shape) == 4, 'wrong shape [batch, channel=1, rows, cols'
        outputs.append(arg)
    return outputs


#ERRORE QUADRATICO MEDIO
def mse(gt, pred, gt_flag=True):
    if gt_flag==True:
        gt, pred = to_numpy(gt, pred)
        mse = np.mean((gt - pred) ** 2) #??? .element() non è un elemento di Numpy
    else:
        mse = -1000
    return mse


#ERRORE ASSOLUTO MEDIO
def mae(gt, pred, gt_flag=True):
    if gt_flag==True:
        gt, pred = to_numpy(gt, pred)
        mae = np.mean(np.absolute(gt - pred))
    else:
        mae = -1000
    return mae


#ERRORE QUADRATICO MEDIO NORMALIZZATO
def nmse(gt, pred, gt_flag=True):
    if gt_flag==True:
        gt, pred = to_numpy(gt, pred)
        nmse = (np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)
    else:
        nmse = -1000
    return nmse


#PEAK SIGNAL TO NOISE RATIO
def psnr(gt, pred, gt_flag=True):
    if gt_flag==True:
        gt, pred = to_numpy(gt, pred)
        psnr = compare_psnr(gt, pred, data_range=1).item()  #data_range indica che i valori dei pixel delle img è
                                                            #normalizzato in un intervallo da 0 a 1
                                                            #il PSNR è calcolato come misura in decibel (dB)
                                                            #.item() è usato per estrarre il valore scalare dal risultato
    else:
        psnr = -1000
    return psnr


#INDICE DI SOMIGLIANZA STRUTTURALE
def ssim(gt, pred, gt_flag=True):
    if gt_flag==True:
        gt, pred = to_numpy(gt, pred)
        ssim = np.mean([compare_ssim(g[0], p[0], data_range=1) \
                for g, p in zip(gt, pred)])   #è una lista comprehension che calcola la SSIM tra ogni coppia
                                                        #di immagini g e p in gt e pred. compare_ssim calcola il SSIM
    else:
        ssim = -1000
    return ssim


#COEFFICIENTE DI DICE
def dice(gt, pred, label=None):
    gt, pred = to_numpy(gt, pred)
    if label is None:  #se label non è specificato...
        gt, pred = gt.astype(np.bool), pred.astype(np.bool)  #... i dati gt e pred vengono convertiti in array booleani
    else:  #se label è specificato...
        gt, pred = (gt == label), (pred == label)  #...i dati gt e pred sono confrontati con il valore di label. Questo
                                                   #produce un array bool solo se il valore corrisponde all'etichetta
                                                   #speficata
    intersection = np.logical_and(gt, pred)   #calcolo l'interesezione tra gt e pred usando AND logico: serve a
                                              #identificare le aree comuni tra le due segmentazioni, ovvero i pixel che
                                              #sono True sia in gt che in pred
    return 2.*intersection.sum() / (gt.sum() + pred.sum())  #calcolo del coefficiente di dice


from scipy.special import xlogy
#INFORMAZIONE MUTUA
def mi(gt, pred, bins=64, minVal=0, maxVal=1):
    assert gt.shape == pred.shape
    gt, pred = to_numpy(gt, pred)
    mi = []
    for x, y in zip(gt, pred):
        Pxy = np.histogram2d(x.ravel(), y.ravel(), bins, \
                range=((minVal,maxVal),(minVal,maxVal)))[0]  #x.ravel() e y.ravel() appiattiscono le immagini in vettori
              #1D. np.histogram2d costruisce un istogramma bidimensionale con bins bin su ciascun asse, basato sugli
              #intervalli dati da minVal e maxVal. [0] estrae solo la matrice di conteggio dell'istogramma
        Pxy = Pxy/(Pxy.sum()+1e-10)  #Normalizza l'istogramma in modo che la somma dei valori sia 1, aggiungendo un
                                     #piccolo valore 1e-10 per evitare divisioni per zero.
        Px = Pxy.sum(axis=1)  #Calcola la distribuzione marginale di x sommando lungo l'asse delle colonne.
        Py = Pxy.sum(axis=0)  #Calcola la distribuzione marginale di y sommando lungo l'asse delle righe.
        PxPy = Px[..., None]*Py[None, ...]  #Calcola la distribuzione di probabilità marginale congiunta PxPy
        #mi = Pxy * np.log(Pxy/(PxPy+1e-6))
        result = xlogy(Pxy, Pxy) - xlogy(Pxy, PxPy)  #utilizza xlogy per calcolare l'informazione mutua
        mi.append(result.sum())  #Somma tutti i valori dell'array risultante per ottenere il valore totale di MI per
                                 #la coppia di immagini.
    return np.mean(mi).element()  #Restituisce la media dei valori di MI calcolati per tutte le coppie di immagin


if __name__ == "__main__":
    gt, pred = np.random.rand(10, 1, 100, 100), np.random.rand(10, 1, 100, 100)
    print('MSE', mse(gt, pred))
    print('NMSE', nmse(gt, pred))
    print('PSNR', psnr(gt, pred))
    print('SSIM', ssim(gt, pred))
    print('MI', mi(gt, pred))
