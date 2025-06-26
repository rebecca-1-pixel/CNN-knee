#!/usr/bin/env python-3
"""
@author: sunkg, refer to the work of XuanKai: https://github.com/woxuankai/SpatialAlignmentNetwork
"""
import math
import functools
import random
import torch
import numpy as np
torch.set_default_dtype(torch.float32)

class Mask(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        #definizione degli attributi necessari alla maschera
        self.register_parameter('weight', None)  #registra un parametro weight e lo imposta a None
        self.weight = torch.nn.Parameter(torch.ones(shape))  #weight è sovrascritto con un tensore di tutti 1.
                                                             #Rappresenta i pesi associati alla maschera.
        # if a weight is already pruned, set to True
        self.register_buffer('pruned', None)  #registra un buffer "pruned" e lo imposta a None
        self.pruned = torch.zeros(shape, dtype=torch.bool)  #viene sovrascritto con un tensore di booleani che indica
                                                            #quali pesi sono stati prunati.

    def prune(self, num, thres=1, random=0):  #metodo usato per prunare i pesi
        '''
        if not random, for abs(w) < thres,
            prune at most 'num' w_i from low to high, with abs(w)
        if random, for abs(w) < uniform[0-thres]
            prune at most 'num' w_i from low to high,
            with abs(w) - uniform[0-thres].
        '''
        assert thres >= 0 and random >= 0 and num >= 0  #verifica che thres, random e num siano positivi
        if num == 0:
            return
        with torch.no_grad():  #disabilita temporaneamente il calcolo del gradiente
            w = self.weight.detach().abs()  #valore assoluto dei pesi. Li scollega dal grafo computazionale per evitare
                                            #che il pruning influenzi il backpropagation
            #uso masked_scatter_ per aggiornare solo alcune parti del tensore
            w.masked_scatter_(self.pruned, \
                    torch.ones_like(w)*(max(random, w.max())+thres))
            w.masked_scatter_(w >= thres, \
                    torch.ones_like(w)*(max(random, w.max())+thres))
            rand = torch.rand_like(w)*random  #se random è maggiore di 0 genera una casualità per il pruning
            _, ind = torch.topk(-(w-rand), num)  #seleziona i pesi più bassi da prunare fino a un massimo di num
            ind = torch.masked_select(ind, w[ind] < thres)
            self.pruned.scatter_( \
                -1, ind, torch.ones_like(self.pruned))  #aggiorna il buffer pruned per segnare i pesi selezionati come prunati.

    def forward(self, image):  #chiamato durante la fase di inferenza per applicare la maschera
        mask = torch.ones_like(self.weight)  #inizialmente mask è piena di 1
        mask.masked_scatter_(self.pruned, torch.zeros_like(self.weight))  #imposta i valori prunati a 0
        # unable to set a leaf variable here
        #self.weight.masked_scatter_(self.pruned, torch.zeros_like(self.weight))
        # mask weight in preventation of weight changing
        return image * (self.weight*mask)[None, None, None, :]  #moltiplica img per la maschera per disattivare i peso prunati

class RandomMask(Mask):
    """When  the  acceleration factorequals four,
    the fully-sampled central region includes 8% of all k-space lines;
    when it equals eight, 4% of all k-space lines are included.
    """
    def __init__(self, sparsity, shape):
        """
        sparsity: float, desired sparsity, can only be either 1/4 or 1/8
        shape: int, output mask shape
        """
        super().__init__(shape)
        center_ratio = sparsity*0.32 # i.e. 4% for 8-fold and 8% for 4-fold
        center_len = round(shape * center_ratio) # to round up to int
        other_ratio = (sparsity*shape - center_len)/(shape - center_len)  #probabilità con cui le righe al di fuori della
        # regione centrale verranno campionate, determinando la densità di campionamento nelle aree periferiche del k-space.
        prob = torch.ones(shape)*1.1  #distribuzione di probabilità uniforme
        # low freq is of the border
        prob[center_len//2:center_len//2-center_len] = other_ratio  #probabiità della regione centrale
        thresh = torch.rand(shape)  #soglia casuale che determinerà quali linee del k-space verranno campionate
        _, ind = torch.topk(prob - thresh, math.floor(sparsity*shape), dim=-1)  #seleziona le righe con la massima prob
                 #per determinare quali linee verranno campionate
        #creazione di una maschera booleana
        self.pruned = \
                torch.ones_like(thresh, dtype=torch.bool).scatter( \
                -1, ind, torch.zeros_like(thresh, dtype=torch.bool))
        

class EquispacedMask(Mask):
    def __init__(self, sparsity, shape):
        """
        sparsity: float, desired sparsity, can only be either 1/4 or 1/8
        shape: int, output mask shape
        """
        super().__init__(shape)
        center_ratio = sparsity*0.32 # i.e. 4% for 8-fold and 8% for 4-fold
        center_len = round(shape * center_ratio) # to round up to int
        self.pruned = torch.zeros(shape, dtype=torch.bool)
        # low freq is of the border
        self.pruned[center_len//2:center_len//2-center_len] = True
        remaining_cnt = math.floor(sparsity*shape - center_len)
        interval = int((shape-center_len-1)//(remaining_cnt-1))
        start_max = (shape - center_len) - \
                ((remaining_cnt-1)*interval + 1) # inclusive
        start = random.randint(0, start_max)
        pruned_part = \
                self.pruned[center_len//2:center_len//2-center_len].clone()
        pruned_part = torch.roll(pruned_part, pruned_part.shape[0]//2)
        pruned_part[start:start+interval*remaining_cnt:interval] = False
        pruned_part = torch.roll(pruned_part, (pruned_part.shape[0]+1)//2)
        # pytorch is buggy if just set False to uncloned pruned_part
        self.pruned[center_len//2:center_len//2-center_len] = pruned_part
        


class LowpassMask(Mask):
    """Low freq only
    """
    def __init__(self, sparsity, shape):
        """
        sparsity: float, desired sparsity
        shape: int, output mask shape
        """
        super().__init__(shape)
        #center_len = int(shape * sparsity+0.5) # to round up to int
        center_len = math.floor(shape * sparsity) # floor to int
        self.pruned = torch.zeros(shape, dtype=torch.bool)
        # low freq is of the border
        self.pruned[center_len//2:center_len//2-center_len] = True

def rescale_prob(x, sparsity):
    """
    Rescale Probability x so that it obtains the desired sparsity
    if mean(x) > sparsity
      x' = x * sparsity / mean(x)
    else
      x' = 1 - (1-x) * (1-sparsity) / (1-mean(x))
    """
    xbar = x.mean()
    if xbar > sparsity:
        return x * sparsity / xbar
    else:
        return 1 - (1 - x) * (1 - sparsity) / (1 - xbar)

class LOUPEMask(torch.nn.Module):
    def __init__(self, sparsity, shape, pmask_slope=5, sample_slope=12):
        """
        sparsity: float, desired sparsity
        shape: int, output mask shape
        sample_slope: float, slope for soft threshold
        mask_param -> (sigmoid+rescale) -> pmask -> (sample) -> mask
        """
        super().__init__()
        assert sparsity <= 1 and sparsity >= 0
        self.sparsity = sparsity
        self.shape = shape
        self.pmask_slope = pmask_slope  #Controlla la pendenza della funzione sigmoide che viene utilizzata per
                                        #trasformare i parametri della maschera in probabilità di campionamento.
        self.sample_slope = sample_slope  #Controlla la pendenza della sigmoide durante la fase di campionamento,
                                          # influenzando la "morbidezza" o "durezza" del campionamento.
        self.register_parameter('weight', None) #nizializza i parametri che verranno poi ottimizzati
        self.register_buffer('pruned', None)  #Inizializza un buffer
        # eps could be very small, or somethinkg like eps = 1e-6
        # the idea is how far from the tails to have your initialization.
        eps = 0.01  #Viene usato per inizializzare casualmente self.weight vicino ai bordi di una sigmoide, evitando valori estremi.
        x = torch.rand(self.shape)*(1-eps*2) + eps
        # logit with slope factor
        self.weight = torch.nn.Parameter( \
                -torch.log(1. / x - 1.) / self.pmask_slope)  #parametro trainabile che viene inizializzato tramite una
                # funzione logit inversa applicata a valori casuali, scalati per pmask_slope.
        self.forward(torch.randn(1, 1, shape, shape)) # to set self.mask

    def forward(self, example):
        assert example.shape[-1] == self.shape
        if False:
            mask = torch.zeros_like(self.weight)
            _, ind = torch.topk(self.weight, \
                    int(self.sparsity*self.shape+0.5), dim=-1)
            mask.scatter_(-1, ind, torch.ones_like(self.weight))
            self.pruned = (mask < 0.5)
            return example * mask[None, None, None, :]
        pmask = rescale_prob( \
                torch.sigmoid(self.weight*self.pmask_slope), \
                self.sparsity)
        thresh = torch.rand(example.shape[0], self.shape).to(pmask)
        _, ind = torch.topk(pmask - thresh, \
                int(self.sparsity*self.shape+0.5), dim=-1)
        not_pruned = torch.zeros_like(thresh).scatter( \
                -1, ind, torch.ones_like(thresh))
        self.pruned = (not_pruned < 0.5)[0]
        if self.training:
            mask = torch.sigmoid((pmask - thresh) * self.sample_slope)
            return example*mask[:, None, None, :]
        else:
            return example*(not_pruned)[:, None, None, :]

    def prune(self, num, thres=1, random=False):
        # nothing happened
        pass


class TaylorMask(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_buffer('weight', None)  #inizializza un buffer per il peso
        #self.weight = torch.ones(shape) # for compactibility
        #self.register_parameter('weight', None) # weights
        #self.weight = torch.nn.Parameter(torch.ones(shape))
        # if a weight is already pruned, set to True
        self.shape = shape
        self.register_buffer('pruned', None)
        self.pruned = torch.zeros(shape, dtype=torch.bool)  #buffer che memorizza quali parti della maschera sono state
                                                            #potate
        self.values = []  #lista che verrà utilizzata per contenere i gradienti accumulati durante il backpropagation


        '''
        def value_forward_hook(self, input, output):
            self.ouptup = output.detach()
        '''

    def prune(self, num, *args, **kwargs):
        #print('BOOM!!')
        w = self.values  #recupera i gradienti accumulati nel forward: essi rappresentano l'importanza di ciascun elemento
                         #nel k-space
        self.values = []  #resetta la lista dei gradienti per prepararla alla successiva iterazione
        if num == 0:
            # used to reset values only
            return
        assert num > 0 and len(w) > 0
        # exclude $num data with
        with torch.no_grad():
            w = torch.stack(w, 0).mean(0)  #Accumula i gradienti attraverso le diverse iterazioni (batch) e calcola
                                                # la media
            w.masked_scatter_(self.pruned, torch.zeros_like(w))  #Imposta a zero i valori già potati in precedenza
            self.weight = w  #Aggiorna il buffer dei pesi con i nuovi valori di w, che ora rappresentano l'importanza
                             #media dei gradienti.
            w.masked_scatter_(self.pruned, w.max()*torch.ones_like(w))
            _, ind = torch.topk(-w, num)  #Seleziona i num elementi con il valore più basso in w, cioè i meno importanti
                                          # (in termini di gradiente). Questi saranno potati.
            self.pruned.scatter_(-1, ind, torch.ones_like(self.pruned))  #Aggiorna il buffer pruned, marcando le
                                                                         #posizioni selezionate come True, indicandole
                                                                         #come potate.


    def forward(self, image):
        def value_backward_hook(self, grad):
            self.values.append(grad.detach()**2)
            #print('HA!!')

        wrapper = functools.partial(value_backward_hook, self)
        functools.update_wrapper(wrapper, value_backward_hook)

        self.mask = torch.ones(self.shape).to(image)  #inizializza la maschera come un tensore pieno di 1
        self.mask.masked_scatter_(self.pruned, torch.zeros_like(self.mask))  #Applica la maschera di pruning, impostando
                                  #a 0 le posizioni che sono state potate. Questo significa che queste posizioni non
                                  #contribuiranno più all'output finale.
        self.mask.requires_grad=True  #Imposta requires_grad=True per la maschera, permettendo ai gradienti di essere
                                       #calcolati rispetto a essa durante il backpropagation.
        self.mask.register_hook(wrapper)  #Registra l'hook del gradiente (il wrapper) su self.mask
        return image * self.mask[None, None, None, :]

if __name__ == '__main__':
    sparsity = 1.0/8  #percentuale del numero totale di elementi che dovrebbero essere mantenuti nella maschera
    print(sparsity)
    shape = 64  #dimensione della maschera che verrà generata
    example = torch.rand(5, 2, shape*2, shape)  #tensor che rappresenta un batch di 5 immagini con 2 canali e una
                                                #dimensione spaziale di 128x64.
    mask = Mask(shape)  #istanzia un oggetto della classe Mask con dimensione specificata
    print(mask.pruned.numpy().astype(np.float).mean())  #verifica quanto della maschera è stato inizialmente potato
    random_mask = RandomMask(sparsity, shape)  #Istanzia un oggetto RandomMask che crea una maschera casuale basata sulla
                                               #sparsenza specificata (0,125) e la dimensione (64).
    print(random_mask.pruned.numpy().astype(np.float).mean())  #stampa la % di elementi pruned nella RandomMask
    taylor = TaylorMask(shape)  #Istanzia un oggetto della classe TaylorMask
    print(taylor.pruned.numpy().astype(np.float).mean())
    lowpass = LowpassMask(sparsity, shape)  #Istanzia un oggetto della classe LowpassMask
    print(lowpass.pruned.numpy().astype(np.float).mean())
    equispaced = EquispacedMask(sparsity, shape)  #Istanzia un oggetto della classe EquispacedMask
    print(equispaced.pruned.numpy().astype(np.float).mean())
    loupe = LOUPEMask(sparsity, shape)  #Istanzia un oggetto della classe LOUPEMask
    print(loupe.pruned.numpy().astype(np.float).mean())

