#!/usr/bin/env python-3
"""
refer to the work of XuanKai: https://github.com/woxuankai/SpatialAlignmentNetwork
"""
import json
import os.path

import numpy as np
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float32)
'''
Ideas on deduplication between checkpoints:
    1. Substitue the f in torch.save with a io.ByteIO file.
        After torch.save, extract the zipfile to destination.
   *2. Save objects seperately in the same folder with np.savez.
    3. More fine-grained saving: save each key:val in state_dict seperately.
'''


#CARICAMENTO CHECKPOINT
def ckpt_load(folder):
    if os.path.isfile(folder): #se l'argomento 'folder' è un file...
        return torch.load(folder)  #... lo carica usando torch.load e restituisce il contenuto
    ckpt = {}   #se invece 'folder' è una directory, inizializza un dizionario
    for key in os.listdir(folder):   #Itera attraverso tutti i file e le sottodirectory all'interno di folder.
        save_path = os.path.join(folder, key)   #Per ogni elemento, si costruisce il percorso completo del file.

        #caricamento della configurazione
        if key == 'config':  #Se il file corrente è denominato config, si presuppone contenga le impostazioni del modello
                             #o della sessione di allenamento
            try:
                ckpt[key] = Config()   #Si crea un'istanza della classe Config
                ckpt[key].load(save_path)   #Si cerca di caricare le impostazioni dal file.
            except UnicodeDecodeError:   #Se c'è un problema di codifica...
                ckpt[key] = torch.load(save_path)   #... si carica il file come un normale file di checkpoint
                                                    #per tutti i file che non sono la configurazione
        else:
            try:
                ckpt[key] = torch.load(save_path, map_location='cuda:0')   # Si tenta di caricare il file come un checkpoint
                            #PyTorch. map_location='cpu' forza il caricamento del file sulla CPU
            except RuntimeError as e:   #Se il caricamento con PyTorch fallisce...
                ckpt[key] = np.load(save_path)   #... Si carica il file come array NumPy
                ckpt[key] = {k: torch.from_numpy(v).cuda() for k, v in ckpt[key].items()}   #Si converte ogni array NumPy in un
                                                                                     #tensore PyTorch
    return ckpt   #restituisco il dizionario dopo aver caricato tutti i componenti del checkpoint


#SALVARE UN CHECKPOINT
def ckpt_save(ckpt, folder):
    assert isinstance(ckpt, dict)  #verifico che ckpt sia effettivamente un dizionario
    # assert not os.path.exists(folder), folder+' already exists'
    if not os.path.exists(folder):   #se la cartella folder non esiste...
        os.mkdir(folder)   #... viene creata
    for key, val in ckpt.items():   #itera su ciascun elemento del dizionario ckpt
        save_path = os.path.join(folder, key)   #per oggi coppia chiave-valore viene indicato un percorso di salvataggio
                                                #all'interno della cartella folder
        if key == 'config':   #se kay è config...
            val.save(save_path)   #...la funzione presume che il valore associato (val) sia un oggetto con un metodo
                                  #save, e chiama val.save(save_path) per salvarlo
        else:   #per tutte le altre chiavi...
            val = {k: v.cuda().numpy() for k, v in val.items()}  #.. si presume che val sia un dizionario contenente
                                                                # tensori PyTorch
            f = open(save_path, 'wb')
            np.savez(f, **val)  #Poi, il risultato viene salvato in un file .npz
            f.close()


#CONFIGURAZIONE DINAMICA
class Config(object):
    def __init__(self, **params):
        super().__init__()
        super().__setattr__('memo', [])  #Crea una lista memo che memorizza i nomi di tutti gli attributi
                                                        #aggiunti all'oggetto.
        for key, val in params.items():  #I parametri passati come keyword arguments (**params) vengono aggiunti all'
            setattr(self, key, val)      #oggetto come attributi, e i loro nomi vengono memorizzati in memo.

    def __setattr__(self, name, value):
        if name not in self.memo:
            self.memo.append(name)
        super().__setattr__(name, value)  #imposta effettivamente il valore dell'attributo

    def __delattr__(self, name):
        self.memo.remove(name)
        super().__delattr__(name)  #elimina effettivamente il valore dell'attributo

    def __str__(self):
        return 'class Config containing: ' \
            + str({key: getattr(self, key) for key in self.memo})

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, param):
        assert param in self.memo, str(param) + ' not found, try ' + str(self.memo)  #verifica che l'attributo richiesto
                                                                                     #esista in memo
        return getattr(self, param)   #resistuisce il valore dell'attributo richiesto

    def __contains__(self, item):
        return item in self.memo  #resituisce True se l'attributo è presente in memo, altrimenti False

    def load(self, save_path):
        for k in self.memo.copy():  #Itera su una copia di memo per eliminare tutti gli attributi esistenti.
            self.pop(k)  #rimuove l'attributo dall'oggetto.
        f = open(save_path, 'r')  #apre il file in modalità lettura
        content = json.load(f)  #Carica il contenuto del file JSON come dizionario.
        f.close()  #chiude il file
        for k, v in content.items():  #Itera sui parametri del dizionario caricato...
            setattr(self, k, v)   #... li imposta come attributi dell'oggetto.

    def save(self, save_path):
        content = {k: getattr(self, k) for k in self.memo}  #Crea un dizionario con tutti gli attributi memorizzati in memo.
        f = open(save_path, 'w')  #apre il file in modalità scrittura
        json.dump(content, f)   # Scrive il dizionario nel file JSON.
        f.close()   #chiude il file


#GESTIONE E SALVATAGGIO DEL MODELLO
class BaseModel(nn.Module):
    def __init__(self, cfg=None, ckpt=None, objects=None):
        super(BaseModel, self).__init__()
        if ckpt is not None:
            self.load(cfg=cfg, ckpt=ckpt, objects=objects)
        else:
            self.build(cfg=cfg)
        self.training = True  #indica che il modello inizialmente è in modalità di addestramento

    def build(self, cfg):
        self.cfg = cfg  #salva la configurazione come attributo della classe, rendendola accessibile ad altri metodi

    def to(self, device):
        for value in self.__dict__.values():  #itera sugli attributi della classe
            if isinstance(value, torch.nn.Module) \
                    or isinstance(value, torch.Tensor):   #se l'attributo è un tensore o un modulo...
                value.to(device)   #... li trasferisce sul dispositivo specificato
            if isinstance(value, torch.optim.Optimizer):   #se l'attributo è un ottimizzatore...
                for param in value.state.values():   #...itera sui parametri dell'ottimizzatore...
                    if isinstance(param, torch.Tensor):   #se il parametro è un tensore...
                        param.data = param.data.to(device)   #...lo trasfrisce al dispositivo specificato
                        if param._grad is not None:   #se il gradiente del parametro è None...
                            param._grad.data = param._grad.data.to(device)   #...lo trasferisce
                    elif isinstance(param, dict):   #se l'attributo è un dizionario...
                        for subparam in param.values():   #itera su tutti i sottoparametri del dizionario...
                            if isinstance(subparam, torch.Tensor):   #... se sono tensori...
                                subparam.data = subparam.data.to(device)   #li trasferisce nel dispositivo specificato
                                if subparam._grad is not None:   #se il gradiente del parametro è None...
                                    subparam._grad.data = \
                                        subparam._grad.data.to(device)   #... lo trasferisce
        return self   #restituisce l'istanza della classe per facilitare il chaining

    def train(self, mode=True):
        for value in self.__dict__.values():   #itera sugli attributi della classe
            if isinstance(value, torch.nn.Module):   #se l'attributo è un modulo...
                value.train(mode)   #...imposta il modulo in modalità di addestramento
        self.training = mode
        return self

    def eval(self):
        for value in self.__dict__.values():
            if isinstance(value, torch.nn.Module):
                value.eval()
        self.training = False   #disabilita l'allenamento
        return self

    def get_saveable(self):
        if self.__dict__.get('_modules') is not None:   #controlla se _modules è presente negli attributi.
            modules = self.__dict__.get('_modules')   #Ottiene tutti i moduli dal dizionario _modules.
            return {key: value for key, value in modules.items() \
                    if isinstance(value, torch.nn.Module)}   #resituisce solo moduli che sono istanze di torch.nn.Module
        else:
            return {key: value for key, value in self.__dict__.items() \
                    if isinstance(value, torch.nn.Module)}  #Se _modules non è presente, itera sugli attributi normali.

    def save(self, ckpt, objects=None):
        saveable = self.get_saveable()   #Ottiene tutti i moduli salvabili.
        if objects is None:   #Se non vengono specificati oggetti...
            objects = saveable.keys()   #...utilizza tutte le chiavi dei moduli salvabili.
        objects = {key: saveable[key].state_dict() for key in objects}   #Ottiene lo stato di ciascun modulo salvabile.
        if hasattr(self, 'cfg'):   #Verifica se la configurazione (cfg) è presente come attributo.
            objects['config'] = self.cfg   #Aggiunge la configurazione all'elenco degli oggetti da salvare
        else:
            print('!!! Missing cfg while saving !!!')
        ckpt_save(objects, ckpt)   #Chiama la funzione ckpt_save per salvare gli oggetti nel checkpoint.

    def load(self, ckpt, cfg=None, objects=None):
        ckpt = ckpt_load(ckpt)   #Carica il checkpoint specificato usando la funzione ckpt_load.
        if cfg is None:   #Se la configurazione non è specificata...
            cfg = ckpt.pop('config')   #...la estrae da checkpoint
        self.build(cfg=cfg)   #Costruisce il modello utilizzando la configurazione caricata.
        saveable = self.get_saveable()   #Ottiene tutti i moduli salvabili.
        if objects is None:   #Se non vengono specificati oggetti...
            objects = saveable.keys()   #...utilizza tutte le chiavi dei moduli salvabili
        objects = {key: saveable[key] for key in objects}   #Prepara gli oggetti per il caricamento.
        for key, value in objects.items():   #Itera sugli oggetti...
            value.load_state_dict(ckpt[key])   #...e carica il loro stato dal checkpoint.


#FUNZIONE DI TEST
def test_main():
    ckpt = ckpt_load(sys.argv[1])  #carica un ckpt da un percorso specificato come 1° argomento della riga di comando
                                   #sys.argv è una lista che contiene gli argomenti passati al programma dalla linea di
                                   #comando. [1] rappresenta il primo argomento passato
    if len(sys.argv) >= 3:   #se sono stai passati almeno 3 argomenti dalla linea di comando
        ckpt_save(ckpt, sys.argv[2])  #salva il ckpt usando il 2° argomento come percorso di destinazione
    else:                    #se sono stati passati meno di 3 argomenti...
        if os.path.isdir(sys.argv[1]):  #se il percorso specificato nel 1° argomento è una directory...
            shutil.rmtree(sys.argv[1])  #...la rimuove
        elif os.path.isfile(sys.argv[1]):  #se il percorso specificato nel 1° argomento è un file...
            os.remove(sys.argv[1])   #...lo rimuove
        else:                 #se non è ne un file ne una directory...
            assert False      #...manda un avviso
        ckpt_save(ckpt, sys.argv[1])   #dopo aver rimosso file o directory esistente,salva il ckpt nel percorso specificato
    '''
    cfg = Config(var1=1, var2=2)
    cfg.var3 = 3
    del cfg.var2
    print(cfg)
    cfg.var1
    cfg['var1']
    model = BaseModel(cfg)
    model.save('/tmp/feel_free_to_delete_it.pth')
    model = BaseModel('/tmp/feel_free_to_delete_it.pth')
    '''


if __name__ == '__main__':
    import sys
    import os
    import shutil

    test_main()
