# -*- coding: utf-8 -*-
"""
@author: sunkg
"""
import torch
import torch.nn as nn
import ViT
import CNN
import utils
import sensitivity_model
from torch.cuda.amp import autocast

#PER RISOLVERE PROBLEMI INPUT/PESI
def ensure_device(tensor, device):
    """
    Garantisce che un tensore si trovi sul dispositivo specificato.
    Se il tensore non è già sul dispositivo, lo sposta.
    """
    if tensor.device != device:
        return tensor.to(device)
    return tensor


#RETE PER ELABORAZIONE DI IMG MEDICHE IN FORMATO COMPLESSO
class DCRB(nn.Module):
    #inizializzazione parametri
    def __init__(self, coils_all, img_size, num_heads, window_size, patch_size =1,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, n_SC=1, embed_dim=96, ds_ref=True, scale = 0.1):
        super().__init__()
        
        #### same stepsize for re and im ###
        self.stepsize = nn.Parameter(0.1*torch.rand(1))   #monitora dim del passo durante l'aggiornamento dell'img nel
                                                          #processo iterativo
        self.LeakyReLU = nn.LeakyReLU()   #funzione attivazione
        self.img_size = img_size   #dim immagine
        self.coils_all = coils_all   #n° delle bobbine
        self.ds_ref = ds_ref   #booleano per includere o meno la mappa dei gradienti
        self.scale = scale   #fattore di scala.

        #rete CNN: NormNet
        self.CNN = CNN.NormNet(in_chans = coils_all*2, out_chans = coils_all, chans = 32) # using Unet for K-space, 2 is for real and imaginary channels

        #rete ViT
        self.ViT = ViT.ViT(dim = 2, img_size=img_size, num_heads=num_heads, window_size=window_size, patch_size=patch_size,
                           mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                           drop_path=drop_path, norm_layer=nn.LayerNorm, n_SC=n_SC, embed_dim=embed_dim, ds_ref=ds_ref) # each of T1 and T2 has two channels for real and imaginary values
    
    
    def forward(self, Ref_img, Ref_Kspace_f, Target_Kspace_u, Target_img_f, mask, sens_maps_updated, idx, gate): 
        #processamento dei dati
        Target_Kspace_f = utils.sens_expand(Target_img_f, sens_maps_updated)
        Ref_Kspace_f = utils.complex_to_chan_dim(Ref_Kspace_f)
        Target_Kspace_f = utils.complex_to_chan_dim(Target_Kspace_f)
        Ref_img = utils.complex_to_chan_dim(Ref_img)
        Target_img_f = utils.complex_to_chan_dim(Target_img_f)
        # Stampa le forme
        #print(f"Forma di Ref_Kspace_f prima della convoluzione: {Ref_Kspace_f.shape}")
        #print(f"Forma di Target_Kspace_f prima della convoluzione: {Target_Kspace_f.shape}")

        # Controlla se il numero di canali è 70, in tal caso ignora questo dato e prosegui
        if Ref_Kspace_f.shape[1] == 70:
            #print(f"Numero di canali di Ref_Kspace_f inatteso: {Ref_Kspace_f.shape[1]}, ignorando questo dato.")
            return Target_img_f.clone() # Salta il dato e termina l'elaborazione

        if Target_Kspace_f.shape[1] == 70:
            #print(f"Numero di canali di Target_Kspace_f inatteso: {Target_Kspace_f.shape[1]}, ignorando questo dato.")
            return Target_img_f.clone()  # Salta il dato e termina l'elaborazione

        device = Ref_Kspace_f.device
        #Controlla e modifica il numero di canali per il K_space_f
        if Ref_Kspace_f.shape[1] != 4:
            conv_ref = torch.nn.Conv2d(in_channels=38, out_channels=2, kernel_size=1).to(device)
            Ref_Kspace_f = conv_ref(Ref_Kspace_f)

        #Controllo e modifico il numero di canali per Target_Kspace_f
        if Target_Kspace_f.shape[1] != 4:
            conv_target = torch.nn.Conv2d(in_channels=38, out_channels=2, kernel_size=1).to(device)
            Target_Kspace_f = conv_target(Target_Kspace_f)

        #Ref_img = utils.complex_to_chan_dim(Ref_img)
        #Target_img_f = utils.complex_to_chan_dim(Target_img_f)


        #print("Forma di Ref_Kspace_f dopo la riduzione:", Ref_Kspace_f.shape)
        #print("Forma di Target_Kspace_f dopo la riduzione:", Target_Kspace_f.shape)

        device = mask.device
        #todo: rimuovi se non necessario
        if Ref_Kspace_f is None or Target_Kspace_f is None:
            #print("I dati non sono validi. Ignorando l'elaborazione di questo dato.")
            return Target_img_f  # Ignora l'elaborazione se i dati non sono validi
            
        #aggiunta mappa dei gradienti se condizione=True
        if self.ds_ref:
            Ref_rss = utils.rss(Ref_img)
            Ref_grad = utils.gradient(Ref_rss)
            Ref_img = torch.cat([Ref_img, Ref_grad], dim=1)
            #print(f"Forma dell'immagine di riferimento dopo aggiunta gradiente: {Ref_img.shape}")

        #prepara input alla rete CNN e ViT
        input_CNN = torch.cat([Ref_Kspace_f, Target_Kspace_f], 1)
        input_ViT = [Target_img_f, Ref_img]
        #passa gli input attraverso le reti
        output_CNN = self.CNN(input_CNN)
        #print("Dimensioni input_ViT[0]:", input_ViT[0].shape)
        #print("Dimensioni input_ViT[1]:", input_ViT[1].shape)
        output_ViT = self.ViT(input_ViT[0], input_ViT[1])

        #converte nuovamente le uscite in formato complesso
        output_CNN = utils.chan_dim_to_complex(output_CNN)      
        output_ViT = utils.chan_dim_to_complex(output_ViT)
        Target_img_f = utils.chan_dim_to_complex(Target_img_f)        

        #calcola due termini di aggiornamento
        Target_Kspace_f_down = utils.sens_expand(Target_img_f, sens_maps_updated)

        def split_complex(tensor):
            real = tensor.real
            imag = tensor.imag
            return torch.cat([real, imag], dim=1)  # Concatenazione lungo la dimensione dei canali

        # Separazione dei componenti reali e immaginari per 'Target_Kspace_f_down'
        if (mask * (mask * Target_Kspace_f_down - Target_Kspace_u)).shape[1] != sens_maps_updated.shape[1]:
            kspace_split = split_complex(mask * (mask * Target_Kspace_f_down - Target_Kspace_u))
            device = kspace_split.device
            conv = torch.nn.Conv2d(in_channels=kspace_split.shape[1], out_channels=sens_maps_updated.shape[1],
                                   kernel_size=1).to(device)
            Target_Kspace_f_down = conv(kspace_split)

        # Separazione dei componenti reali e immaginari per 'output_CNN'
        if output_CNN.shape[1] != sens_maps_updated.shape[1]:
            output_CNN_split = split_complex(output_CNN)
            device = output_CNN_split.device
            conv = torch.nn.Conv2d(in_channels=output_CNN_split.shape[1], out_channels=sens_maps_updated.shape[1],
                                   kernel_size=1).to(device)
            output_CNN = conv(output_CNN_split)
        term1 = 2*utils.sens_reduce(mask*(mask*Target_Kspace_f_down-Target_Kspace_u), sens_maps_updated)
        term2 = utils.sens_reduce(output_CNN, sens_maps_updated)
        #print(f"Forma di term1: {term1.shape}")
        #print(f"Forma di term2: {term2.shape}")
        
        #aggiorna l'immagine target
        Target_img_f = Target_img_f-self.stepsize*(term1+self.scale*term2+self.scale*output_ViT)
        #print(f"Forma finale di Target_img_f: {Target_img_f.shape}")

        return Target_img_f
        

class fD2RT(nn.Module):
    def __init__(self, coils, img_size, num_heads, window_size, patch_size = None,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, n_SC=1, num_recurrent=5, embed_dim=96, sens_chans=8,
                 sens_steps=4, mask_center=True, ds_ref=True, scale = 0.1):
        super().__init__()
        #inizializza la rete SensitivityModel: calcola mappe sansibilità
        self.sens_net = sensitivity_model.SensitivityModel(
            chans=sens_chans,
            sens_steps=sens_steps,
            mask_center=mask_center
        )
        #print(f"Initial sensitivity channels: {sens_chans}")

        self.ds_ref = ds_ref
        self.scale = scale  # scaling layer 
        self.coils = coils//2 # coils of single modality
        self.stepsize = nn.Parameter(0.1*torch.rand(1))   #monitora la dim del passo durante l'aggiornamento dell'img
        #blocchi DCRB
        self.num_recurrent = num_recurrent
        self.recurrent = nn.ModuleList([DCRB(coils_all=coils, img_size=img_size, num_heads=num_heads, window_size=window_size,  
                         patch_size = int(patch_size[i]), mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                         attn_drop=attn_drop, drop_path=drop_path, norm_layer=nn.LayerNorm, n_SC=n_SC, embed_dim=embed_dim, 
                         ds_ref= ds_ref, scale = scale) for i in range(num_recurrent)])
        
        #blocchi CONVOLUZIONALI usati in SMRB
        self.ConvBlockSM = nn.ModuleList([CNN.ConvBlockSM(in_chans = 2, conv_num = 2) for _ in range(num_recurrent-1)])
    
    #funzione responsabile dell'aggiornamento delle mappe
    def SMRB(self, Target_img_f, sens_maps_updated, Target_Kspace_u, mask, gate, idx):
        Target_Kspace_f = utils.sens_expand(Target_img_f, sens_maps_updated)   #espande l'img target nello spazio K
        B, C, H, W = sens_maps_updated.shape
        #prepara le mappe di sensibilità per essere elaborate da ConvBlockSM
        sens_maps_updated_ = sens_maps_updated.reshape(B*C, 1, H, W)
        sens_maps_updated_ = utils.complex_to_chan_dim(sens_maps_updated_)
        sens_maps_updated_ = self.ConvBlockSM[idx](sens_maps_updated_)
        sens_maps_updated_ = utils.chan_dim_to_complex(sens_maps_updated_)
        sens_maps_updated_ = sens_maps_updated_.reshape(B, C, H, W)
        #aggiorna mappe di sensibilità in base alla differenza tra i dati del k-space target e quelli ricostruiti
        sens_maps_updated = sens_maps_updated - self.stepsize*(2*utils.ifft2(mask*(mask*Target_Kspace_f - Target_Kspace_u) * Target_img_f.conj()) + self.scale*sens_maps_updated_)
        sens_maps_updated = sens_maps_updated / (utils.rss(sens_maps_updated) + 1e-12)
        sens_maps_updated = sens_maps_updated * gate
        return sens_maps_updated

            
    def forward(self, Ref_Kspace_f, Target_Kspace_u, mask, num_low_frequencies):
        #print(f"Ref_Kspace_f shape: {Ref_Kspace_f.shape if Ref_Kspace_f is not None else 'None'}")
        #print(f"Target_Kspace_u shape: {Target_Kspace_u.shape}")
        #print(f"Mask shape: {mask.shape}")
        #print(f"Number of low frequencies: {num_low_frequencies}")
        #inizializza variabili di registrazione e le mappe di sensibilità e il gate
        rec = []
        SMs = []
        if self.coils == 1:
            sens_maps_updated = torch.ones_like(Target_Kspace_u)
            gate = torch.ones_like(sens_maps_updated).cuda()

        else:   #Se ci sono più bobine...
                #...utilizza SensitivityModel per inizializzare le mappe di sensibilità e il gate, usando le frequenze
                #basse del K-space.
            sens_maps_updated, gate = self.sens_net(Target_Kspace_u, num_low_frequencies)
            #print(f"forma aggiornata delle mappe di sensibilità iniziali: {sens_maps_updated.shape}")  # Verifica la forma delle mappe aggiornate
            #print(f"forma gate: {gate.shape}")

        if Ref_Kspace_f != None:   #se è fornito un K-space di riferimento...
            #...riduce questo K-space per ottenere un'immagine di riferimento
            Ref_img = utils.sens_reduce(Ref_Kspace_f, sens_maps_updated)
            #print(f"forma img riferimento: {Ref_img.shape}")

        #Inizializza l'immagine target riducendo il k-space target con le mappe di sensibilità
        Target_img_f = utils.sens_reduce(Target_Kspace_u, sens_maps_updated)
        #print(f"dim img target: {Target_img_f.shape}")
        SMs.append(sens_maps_updated)
        rec.append(Target_img_f)
        
        #### DCRB blocks #### 
        for idx, DCRB_ in enumerate(self.recurrent):
            if Ref_Kspace_f == None:
                Ref_img = Target_img_f.clone()
                Ref_Kspace_f = utils.sens_expand(Ref_img, sens_maps_updated)

                #print(f"iterazione {idx} - forma img riferimento: {Ref_img.shape}, forma img target: {Target_img_f.shape}")
                
            #### Update of SM by SMRB ####
            if (self.coils != 1) & (idx != 0):            
                sens_maps_updated = self.SMRB(Target_img_f, sens_maps_updated, Target_Kspace_u, mask, gate, idx-1)            
                SMs.append(sens_maps_updated)
                Ref_img = utils.sens_reduce(Ref_Kspace_f, sens_maps_updated)
                #print(
                    #f"forma delle mappe di sensibilità aggiornate: {sens_maps_updated.shape}")  # Stampa la forma delle mappe di sensibilità aggiornate
                #print(f"forma img di riferimento dopo SMRB: {Ref_img.shape}")

                #### Update of MR image by DCRB ####
            Target_img_f = DCRB_(Ref_img, Ref_Kspace_f, Target_Kspace_u, Target_img_f, mask, sens_maps_updated, idx, gate)
            #print(f"forma img target dopo il blocco DCRB {idx}: {Target_img_f.shape}")

            # Debug per controllare se Target_img_f è valido
            if Target_img_f is None:
                raise ValueError(f"DCRB_ ha restituito None all'iterazione {idx}. Verifica il codice di DCRB_.")
            #print(f"forma img target dopo il blocco DCRB {idx}: {Target_img_f.shape}")

            rec.append(Target_img_f)

            rec.append(Target_img_f)
        #print(f"forma dell'img target finale: {Target_img_f.shape}")
        return rec, utils.rss(Target_img_f), sens_maps_updated, Target_img_f
