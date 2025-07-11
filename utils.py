# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 09:20:31 2022
@author: sun kaicong combined with the work of XuanKai: https://github.com/woxuankai/SpatialAlignmentNetwork
"""
import torch.nn as nn
from timm.models.layers import to_2tuple
import math
import torch
import numpy as np
import torch.fft
import torch.nn.functional as F
import nibabel as nib
import h5py


def fft2(x):
    assert len(x.shape) == 4
    x = torch.fft.fft2(x, norm='ortho')
    return x


def ifft2(x):
    assert len(x.shape) == 4
    x = torch.fft.ifft2(x, norm='ortho')
    return x


def fftshift2(x):
    if len(x.shape) == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif len(x.shape) == 3:
        x = x.unsqueeze(0)

    assert len(x.shape) == 4
    x = torch.roll(x, (x.shape[-2] // 2, x.shape[-1] // 2), dims=(-2, -1))
    return x


def ifftshift2(x):
    assert len(x.shape) == 4
    x = torch.roll(x, ((x.shape[-2] + 1) // 2, (x.shape[-1] + 1) // 2), dims=(-2, -1))
    return x


def rss(x):
    assert len(x.shape) == 4
    return torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)


def rss2d(x):
    assert len(x.shape) == 2
    return (x.real ** 2 + x.imag ** 2).sqrt()


def ssimloss(X, Y):
    assert not torch.is_complex(X)
    assert not torch.is_complex(Y)
    win_size = 7
    k1 = 0.01
    k2 = 0.03
    w = torch.ones(1, 1, win_size, win_size).to(X) / win_size ** 2
    NP = win_size ** 2
    cov_norm = NP / (NP - 1)
    data_range = 1
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    ux = F.conv2d(X, w)
    uy = F.conv2d(Y, w)
    uxx = F.conv2d(X * X, w)
    uyy = F.conv2d(Y * Y, w)
    uxy = F.conv2d(X * Y, w)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    A1, A2, B1, B2 = (
        2 * ux * uy + C1,
        2 * vxy + C2,
        ux ** 2 + uy ** 2 + C1,
        vx + vy + C2,
    )
    D = B1 * B2
    S = (A1 * A2) / D
    return 1 - S.mean()


def gaussian_kernel_1d(sigma, device):
    kernel_size = int(2 * math.ceil(sigma * 2) + 1)
    x = torch.linspace(-(kernel_size - 1) // 2, (kernel_size - 1) // 2, kernel_size, device=device)
    kernel = 1.0 / (sigma * math.sqrt(2 * math.pi)) * torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / torch.sum(kernel)
    return kernel


def gaussian_kernel_2d(sigma, device):
    y_1 = gaussian_kernel_1d(sigma[0], device)
    y_2 = gaussian_kernel_1d(sigma[1], device)
    kernel = torch.tensordot(y_1, y_2, 0)
    kernel = kernel / torch.sum(kernel)
    return kernel


def gaussian_smooth(img, sigma):
    sigma = max(sigma, 1e-12)
    #imposto il dispositivo di img
    device = img.device
    kernel = gaussian_kernel_2d((sigma, sigma), device)[None, None, :, :]
    padding = kernel.shape[-1] // 2
    img = torch.nn.functional.conv2d(img, kernel, padding=padding)
    return img


def compute_marginal_entropy(values, bins, sigma):
    normalizer_1d = np.sqrt(2.0 * np.pi) * sigma
    sigma = 2 * sigma ** 2
    p = torch.exp(-((values - bins).pow(2).div(sigma))).div(normalizer_1d)
    p_n = p.mean(dim=1)
    p_n = p_n / (torch.sum(p_n) + 1e-10)
    return -(p_n * torch.log(p_n + 1e-10)).sum(), p


def _mi_loss(I, J, bins, sigma):
    # compute marjinal entropy
    ent_I, p_I = compute_marginal_entropy(I.view(-1), bins, sigma)
    ent_J, p_J = compute_marginal_entropy(J.view(-1), bins, sigma)
    # compute joint entropy
    normalizer_2d = 2.0 * np.pi * sigma ** 2
    p_joint = torch.mm(p_I, p_J.transpose(0, 1)).div(normalizer_2d)
    p_joint = p_joint / (torch.sum(p_joint) + 1e-10)
    ent_joint = -(p_joint * torch.log(p_joint + 1e-10)).sum()

    return -(ent_I + ent_J - ent_joint)


def mi_loss(I, J, bins=64, sigma=1.0 / 64, minVal=0, maxVal=1):
    bins = torch.linspace(minVal, maxVal, bins).to(I).unsqueeze(1)
    neg_mi = [_mi_loss(I, J, bins, sigma) for I, J in zip(I, J)]
    return sum(neg_mi) / len(neg_mi)


def ms_mi_loss(I, J, bins=64, sigma=1.0 / 64, ms=3, smooth=3, minVal=0, maxVal=1):
    smooth_fn = lambda x: torch.nn.functional.avg_pool2d(
        gaussian_smooth(x, smooth), kernel_size=2, stride=2)
    loss = mi_loss(I, J, bins=bins, sigma=sigma, minVal=minVal, maxVal=maxVal)
    for _ in range(ms - 1):
        I, J = map(smooth_fn, (I, J))
        loss = loss + mi_loss(I, J, bins=bins, sigma=sigma, minVal=minVal, maxVal=maxVal)
    return loss / ms


def lncc_loss(I, J, win=None):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

    ndims = len(list(I.size())) - 2
    assert ndims == 2, "volumes should be 2 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    # I2 = I*I
    # J2 = J*J
    # IJ = I*J

    sum_filt = torch.ones([1, 1, *win]).to(I)

    pad_no = math.floor(win[0] / 2)

    stride = (1, 1)
    padding = (pad_no, pad_no)

    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross * cross / (I_var * J_var + 1e-5)

    return -1 * torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross


def ms_lncc_loss(i, j, win=None, ms=3, sigma=3):
    smooth_fn = lambda x: torch.nn.functional.avg_pool2d(gaussian_smooth(x, sigma), kernel_size=2, stride=2)
    loss = lncc_loss(i, j, win)
    for _ in range(ms - 1):
        i, j = map(smooth_fn, (i, j))
        loss = loss + lncc_loss(i, j, win)
    return loss / ms


def correlation_loss(SM):
    B, C, H, W = SM.shape
    SM_ = SM.view(B, C, -1)
    loss = 0
    for i in range(B):
        cc = torch.corrcoef(SM_[i, ...])
        loss += F.l1_loss(cc, torch.eye(C).to(SM.device))
    return loss


def gradient(x, h_x=None, w_x=None):
    if h_x is None and w_x is None:
        h_x = x.size()[2]
        w_x = x.size()[3]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    xgrad = torch.pow(torch.pow((r - l), 2) + torch.pow((t - b), 2), 0.5)
    return xgrad


def convert(nii_path, h5_path, protocal):
    # convert nii file with path nii_path to h5 file stored at h5_path
    # protocal name as string
    h5 = h5py.File(h5_path, 'w')
    nii = nib.load(nii_path)
    array = nib.as_closest_canonical(nii).get_fdata()  # convert to RAS
    array = array.T.astype(np.float32)
    h5.create_dataset('image', data=array)
    h5.attrs['max'] = array.max()
    h5.attrs['acquisition'] = protocal
    h5.close()


def crop_Kspace(Kspace_f, SR_scale):
    Kspace_f = fftshift2(Kspace_f)
    B, C, H, W = Kspace_f.shape
    if SR_scale != 1:
        margin_H = int(H * (SR_scale - 1) / SR_scale // 2)
        margin_W = int(W * (SR_scale - 1) / SR_scale // 2)
        Kspace_f = Kspace_f[:, :, margin_H:margin_H + H // SR_scale,
                   margin_W:margin_W + W // SR_scale]  # roll it by half
    Kspace_f = ifftshift2(Kspace_f)
    return Kspace_f


def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
    assert torch.is_complex(x)
    #print(f"Forma del tensore di input: {x.shape}")
    result=torch.cat([x.real, x.imag], dim=1)
    #print(f"Forma del tensore di output: {result.shape}")
    return result


def chan_dim_to_complex(x: torch.Tensor) -> torch.Tensor:
    assert not torch.is_complex(x)
    _, c, _, _ = x.shape

    # Se c è dispari, aggiungi un canale extra con valori 0 (puoi anche usare un altro approccio)
    if c % 2 != 0:
        # Aggiungi un canale con valori zero per rendere la dimensione pari
        x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)  # Aggiungi un canale con valori 0
        _, c, _, _ = x.shape  # Ricalcola la nuova dimensione del canale

    #print(f"Shape prima di complex_to_chan_dim: {x.shape}")
    assert c % 2 == 0  # Ora dovrebbe essere sempre pari
    c = c // 2
    return torch.complex(x[:, :c], x[:, c:])

def UpImgComplex(img_complex, SR_scale):
    img_real = complex_to_chan_dim(img_complex)
    img_real = nn.functional.interpolate(img_real, scale_factor=SR_scale, mode='bicubic')
    return chan_dim_to_complex(img_real)


def norm(x: torch.Tensor):
    # group norm
    b, c, h, w = x.shape
    assert c % 2 == 0
    x = x.view(b, 2, c // 2 * h * w)

    mean = x.mean(dim=2).view(b, 2, 1)
    std = x.std(dim=2).view(b, 2, 1)

    x = (x - mean) / (std + 1e-12)

    return x.view(b, c, h, w), mean, std


def unnorm(
        x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
):
    b, c, h, w = x.shape
    assert c % 2 == 0
    x = x.view(b, 2, c // 2 * h * w)
    x = x * std + mean
    return x.view(b, c, h, w)


def preprocess(x):
    assert torch.is_complex(x)
    # x, mean, std = norm(x)
    x = complex_to_chan_dim(x)
    x, mean, std = norm(x)
    return x, mean, std


def postprocess(x, mean, std):
    x = unnorm(x, mean, std)
    x = chan_dim_to_complex(x)
    return x


def pad(x, window_size):
    _, _, h, w = x.size()
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x, (mod_pad_w, mod_pad_h)


def unpad(
        x: torch.Tensor,
        w_pad: int,
        h_pad: int
) -> torch.Tensor:
    return x[..., 0: x.shape[-2] - h_pad, 0: x.shape[-1] - w_pad]


def check_image_size(img_size, window_size):
    h, w = img_size
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size
    return h + mod_pad_h, w + mod_pad_w


def sens_expand(image: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    #print(f"dim immagine prima dell'espansione: {image.shape}")
    #print(f"dim della mappa di sensibilità: {sens_maps.shape}")
    res=fft2(image * sens_maps)
    #print(f"forma dopo l'espansione: {res.shape}")
    return res


def sens_reduce(kspace: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    #print(f"la forma di kspace dentro sens_reduce è: {kspace.shape}")
    #print(f"la forma di sens_maps dentro sens_reduce è: {sens_maps.shape}")

    # Applicazione della trasformata inversa di Fourier
    kspace_ifft = ifft2(kspace)  # Risultato della trasformata inversa di Fourier

    # Moltiplicazione per le mappe di sensibilità con il coniugato
    result = kspace_ifft * sens_maps.conj()

    # Somma lungo i canali (dim=1)
    reduced_result = result.sum(dim=1, keepdim=True)  # Mantenere la dimensione del canale

    #print(f"forma dopo la somma: {reduced_result.shape}")
    return reduced_result


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 2.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size=4, in_chans=2, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = to_2tuple(patch_size)
        self.patches_resolution = tuple([self.img_size[0] // patch_size, self.img_size[1] // patch_size])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # B N_patch C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size. 
        patch_size (int): Patch token size. Default: 4.
        out_chans (int): Number of output image channels. Default: 2.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size=4, out_chans=2, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = tuple([self.img_size[0] // patch_size, self.img_size[1] // patch_size])
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(embed_dim, out_chans, kernel_size=int(np.floor(patch_size * 1.5)), stride=1,
                              padding='same')
        #### with gated conv ####
        self.gate_proj = nn.Conv2d(embed_dim, out_chans, kernel_size=int(np.floor(patch_size * 1.5)), stride=1,
                                   padding='same')
        self.act_layer = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.scale = nn.Parameter(torch.rand(1))

    def forward(self, x):
        x = x.transpose(1, 2).view(-1, self.embed_dim, self.patches_resolution[0], self.patches_resolution[1])
        x = nn.functional.interpolate(x, scale_factor=self.patch_size, mode='bicubic')
        #### with gated conv ####
        x_origin = self.act_layer(self.proj(x))
        x_gate = torch.sigmoid(self.scale * self.gate_proj(x))
        x = x_origin * x_gate
        return x

    def flops(self):
        flops = 0
        return flops


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class Upsample(nn.Sequential):  #ESEGUE L'UPSAMPLING DI FEATURES MAPS O IMMAGINI. è una classe di nn.Sequential che
                                #permette di combinare una sequenza di moduli di un'unica rete neurale
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):  #prende in ingresso un fattore di scala per l'upsampling e il n di canali usati
                                          #per operazioni intermedie
        m = []   #lista che conterrà i moduli per il blocco di upsampling. questi moduli saranno successivamente passati
                 #a nn.Sequential per essere eseguiti in sequenza

        if (scale & (scale - 1)) == 0:  # scale = 2^n.  verifica se il pattore di scala è una potenza di 2
            for _ in range(int(math.log(scale, 2))): #ciclo che itera il n° di volte necessario per ottenere il
                                                          #fattore di scala desiderato
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))   #raddoppia la risoluzione spaziale e il n° di canali viene ridotto di un
                                               #fattore di 4
        elif scale == 3:  #caso in cui il fattore di scala è una potenza di 3
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:  #gestisce l'errore nel caso in un cui il fattore non sia potenza di 2 o 3
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')

        super(Upsample, self).__init__(*m)  #i moduli definiti in "m" vengono passati al costruttore della classe base
                                            #(nn.Sequential) che crea una sequenza di moduli da eseguire uno dopo l'altro


class UpsampleOneStep(nn.Sequential):   #ESEGUE L'UPSAMPLING MA IN UNA VERSIONE PIU SEMPLIFICATA per i contesti in cui è
                                        #necessario risparmiare parametri e calcoli
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution  #risoluzione dell'input
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):   #calcola la complessità computazionale, fornendo un'indicazione di quante operazioni di floating-
                       #point vengono eseguite durante il processo di upsampling.
        H, W = self.input_resolution  #H e W sono le dimensioni spaziali dell'input
        flops = H * W * self.num_feat * 3 * 9
        return flops


class PatchMerging(nn.Module):   #si occupa di "mergere" (combinare) le patch di immagini in un modo che riduce la
                                 #dimensione spaziale e aumenta la dimensionalità dei canali.
                                 #Le "patch" nel contesto del codice che hai fornito si riferiscono a segmenti o porzioni
                                 #dell'immagine (o della feature map) che vengono estratti e trattati separatamente.
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)   #layer di convoluzione lineare
        self.norm = norm_layer(4 * dim)   #layer che normalizza le feature map

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution  #estrae H e W dalla tupla di risoluzione
        B, L, C = x.shape  #Questa riga estrae le dimensioni del tensor x che rappresenta l'input al metodo forward del modulo PatchMerging
        assert L == H * W, "input feature has wrong size"   #L è in n° totale di posizioni spaziali nel tensore
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)   #Reshape del tensor per ottenere una forma (B, H, W, C)

        #estrazione delle patch 2x2 dall'immagine originale. ogni variabile rappresenta una parte della griglia 2x2 dell'immagine
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C  Concatenazione delle patch lungo la dimensione dei
                                                    # canali, producendo un tensor di dimensioni (B, H/2, W/2, 4*C)
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C  Riorganizzazione in una forma di (B, H/2 * W/2, 4 * C) per preparare
                                  # il tensor alla normalizzazione e riduzione dei canali.

        x = self.norm(x)   #Applicazione della normalizzazione alle feature map.
        x = self.reduction(x)   #Applicazione della riduzione dei canali

        return x

    def extra_repr(self) -> str:  #Fornisce una rappresentazione aggiuntiva della classe che è utile per il debugging e
                                  #la visualizzazione della configurazione del modulo.
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # print('in_features', in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)  #primo livello lineare che mappa da in a hidden
        self.act = act_layer()  #viene applicata la funzione di attivazione dopo il primo livello
        self.fc2 = nn.Linear(hidden_features, out_features)  #secondo livello lineare che mappa da hidden a out
        self.drop = nn.Dropout(drop)   #Il livello di dropout, applicato dopo le funzioni di attivazione e il secondo
                                       #livello lineare per ridurre l'overfitting.

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x   #restituisce l'output della rete che ha "out_features" dimensioni


def window_partition(x, window_size):   #SUDDIVIDE UN TENSORE 4D IN UNA SERIE DI FINESTRE O "PATCH" DI DIMENSIONI FISSE
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape  #estrae le dimensioni di x
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)   #si sta suddividendo l'immagine in
                                                                                     # patch di dimensione window_size x
                                                                                     #window_size.
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    #Cambia l'ordine delle dimensioni del tensor per riordinare le dimensioni delle finestre, garantisce che i dati del
    #tensor siano memorizzati in modo contiguo in memoria e riformatta il tensore
    return windows   #Restituisce il tensor che ora contiene tutte le finestre estratte, dove ogni finestra ha dimensioni
                     #(window_size, window_size, C).


def window_reverse(windows, window_size, H, W):   #RIPORTAA UN TENORE SUDDIVISO IN UNA SERIE DI FINESTRE IN UNO 4D
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


