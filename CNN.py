"""
Oct 12, 2021
Combined and modified by Kaicong Sun <sunkc@shanghaitech.edu.cn>
"""

import math
from typing import List, Tuple
import torch
from torch import nn
from torch.nn import functional as f
import utils
torch.set_default_dtype(torch.float32)


#ResNet: RISOLUZIONE PROBLEMI LEGATI ALL'ADDESTRAMENTO
class ResNet(nn.Module):
    """
    A convolutional network with residual connections.
    """

    def __init__(self, chans_in, chans_max, num_conv, channel_scale=2):
        """
        Args:
            chans_in: Number of channels in the input.
            chans_max: Number of maximum channels
            out_chans: Number of channels in the output.
            channel_scale: combine two modelities in channel.
        """
        super().__init__()
        #definizione delle variabili di classe
        self.chans_in = chans_in   #canali input
        self.chans_out = chans_in // channel_scale   #canale output. 'channel_scale' è usato per ridurre i canali output
        self.chans_max = chans_max   #n° max di canali usati nei livelli conv interni della rete
        self.num_conv = num_conv   #n° di convoluzioni

        #sequenza di layer che compongono il blocco di input: costruisce feature map
        self.input_layers = nn.Sequential(
            nn.Conv2d(self.chans_in, self.chans_max, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.chans_max),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # sequenza di layer che compongono il blocco intermedio
        self.inter_layers = nn.Sequential(
            nn.Conv2d(self.chans_max, self.chans_max, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.chans_max),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # sequenza di layer che compongono il blocco di output
        self.output_layers = nn.Sequential(
            nn.Conv2d(self.chans_max, self.chans_out, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.chans_max),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        #definisco i moduli della rete(liste): utili a riorganizzare e raccogliere i moduli definiti sopra in una
        #struttura ordinata che può essere facilmente combinata in sequenze più grandi
        m_head = [self.input_layers]

        m_body = [self.inter_layers for _ in range(self.num_conv)]

        m_tail = [self.output_layers]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    #definisco il percorso dei dati
    def forward(self, x):
        x_ = self.head(x)
        x_ = self.body(x_)
        res = self.tail(x_)
        output = x[:, self.chans_in // 2:, :, :] + res  #Combina il risultato (res) con una parte dell'input originale

        return output


#UNet: SEGMENTAZIONE DI IMMAGINI
class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    #il costruttore definisce i parametri principali del modello
    def __init__(
            self,
            in_chans: int,  #num canali input
            out_chans: int, #num canali output
            chans: int = 16,  # 32.  num canali primo blocco convoluzionale. questo valore viene raddoppiato ad ogni
                                    #livello di down sampling
            num_pool_layers: int = 4,  #num livelli down-sampling e up-sampling della rete
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        #strati di DOWN-SAMPLING
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans)])
        ch = chans  #ch è utilizzata per tenere traccia del numero corrente di canali mentre si costruiscono i blocchi
                    #di convoluzione.
        for _ in range(num_pool_layers - 1):  #costruire strati di down sampling successivi
            self.down_sample_layers.append(ConvBlock(ch, ch * 2))  #creo un nuovo blocco di  convoluzione con ConvBlock
                                                                   #e lo aggiungo alla lista "down_sample_layers"
            ch *= 2  #raddoppio il numero di canali alla fine di ogni interazione
        self.conv = ConvBlock(ch, ch * 2)  #creo un ulteriore blocco: processa l'output dell'ultimo livello

        #strati di UP-SAMPLING
        self.up_conv = nn.ModuleList()  #ogni blocco eseguirà convoluzione sui tensori concatenati
        self.upsampling_conv = nn.ModuleList()  #ogni blocco eseguirà conv sui tensori prima di concatenarli con quelli
                                                #provenienti dal down-sampling

        for _ in range(num_pool_layers - 1):
            self.upsampling_conv.append(ConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch))
            ch //= 2

        self.upsampling_conv.append(ConvBlock(ch * 2, ch))

        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        assert not torch.is_complex(image)
        stack = []  #memorizza output dei livelli di down per applicarli in quelli di up
        output = image   #memorizza l'output di ogni livello

        # applica down-sampling layers
        for layer in self.down_sample_layers:   #applica ciascun layer di down-sampling ad output
            output = layer(output)    #applica il layer corrente all'output
            stack.append(output)      #aggiunge l'output corrente allo stack per conservarlo
            output = f.avg_pool2d(output, kernel_size=2, stride=2, padding=0)  #fa un average pooling per ridurre la
                                                                               #risoluzione spaziale all'output

        output = self.conv(output)  #applica un uteriore convoluzione al tensore ridotto

        for up_conv, conv in zip(self.upsampling_conv, self.up_conv):  #per ciascun livello di up-sampling vengono usate
                                                                       #coppie di blocchi di convoluzione

            downsample_layer = stack.pop()    #recupera il tensore corrispondente dalla fase di down-sampling

            output = up_conv(output)  #applica il blocco di conv di up-sampling al tensore corrente
            output = f.interpolate(output, scale_factor=2)  #interpola il tensore per raddoppiare la sua risoluzione spaziale

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]  #se necessario, applica un PADDING RIFLESSIVO per gestire le dim dispari
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = f.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)  #concatena il tensore interpolato con quello
                                                                          #recuperato dallo stack lungo la dim dei canali
            output = conv(output)  #applica il secondo blocco di convoluzione al tensore concatenatp

        return output  #restituisce il tensore finale che ha la stessa risoluzione dell'input iniziale ma con un numero
                       #diverso di canali.


#NormNet: VARIANTE UNet CHE INCLUDE LA NORM AGGIUNTIVA SULL'INTERA IMG COMPLESSA PRIMA DI PASSARE NELLA RETE
class NormNet(nn.Module):
    """
    Normalized Net model: in Unet or ResNet

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.

    Note NormUnet is designed for complex input/output only.
    """

    def __init__(
            self,
            in_chans: int,  #canali input complesso
            out_chans: int, #canali output conplesso
            chans: int = 32,  # 32 canali della prima convoluzione
            num_pool_layers: int = 3,  # canali di dowm-sampling e up-sampling

    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the complex input.
            out_chans: Number of channels in the complex output.
        """
        super().__init__()
        #crea un'istanza della UNet configurata con i seguenti parametri
        self.NormNet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,  # 32
            num_pool_layers=num_pool_layers)

    @staticmethod  #CONVERTE un canale complesso in 2 canali reali concatenati lungo la dimensione dei canali
    def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
        assert torch.is_complex(x)  #verifica che il tensore sia complesso
        return torch.cat([x.real, x.imag], dim=1)  #concatena parte reale e parte immaginaria

    @staticmethod  #CONVERTE un tensore con canali reali concatenati in un tensore complesso
    def chan_dim_to_complex(x: torch.Tensor) -> torch.Tensor:
        assert not torch.is_complex(x)  #verifica che non sia complesso
        _, c, _, _ = x.shape  #usato per estrarre dimensioni specifiche da un tensore quando non si ha bisogno di tutte
                              #le dimensioni
        assert c % 2 == 0  #verifica che il numero di canali sia pari
        c = c // 2  #divide i canali in due parti e..
        return torch.complex(x[:, :c], x[:, c:])  #..seleziona i primi c canali di x (reale) e i secondi c canali di x(imm)
                                                  #per creare poi un tensore complesso con parte reale e immaginaria

    @staticmethod #NORMALIZZA un tenore
    def norm(x: torch.Tensor):
        # group norm
        b, c, h, w = x.shape   #estrazione delle dimensioni del tensore
        assert c % 2 == 0    #verifica che c sia pari
        x = x.view(b, 2, c // 2 * h * w)  #riformattazione del tenore: b è uguale, 2 indica che il tensore è diviso in 2
                                          #parti (reale e immaginaria), appiattisce i canalli, altezza e larghezza in
                                          #un'unica dimensione
        mean = x.mean(dim=2).view(b, 2, 1)  #calcola la media lungo dim 2 (la dimensione appiattita) e riformatta
        std = x.std(dim=2).view(b, 2, 1)    #calcola la deviazione standard lungo dim 2 (la dimensione appiattita) e
                                            #riformatta

        x = (x - mean) / (std + 1e-12)  #la normalizzazione avviene sottraendo la media e dividendo le la dev standard

        return x.view(b, c, h, w), mean, std  #il tensore normalizzato viene riportato alla forma originale

    @staticmethod  #DENORMALIZZAZIONE del tensore
    def unnorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        b, c, h, w = x.shape  #estrae le dimensioni dal tensore
        assert c % 2 == 0  #verifica che c sia pari
        x = x.contiguous().view(b, 2, c // 2 * h * w)  #con x.contiguous si assicura che x sia stato memorizzato in modo
                                                       #contiguo e con view riformata per facilitare la denormalizzazione
        x = x * std + mean  #denormalizzazione che avviene invertendo il processo di normalizzazione
        return x.view(b, c, h, w)  #riporta il tensore alla sua forma originale

    @staticmethod  #PADDING: per assicurarsi che le dimensioni spaziali di x siano multipli di 16
    def pad(
            x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:  #Accetta un tensore x e restituisce una tupla
                                                                      #contenente il tensore con padding aggiunto e alcune
                                                                      #informazioni sul padding applicato.
        _, _, h, w = x.shape  #estrae altezza e larghzza del tensore
        w_mult = ((w - 1) | 15) + 1  #
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = f.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    @staticmethod  #UNPADDING
    def unpad(
            x: torch.Tensor,
            h_pad: List[int],
            w_pad: List[int],
            h_mult: int,
            w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    def forward(
            self,
            x: torch.Tensor,
            # ref: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert len(x.shape) == 4  #verifica che il tensore abbia 4 dimensioni

        x, mean, std = self.norm(x)  #normalizzazione del tensore di input
        x, pad_sizes = self.pad(x)  #padding del tensore di input

        x = self.NormNet(x)  #passo l'input alla NormNet. (NormNet è a sua volta un'istanza della UNet)

        x = self.unpad(x, *pad_sizes)   #una volta finita l'elaborarione, tolgo il padding..
        x = self.unnorm(x, mean, std)   #.. e denormalizzo il vettore

        return x  #restituisce il tensore elaborato dalla rete neurale


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:   #l'input del blocco è un tensore 4D
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)  #restituisce l'img elaborata con out_chans


class GatedConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.gatedlayers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            torch.sigmoid()
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        x_img = self.layers(image)
        x_gate = self.gatedlayers(image)
        x = x_img * x_gate
        return x


#BLOCCO CONVOLUZIONALE CHE INCLUDE: NORM, ATTIV, ATTENZIONE SPAZIALI E CONNESSIONI RES
class ConvBlockSM(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans=2, conv_num=0, out_chans=None, max_chans=None):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.chans_max = max_chans or in_chans
        self.out_chans = out_chans or in_chans

        self.layers = []
        self.layers.append(nn.Sequential(nn.Conv2d(self.in_chans, self.chans_max, kernel_size=3, padding=1, bias=False),
                                         nn.InstanceNorm2d(self.chans_max),
                                         nn.LeakyReLU(negative_slope=0.2, inplace=True)))

        # ### Spatial Attention ####
        self.SA = utils.SpatialAttention()

        for index in range(conv_num):
            if index == conv_num - 1:
                self.layers.append(
                    nn.Sequential(nn.Conv2d(self.chans_max, self.out_chans, kernel_size=3, padding=1, bias=False),
                                  nn.InstanceNorm2d(self.in_chans),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True)))
            else:
                self.layers.append(
                    nn.Sequential(nn.Conv2d(self.chans_max, self.chans_max, kernel_size=3, padding=1, bias=False),
                                  nn.InstanceNorm2d(self.chans_max),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True)))

        self.body = nn.Sequential(*self.layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.body(image)
        output = output + image[:, :2, :, :]
        # ### Spatial Attention ###
        output = self.SA(output) * output

        return output

#BLOCCO CHE USA LA CONV TRASPOSTA PER AUMENTARE LA RISOLUZIONE
class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)
