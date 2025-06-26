import torch
import torch.nn as nn
import utils
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        channel_scale: number of branches as input.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    #costruttore init che inizializza i parametri necessari
    def __init__(self, dim, window_size, num_heads, channel_scale=1, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim   #dim canali di input
        self.window_size = window_size  # Wh, Ww: tupla di due valori, altezza e larghezza
        self.num_heads = num_heads
        head_dim = self.dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.channel_scale = channel_scale
        #tabella dei bias di posizione relativi
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        #relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        #strati lineari per Q,K,V e proiezione
        self.q = nn.Linear(self.dim, self.dim // self.channel_scale, bias=qkv_bias)
        self.kv = nn.Linear(self.dim, self.dim * 2 // self.channel_scale, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim // self.channel_scale, dim // self.channel_scale)

        self.proj_drop = nn.Dropout(proj_drop)
        #inizializzazione della tabella dei bias di posizione relativi e softmax
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    #passaggio in avanti: processo che avviene quando il modulo riceve un input durante il training o l'inferenza
    def forward(self, x, y, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # 1️⃣ **Limitare N per evitare eccessivo consumo di memoria**
        max_seq_len = 512  # Puoi modificarlo in base alla memoria disponibile
        if N > max_seq_len:
            x = x[:, :max_seq_len, :]
            y = y[:, :max_seq_len, :]
            N = max_seq_len  # Aggiornare la dimensione di N

        # 2️⃣ **Sottocampionamento per ridurre seq_len**
        stride = 2  # Se vuoi meno riduzione, usa stride=1 o 1.5
        x = x[:, ::stride, :]
        y = y[:, ::stride, :]
        N = x.shape[1]  # Aggiorniamo N dopo il sottocampionamento

        q = self.q(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads // self.channel_scale).permute(2, 0, 3, 1,
                                                                                                           4)
        kv = self.kv(y).reshape(B_, N, 2, self.num_heads, C // self.num_heads // self.channel_scale).permute(2, 0, 3, 1,
                                                                                                             4)
        q, k, v = q[0], kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        # 4️⃣ **Calcolo dell'attenzione ottimizzata**
        attn = (q @ k.transpose(-2, -1))  # Matrice di attenzione

        # bias di posizione relativo
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        # **Verifica delle dimensioni di attn e relative_position_bias**
        print(f"Dimensione di attn: {attn.shape}")
        print(f"Dimensione di relative_position_bias: {relative_position_bias.shape}")

        # **Adattamento delle dimensioni**
        # Se attn è più piccolo (128x128), riduci relative_position_bias
        if attn.shape[-2:] != relative_position_bias.shape[-2:]:
            print("Ridimensionamento di relative_position_bias")
            # Riduci relative_position_bias alle dimensioni di attn
            relative_position_bias = relative_position_bias[:, :attn.shape[-2], :attn.shape[-1]]

            # Verifica che relative_position_bias si adatti a attn
            if relative_position_bias.shape[-2] != attn.shape[-2]:
                relative_position_bias = relative_position_bias.repeat(1, 1,
                                                                       attn.shape[-2] // relative_position_bias.shape[
                                                                           -2])

            # Se necessario, ridimensiona nuovamente relative_position_bias
            if relative_position_bias.shape[-2] != attn.shape[-2]:
                relative_position_bias = relative_position_bias[:, :attn.shape[-2], :attn.shape[-2]]

            print(f"Dimensione dopo ridimensionamento: {relative_position_bias.shape}")

            # Somma corretta
            attn = attn + relative_position_bias.unsqueeze(0)
        else:
            attn = attn + relative_position_bias.unsqueeze(0)

        # applicazione di Mask se presente
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        # dropout e combinazione pesata
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C // self.channel_scale)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    #Questo metodo restituisce una rappresentazione testuale delle impostazioni principali del modulo
    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class SwinTransformerElement(nn.Module):
    r""" Swin Transformer Element.

    Args:
        embed_dim: Length of embeded vector in ViT.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, embed_dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = embed_dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size  # quantità di spostamento applicata alle finestre per implementare lo SW-MSA
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        if min(self.input_resolution) <= self.window_size:  # se la dim della finestra è > della risoluzione dell'img...
            self.shift_size = 0  # ...disabilito lo shift e..
            self.window_size = min(self.input_resolution)  # ..riduco la finestra alla risoluzione dell'immagine
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = self.norm_layer(self.dim)  # normalizzazione dei dati di input prima dell'attenzione
        self.attn = WindowAttention(
            self.dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()  # implementa tecnica di regularizzazione
        self.norm2 = self.norm_layer(self.dim)  # normalizzazione dopo l'attenzione
        mlp_hidden_dim = int(self.dim * mlp_ratio)
        self.mlp = utils.Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:  # se lo shift è abilitato..
            attn_mask = self.calculate_mask(self.input_resolution)  # viene calcolata una maschera di attenzione per
            # gestire i bordi tra le finestre
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    # MASCHERA DI ATTENZIONE
    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (
            slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (
            slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = utils.window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, y):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x

        assert x.shape == y.shape
        # l'inpput x viene normalizzato e ridimensionato per adattarsi alla risoluzione dell'img. vale anche per y
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        y = self.norm1(y)
        y = y.view(B, H, W, C)
        # cyclic shift
        if self.shift_size > 0:  # se lo shit è abilitato...
            # ...l'img viene traslata per poter applicare SW-MSA
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:  # altrimenti...
            shifted_x = x  # ...rimane invariata
            shifted_y = y

        # l'immagine shiftata viene divisa in finestre e l'attenzione viene calcolata proprio in queste
        x_windows = utils.window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        y_windows = utils.window_partition(shifted_y, self.window_size)  # nW*B, window_size, window_size, C
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn(x_windows, y_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        print(f"attn_windows.shape before view: {attn_windows.shape}")
        print(f"self.window_size: {self.window_size}, C: {C}")
        # le finestre vengono ricomposte per formare l'img
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = utils.window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # se l'immagine è stata shiftata, viene riportata alla forma originale
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # Infine, si aggiunge il risultato dell'attenzione alla connessione residua e si passa attraverso l'MLP.
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class FusionTransformer(nn.Module):
    r""" Fusion Transformer.

    Args:
        embed_dim: Length of embeded vector in ViT.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        channel_scale: number of branches to be fused.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if  set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, embed_dim, input_resolution, num_heads, window_size=7, shift_size=0, channel_scale=3,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        #INIZIALIZZAZIONE PARAMETRI
        self.dim = embed_dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.channel_scale = channel_scale
        if min(self.input_resolution) <= self.window_size:
            #se la dim della finestra è maggiore della risoluzione...
            self.shift_size = 0   #...viene disabilitato lo shift e...
            self.window_size = min(self.input_resolution)   #.. riduco la finestra alla risoluzione dell'immagine
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(self.dim)  #creo uno strato normalizzato
        self.attn = WindowAttention(
            self.dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, channel_scale=channel_scale,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(self.dim // channel_scale)  #Definisce un altro strato di normalizzazione, ma con una dim
                                                            # ridotta per via della divisione per channel_scale
        mlp_hidden_dim = int(self.dim // channel_scale * mlp_ratio)
        self.mlp = utils.Mlp(in_features=self.dim // channel_scale, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                             drop=drop)
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    #MASCHERA PER APPLICARE IL MECCASINO SW-MSA.
    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (
        slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (
        slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = utils.window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = utils.window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        attn_windows = self.attn(x_windows, x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C // self.channel_scale)
        shifted_x = utils.window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C // self.channel_scale)

        # FFN
        x = self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class BasicLayer(nn.Module):
    """ A basic Swin Transformer Block for one stage.

    Args:
        embed_dim: Length of embeded vector in ViT.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, embed_dim, input_resolution, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = embed_dim
        self.input_resolution = input_resolution

        #COSTRUZIONE DEI BLOCCHI DI SWIN
        self.blocksSwin = nn.Sequential(
            SwinTransformerElement(embed_dim=embed_dim, input_resolution=input_resolution,
                                   num_heads=num_heads, window_size=window_size,
                                   shift_size=0,
                                   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop, attn_drop=attn_drop,
                                   drop_path=drop_path,
                                   norm_layer=norm_layer),
            SwinTransformerElement(embed_dim=embed_dim, input_resolution=input_resolution,
                                   num_heads=num_heads, window_size=window_size,
                                   shift_size=window_size // 2,
                                   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop, attn_drop=attn_drop,
                                   drop_path=drop_path,
                                   norm_layer=norm_layer))

        #COSTRUZIONE MLP
        self.mlp = utils.Mlp(in_features=embed_dim * 2, hidden_features=embed_dim * mlp_ratio, out_features=embed_dim)

    def forward(self, x, y):
        x_ = self.blocksSwin[0](x, y)  #Il 1° blocco Swin Transformer viene applicato all'input x e y, producendo x_
        x = self.blocksSwin[1](x_, y)  #Il 2° blocco Swin Transformer viene applicato all'output del 1° blocco,
                                       #producendo x
        return x


class SelfCrossBlocks(nn.Module):
    """ A self-cross attention Swin Transformer layer.

    Args:
        embed_dim: Length of embeded vector in ViT.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, embed_dim, input_resolution, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, n_SC=1):

        super().__init__()
        self.input_resolution = input_resolution
        self.n_SC = n_SC

        #LISTA DI MODULI blockSC CHE CONTIENE UNA SERIE DI ISTANZE DI BASIC LAYER
        self.blockSC = nn.ModuleList([
            BasicLayer(embed_dim=embed_dim, input_resolution=input_resolution, num_heads=num_heads,
                       window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                       drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm)
            for _ in range(self.n_SC * 2)])  #n° blocchi è pari a quello tra () perchè per ogni livello di self cross -
                                             #attention vengono usati 2 blocchi: 1 per self-attention e uno per cross

    def forward(self, x, y):

        for blk_index in range(len(self.blockSC)):  #itero attraverso tutti i blocchi self-cross
            if blk_index % 2 == 0:   #se l'indice del blocco è pari...
                #... esegue self-attention su x e y separatamente, aggiornando x_ e y_
                x_ = self.blockSC[blk_index](x, x)
                y_ = self.blockSC[blk_index](y, y)
            else:   #atrimenti...
                #esegue cross-attention incrociata
                x = self.blockSC[blk_index](x_, y_)
                y = self.blockSC[blk_index](y_, x_)
        return x, y

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class ViT(nn.Module):
    """Self Cross Transformer Block (SCTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        img_size: Input image size.
        patch_size: Patch size.
        n_SC: Number of self- and cross-attention pair.
        embed_dim: Length of embeded vector in ViT.
        ds_ref: If True, use gradient map of the reference image.

    """

    def __init__(self, dim, img_size, num_heads, window_size, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., patch_norm=True,
                 drop_path=0., norm_layer=nn.LayerNorm, patch_size=4, n_SC=1, embed_dim=32, ds_ref=True):
        super(ViT, self).__init__()

        self.dim = dim
        self.norm_layer = norm_layer
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.img_size = img_size

        #PATCH EMBEDDING
        self.patch_embed_t = utils.PatchEmbed(
            img_size=self.img_size, patch_size=self.patch_size, in_chans=self.dim, embed_dim=embed_dim,
            norm_layer=self.norm_layer if self.patch_norm else None)

        self.patch_embed_r = utils.PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim + 1 if ds_ref else dim, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        #PATCH UNEMBEDDING
        self.patch_unembed = utils.PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, out_chans=self.dim, embed_dim=embed_dim,
            norm_layer=self.norm_layer if self.patch_norm else None)

        self.input_resolution = self.patch_embed_t.patches_resolution

        self.SCBs = SelfCrossBlocks(embed_dim=embed_dim, input_resolution=self.input_resolution,
                                    num_heads=num_heads, window_size=window_size,
                                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                    norm_layer=norm_layer, n_SC=n_SC)

        self.FusionT = FusionTransformer(embed_dim=3 * embed_dim, input_resolution=self.input_resolution,
                                         num_heads=num_heads, window_size=window_size, channel_scale=3,
                                         shift_size=0, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop, norm_layer=norm_layer)

    def forward(self, x, y):
        #x e y vengono padati per assicurare che la loro dim sia compatibile con i calcolo del patch
        x, pad_size = utils.pad(x, self.window_size * self.patch_size)
        y, pad_size = utils.pad(y, self.window_size * self.patch_size)
        #vengono applicati i moduli di embedding per trasformare i patch di x e y in vettori
        x = self.patch_embed_t(x)
        y = self.patch_embed_r(y)
        #i vettori vengono passati a SelfCrossBlocks producendo versioni aggiornate di x e y
        x, y_ = self.SCBs(x, y)
        #i vettori x,y_ e y vengono concatenati lungo la dim dei canali per creare un'unica rappresentazione
        concat = torch.cat([x, y_, y], dim=2)
        #concat passa attraverso FusionTransformer che fonde le informazioni
        x = self.FusionT(concat)
        #il risultato è trasformato in patch e riconvertito nell'immagine originale
        x = self.patch_unembed(x)
        #eseguo l'unpading per riportarlo alla dim originale
        x = utils.unpad(x, pad_size[0], pad_size[1])
        return x


#def setup_and_launch():
    #dim = 2
    #embed_dim = 16
    #window_size = 8
    #height = width = imgsize = 128
    #patch_size = 4
    #model = ViT(dim=dim, img_size=(height, width), patch_size=patch_size,
     #           window_size=window_size, num_heads=8, embed_dim=embed_dim, mlp_ratio=16)
    #x = torch.randn((1, dim, height, width))
    #x = model(x, x)


if __name__ == '__main__':
    dim = 2
    embed_dim = 16
    window_size = 8
    height = width = imgsize = 128
    patch_size = 4
    model = ViT(dim=dim, img_size=(height, width), patch_size=patch_size,
                window_size=window_size, num_heads=8, embed_dim=embed_dim, mlp_ratio=16)
    x = torch.randn((1, dim, height, width))
    x = model(x, x)
