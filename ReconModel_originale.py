class ReconModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #print('Numero di coil: ', self.cfg.coils)

        self.Eval = None
        self.metric_SSIM_raw = None
        self.metric_PSNR_raw = None
        self.metric_MSE = None
        self.metric_MAE = None
        self.metric_SSIM = None
        self.metric_PSNR = None
        self.loss_ssim = None
        self.loss_TV = None
        self.local_fidelities = None
        self.loss_fidelity = None
        self.loss_all = None
        self.Ref_Kspace_sampled = None
        self.loss_consistency = None
        self.mask_ref = None
        self.Ref_img_f = None
        self.mask_tar = None
        self.Target_sampled_rss = None
        self.Ref_Kspace_f = None
        self.Ref_f_rss = None
        self.Target_Kspace_sampled = None
        self.Target_Kspace_f = None
        self.Target_f_rss = None
        self.rhos = generate_rhos(self.cfg.num_recurrent)  #calcola la rhos usando la funzione generate_rhos
        self.device = self.cfg.device  #imposta il dispositivo per il modello
        self.num_low_frequencies = int(
            self.cfg.img_size[1] * self.cfg.sparsity_tar * 0.32)  #n° di frequenze basse basato sulla configurazione

        #inizializza maschere di riferimento e maschere target usando classi specifiche nel dizionario Masks
        #vengono trasferite al dispositivo indicato da 'device'
        self.net_mask_ref = masks[self.cfg.mask](self.cfg.sparsity_ref, self.cfg.img_size[1]).to(
            self.device)  # the number of columns
        self.net_mask_tar = masks[self.cfg.mask](self.cfg.sparsity_tar, self.cfg.img_size[1]).to(
            self.device)  # the number of columns


        #inizializza la rete neurale usando una configurazione complessa
        self.net_R = fD2RT.fD2RT(coils=self.cfg.coils * 2, img_size=self.cfg.img_size,
                                 num_heads=self.cfg.num_heads, window_size=self.cfg.window_size,
                                 patch_size=self.cfg.patch_size,
                                 mlp_ratio=self.cfg.mlp_ratio, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                                 drop_path=0., norm_layer=nn.LayerNorm, n_SC=self.cfg.n_SC,
                                 num_recurrent=self.cfg.num_recurrent,
                                 embed_dim=self.cfg.embed_dim, sens_chans=self.cfg.sens_chans,
                                 sens_steps=self.cfg.sens_steps,
                                 ds_ref=self.cfg.ds_ref).to(self.device)  # 2*coils for target and reference

    def set_input_no_gt(self, target_img_sampled, ref_img_f=None):
        #inizializza le variabili per memorizzare le img target e il loro spazio K
        b, c, h, w = self.Target_img_sampled.shape
        self.Target_f_rss = torch.zeros([b, 1, h, w], dtype=torch.complex64)
        self.Target_Kspace_f = torch.zeros([b, c, h, w], dtype=torch.complex64)

        #calcola la trasformata di Fourier dell'immagine target campionata
        self.Target_Kspace_sampled = fft2(target_img_sampled)
        if ref_img_f is None:  #se l'img di riferimento non è disponibile...
            self.Ref_Kspace_f = None  #...inizializzo le variabili con valori predefiniti
            self.Ref_f_rss = torch.ones_like(self.Target_f_rss)
        else:   #se invece img è disponibile...
            self.Ref_f_rss = rss(ref_img_f)
            self.Ref_Kspace_f = fft2(ref_img_f)   #...ne calcolo la trasformata
            #print(f"Ref_Kspace_f shape in no_gt: {self.Ref_Kspace_f.shape}")

        self.Target_sampled_rss = rss(ifft2(self.Target_Kspace_sampled))  #Calcola la radice quadrata della somma dei
                                                         # quadrati delle immagini campionate (RSS) dell'immagine target

        with torch.no_grad():  #Aggiorna le maschere di rif e targ senza effettuare il tracciamento dei gradienti
            self.mask_ref = torch.logical_not(self.net_mask_ref.pruned)
            self.mask_tar = torch.logical_not(self.net_mask_tar.pruned)

    def set_input_gt(self, target_img_f, ref_img_f=None):
        #Calcola la RSS e la trasformata di Fourier dell'immagine target fornita
        self.Target_f_rss = rss(target_img_f)
        self.Target_Kspace_f = fft2(target_img_f)
        #print(f"Target_Kspace_f shape: {self.Target_Kspace_f.shape}")

        if ref_img_f is None:
            self.Ref_img_f = None
            self.Ref_f_rss = torch.ones_like(self.Target_f_rss)
        else:
            self.Ref_f_rss = rss(ref_img_f)
            self.Ref_Kspace_f = fft2(ref_img_f)
            #print(f"Ref_Kspace_f shape: {self.Ref_Kspace_f.shape}")

        with torch.no_grad():  # avoid update of mask
            self.mask_ref = torch.logical_not(self.net_mask_ref.pruned)
            self.mask_tar = torch.logical_not(self.net_mask_tar.pruned)
            self.Target_Kspace_sampled = self.Target_Kspace_f * self.mask_tar
            if self.Ref_Kspace_f is not None:  #calcola le versioni campionate delle trasformate di Fourier.
                self.Ref_Kspace_sampled = self.Ref_Kspace_f * self.mask_ref
        #Calcola la RSS dell'immagine target campionata.
        self.Target_sampled_rss = rss(ifft2(self.Target_Kspace_sampled))

    def forward(self, target_img_f, ref_img_f=None):
        #print(f"Numero di canali in ingresso: {target_img_f.shape[1]}")
        #se non lo sono già, converto le immagini in formato complesso
        if not torch.is_complex(target_img_f):
            target_img_f = utils.chan_dim_to_complex(target_img_f)
            #print(f"Numero di canali dell'input target dopo la conversione complessa: {target_img_f.shape[1]}")
            if ref_img_f is not None:
                #print(f"Il n° iniziale di canali dell'input di riferimento: {ref_img_f.shape[1]}")
                ref_img_f = utils.chan_dim_to_complex(ref_img_f)
                #print(f"Numero di canali dell'input di riferimento dopo la conversione complessa: {ref_img_f.shape[1]}")

        #Imposta i dati di input utilizzando il metodo appropriato basato sulla configurazione.
        #Utilizza l'accuratezza automatica per il calcolo con precisione mista.
        with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
            if self.cfg.GT is True:
                self.set_input_gt(target_img_f, ref_img_f)
            else:
                self.set_input_no_gt(target_img_f, ref_img_f)

            #if self.Ref_Kspace_sampled is not None:
                #print(f"Numero di canali di `Ref_Kspace_f`: {self.Ref_Kspace_sampled.shape[1]}")
            #else:
                #print("self.Ref_Kspace_sampled è None")

            #Esegue la rete neurale net_R per ottenere le ricostruzioni dell'immagine e delle mappe di sensibilità.
            (self.recs_complex, self.rec_rss, self.sens_maps, self.rec_img) = self.net_R(
                Ref_Kspace_f=self.Ref_Kspace_sampled,
                Target_Kspace_u=self.Target_Kspace_sampled,
                mask=self.mask_tar,
                num_low_frequencies=self.num_low_frequencies
            )

            #Calcola la perdita di fedeltà tra le immagini ricostruite e target.
            self.loss_all = 0
            self.loss_fidelity = 0  #accumula la perdita
            self.local_fidelities = []
            for i in range(self.cfg.num_recurrent):
                loss_fidelity = F.l1_loss(rss(self.recs_complex[i]), self.Target_f_rss) + self.cfg.lambda0 * F.l1_loss(
                    utils.sens_expand(self.recs_complex[i], self.sens_maps), self.Target_Kspace_f)
                self.local_fidelities.append(self.rhos[i] * loss_fidelity)
                self.loss_fidelity += self.local_fidelities[-1]

            # Calcola altre perdite, inclusa la perdita di consistenza, la perdita TV e la perdita SSIM.
            self.loss_all += self.loss_fidelity

            self.loss_consistency = self.cfg.lambda1 * F.l1_loss(
                self.mask_tar * utils.sens_expand(self.rec_img, self.sens_maps), self.Target_Kspace_sampled)

            self.loss_all += self.loss_consistency

            self.loss_TV = tv_loss(torch.abs(self.sens_maps), self.cfg.lambda3)
            self.loss_all += self.loss_TV

            self.loss_ssim = self.cfg.lambda2 * ssimloss(self.rec_rss, self.Target_f_rss)
            self.loss_all += self.loss_ssim
            # Restituisce le perdite calcolate e altre metriche per l'addestramento e la valutazione.
            return self.local_fidelities, self.loss_fidelity, self.loss_consistency, self.loss_ssim, self.loss_all

    def test(self, target_img_f, ref_img_f=None):
        assert self.training is False
        with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
            with torch.no_grad():
                self.forward(target_img_f, ref_img_f)
                self.metric_PSNR = metrics.psnr(self.Target_f_rss, self.rec_rss, self.cfg.GT)
                self.metric_SSIM = metrics.ssim(self.Target_f_rss, self.rec_rss, self.cfg.GT)
                self.metric_MAE = metrics.mae(self.Target_f_rss, self.rec_rss, self.cfg.GT)
                self.metric_MSE = metrics.mse(self.Target_f_rss, self.rec_rss, self.cfg.GT)

                self.metric_PSNR_raw = metrics.psnr(self.Target_f_rss, self.Target_sampled_rss, self.cfg.GT)
                self.metric_SSIM_raw = metrics.ssim(self.Target_f_rss, self.Target_sampled_rss, self.cfg.GT)
                self.Eval = tuple([self.metric_PSNR, self.metric_SSIM])

    @staticmethod
    def prune(*args, **kwargs):
        assert False, 'Take care of amp'  #suggerisce che il metodo prune non è stato implementato o è stato lasciato
                                          #come un segnaposto.
        return self.net_mask_tar.prune(*args, **kwargs)

    def get_vis(self, content=None):
        assert content in [None, 'scalars', 'histograms', 'images']  #controlla che content sia uno dei valori tra []
        vis = {}  #lista vuota che conterrà tutti i dati di visualizzazione
        if content == 'scalars' or content is None:  #se content è scalars o None...
            vis['scalars'] = {}  #...inizializza una chiave scalars nel dizionario
            #Filtra tutti gli attributi della classe che iniziano con 'loss_' e aggiungi i valori corrispondenti al
            #dizionario sotto la chiave 'scalars'
            for loss_name in filter(lambda x: x.startswith('loss_'), self.__dict__.keys()):
                loss_val = getattr(self, loss_name)
                if loss_val is not None:
                    vis['scalars'][loss_name] = loss_val.detach().item()  #detach() è usato per fare in modo che i
                                                                             #pesi non partecipino al calcolo del grad.
            #filtra tutti gli attributi che iniziano con 'metric_' e aggiungi i valori corrispondenti al dizionario
            #sotto la chiave 'scalars'.
            for metric_name in filter(lambda x: x.startswith('metric_'), self.__dict__.keys()):
                metric_val = getattr(self, metric_name)
                if metric_val is not None:
                    vis['scalars'][metric_name] = metric_val

        if content == 'images' or content is None:  #se content è images o None...
            vis['images'] = {}   #...inizializza una chiave images nel dizionario
            #Filtra tutti gli attributi che terminano con '_rss' e aggiungi i valori corrispondenti al dizionario
            #sotto la chiave 'images'
            for image_name in filter(lambda x: x.endswith('_rss'), self.__dict__.keys()):
                image_val = getattr(self, image_name)
                if (image_val is not None) \
                        and (image_val.shape[1] == 1 or image_val.shape[1] == 3) \
                        and not torch.is_complex(image_val): #verifica che img non sia complessa e abbia 1 o 3 canali
                    vis['images'][image_name] = image_val.detach()

        if content == 'histograms' or content is None:  #se content è histograms o None...
            vis['histograms'] = {}   #...inizializza una chiave histograms nel dizionario
            if self.net_mask_tar.weight is not None:
                #. Aggiunge gli istogrammi delle pesature della maschera di rete se disponibili.
                vis['histograms']['weights'] = {'values': self.net_mask_tar.weight.detach()}
        return vis
