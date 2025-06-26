#!/usr/bin/env python3

import os
import numpy as np
import h5py
import torch
import torch.utils.data
import imageio
import utils

def center_crop(data, shape):
    if shape[0] <= data.shape[-2]:
        w_from = (data.shape[-2] - shape[0]) // 2
        w_to = w_from + shape[0]
        data = data[..., w_from:w_to, :]
    else:
        w_before = (shape[0] - data.shape[-2]) // 2
        w_after = shape[0] - data.shape[-2] - w_before
        pad = [(0, 0)] * data.ndim
        pad[-2] = (w_before, w_after)
        data = np.pad(data, pad_width=pad, mode='constant', constant_values=0)
    if shape[1] <= data.shape[-1]:
        h_from = (data.shape[-1] - shape[1]) // 2
        h_to = h_from + shape[1]
        data = data[..., :, h_from:h_to]
    else:
        h_before = (shape[1] - data.shape[-1]) // 2
        h_after = shape[1] - data.shape[-1] - h_before
        pad = [(0, 0)] * data.ndim
        pad[-1] = (h_before, h_after)
        data = np.pad(data, pad_width=pad, mode='constant', constant_values=0)
    return data

class VolumeDataset(torch.utils.data.Dataset):
    def __init__(self, volume, crop=None, q=0, flatten_channels=False):
        super().__init__()
        assert q < 0.5
        self.volume = volume
        self.flatten_channels = flatten_channels
        self.crop = crop
        with h5py.File(volume, 'r') as h5:
            data = h5.get('image') or h5.get('kspace')
            assert data is not None, "Neither 'image' nor 'kspace' found in HDF5 file"
            length = data.shape[0] if len(data.shape) == 3 else data.shape[0:2][0]
            self.channels = 1 if len(data.shape) == 3 else data.shape[1]
            self.protocal = h5.attrs.get('acquisition', 'Unknown')
        self.start = round(length * q)
        self.stop = length - self.start

    def __len__(self):
        return (self.stop - self.start) * self.channels if self.flatten_channels else self.stop - self.start

    def __getitem__(self, index):
        with h5py.File(self.volume, 'r') as h5:
            data = h5.get('image') or h5.get('kspace')

            if 'kspace' in h5:
                data = utils.ifftshift2(utils.ifft2(utils.fftshift2(
                    torch.from_numpy(np.array(data, dtype=np.complex64))
                ))).numpy()

            index += self.start
            if index >= data.shape[0]:
                index = 0

            # Prendi solo la slice necessaria
            if not self.flatten_channels:
                i = data[index]
            else:
                i = data[index // self.channels][index % self.channels][None, ...]

            # Assicura sempre 3 dimensioni
            i = i if len(i.shape) == 3 else i[None, ...]

            # Riduci al minimo la memoria allocata
            i = np.abs(i).astype(np.float32, copy=False)  # Evita copie inutili
            max_val = h5.attrs.get('max', np.max(i))

            if max_val > 0:
                i /= max_val  # Normalizzazione in-place

        if self.crop is not None:
            i = center_crop(i, (self.crop, self.crop))

        return i


class DummyVolumeDataset(torch.utils.data.Dataset):
    def __init__(self, ref):
        super().__init__()
        sample = ref[0]
        self.shape = sample.shape
        self.dtype = sample.dtype
        self.len = len(ref)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return np.zeros(self.shape, dtype=self.dtype)


class AlignedVolumesDataset(torch.utils.data.Dataset):
    def __init__(self, *volumes, protocals, crop=None, q=0, flatten_channels=False, exchange_modal=False):
        super().__init__()
        volumes: list = [VolumeDataset(x, crop, q=q, flatten_channels=flatten_channels) for x in volumes]
        print(volumes)
        for x in volumes:
            print("Volume shape:", x[0].shape)
        assert len({len(x) for x in volumes}) == 1
        assert len({x[0].shape for x in volumes}) == 1
        self.crop = crop
        volumes: dict = {volume.protocal: volume for volume in volumes}
        volumes['None'] = DummyVolumeDataset(next(iter(volumes.values())))
        volume_keys = volumes.keys()
        print('VOLUME KEYS: ', volume_keys)
        print('Protocals', protocals)
        if exchange_modal is True:
            p = torch.rand(1)
            if p > 0.5:
                protocals.reverse()
        # print(volumes)
        # for x in protocals:
        #     assert x in volumes.keys(), x + ' not found in ' + str(volumes.keys())
        volumes = [volumes[protocal] for protocal in protocals if protocal in volume_keys]
        print('VOLUMES: ', volumes)
        self.volumes = volumes

    def __len__(self):
        return len(self.volumes[0])

    def __getitem__(self, index):
        images = [volume[index] for volume in self.volumes]
        return images

def get_paired_volume_datasets(csv_path, protocals=None, crop=None, q=0, flatten_channels=False, basepath=None,
                               exchange_modal=False):
    datasets = []

    print(f"🔍 Leggendo il file CSV: {csv_path}")

    with open(csv_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        print("⚠️ ERRORE: Il file CSV è vuoto!")
        return []

    for line in lines:
        basepath = 'C:/Users/rebic/PycharmProjects/test_torch/jCAN-main'
        dataset_paths = [os.path.join(basepath, filepath.strip()) for filepath in line.strip().split(',')]

        # Stampa i percorsi letti dal CSV
        print(f"📂 Percorsi file dal CSV: {dataset_paths}")

        # Controllo se i file esistono
        valid_paths = [path for path in dataset_paths if os.path.exists(path)]
        missing_paths = [path for path in dataset_paths if not os.path.exists(path)]

        if missing_paths:
            print(f"❌ ATTENZIONE: I seguenti file non esistono:\n{missing_paths}")

        if not valid_paths:
            print("⚠️ ERRORE: Nessun file valido trovato, salto questo dataset.")
            continue

        try:
            dataset = AlignedVolumesDataset(*valid_paths, protocals=protocals, crop=crop, q=q,
                                            flatten_channels=flatten_channels, exchange_modal=exchange_modal)
            datasets.append(dataset)
        except Exception as e:
            print(f"🚨 ERRORE durante la creazione del dataset: {e}")

    if not datasets:
        print("⚠️ ERRORE: Nessun dataset è stato creato!")

    return datasets


class TiffPaired(torch.utils.data.Dataset):
    def __init__(self, tiffs, crop=None):
        super().__init__()
        self.tiffs = tiffs
        self.crop = crop

    def __len__(self):
        return len(self.tiffs)

    def __getitem__(self, ind):
        img = imageio.v2.imread(self.tiffs[ind])
        assert len(img.shape) == 2
        t1, t2 = np.split(img, 2, axis=-1)
        t1, t2 = map(lambda x: np.stack([x, np.zeros_like(x)], axis=0), (t1, t2))
        if self.crop is not None:
            t1, t2 = map(lambda x: center_crop(x, [self.crop] * 2), (t1, t2))
        return t1, t2


def test_get_paired_volume_datasets():
    # test get_paired_volume_datasets
    datasets = get_paired_volume_datasets('D:/tesi rebecca/jCAN-main',
                                          protocals=['CORPD_FBK'], flatten_channels=False)
    print(sum([len(dataset) for dataset in datasets]))


if __name__ == '__main__':
    test_get_paired_volume_datasets()

