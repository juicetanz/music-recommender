import pandas as pd
import os

import torch
import torchaudio
from torch.utils.data import TensorDataset

df = pd.read_csv('songs.csv', usecols=['song','energy'])
train_folder = os.listdir('candidates')

def GetData(df):
    min = 1885824

    images = torch.zeros(len(train_folder), 2, min)
    energy_levels = torch.zeros(len(train_folder), 1)

    for idx, song in enumerate(train_folder):
        print('Images loaded: ' + str(idx + 1) + '/' + str(len(train_folder)) + ' Current image: ' + song[:-4])

        wf, sr = torchaudio.load('candidates/' + song)
        wf = wf[:,:min]
        print(wf.shape)
        images[idx] = wf

        energy_levels[idx] = torch.tensor(df.loc[df['song'] == song[:-4], 'energy'].item())

    images = torch.mean(images, dim=1)
    images = torch.unsqueeze(images, 1)
    print(images.shape)
    return TensorDataset(images, energy_levels)

image_energy = GetData(df)
#torch.save(image_energy, 'data/images.json')
