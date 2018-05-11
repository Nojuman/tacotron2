import os
import sys
import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.utils.data as Utils
from tqdm import tqdm

import hyperparams as hp
from audio import load_wav, wav_to_spectrogram

import csv
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from IPython.display import clear_output
import random
import sys
import time
import json
import seaborn as sns

if not os.path.exists("tacaudio/"):
    os.makedirs("tacaudio/")
    print("Created a 'tacaudio' folder to store voice data")

class LJSpeechDataset(Dataset):

    def __init__(self, path, text_transforms=None, audio_transforms=None, cache=False, sort=True):
        self.path = path
        self.metadata = pd.read_csv(f'{path}/metadata.csv', sep='|',
                                    names=['wav', 'transcription', 'text'],
                                    usecols=['wav', 'text'])
        self.metadata.dropna(inplace=True)
        self.audio_transforms = audio_transforms
        self.text_transforms = text_transforms
        self.cache = cache
        if sort:
            self.metadata['length'] = self.metadata['wav'].apply(
                    lambda x: librosa.get_duration(filename=f'{path}/wavs/{x}.wav'))
            self.metadata.sort_values(by=['length'], inplace=True)
        if cache:
            self.cache_spectrograms()

    def cache_spectrograms(self):
        wav_filenames = self.metadata['wav']
        spectrograms_path = f'{self.path}/spectrograms'
        if not os.path.exists(spectrograms_path):
            os.makedirs(spectrograms_path)
            print('Building Cache..')
            for name in tqdm(wav_filenames, total=len(wav_filenames)):
                audio, _ = load_wav(f'{self.path}/wavs/{name}.wav')
                S = self.audio_transforms(audio)
                np.save(f'{spectrograms_path}/{name}.npy', S)

    def __getitem__(self, index):
        text = self.metadata.iloc[index]['text']
        filename = self.metadata.iloc[index]['wav']
        if self.text_transforms:
            text = self.text_transforms(text)
        if self.cache:
            audio = np.load(f'{self.path}/spectrograms/{filename}.npy')
            return text, audio

        audio, _ = load_wav(f'{self.path}/wavs/{filename}.wav')
        if self.audio_transforms:
            audio = self.audio_transforms(audio)
            
        print("TEXT", text)
        print("AUDIO", audio)
        return text, audio

    def __len__(self):
        return len(self.metadata)


class WaveNetDataset(Dataset):
    """
    loads spectrogram and raw audio pairs for testing wavnet

    real spectrogramnet outputs much be cached and used in a separate dataset
    """

    def __init__(self, path):
        self.path = path
        self.metadata = pd.read_csv(f'{path}/metadata.csv', sep='|',
                                    names=['wav', 'transcription', 'text'],
                                    usecols=['wav', 'text'])
        self.metadata.dropna(inplace=True)

    def __getitem__(self, index):
        wav_filename = self.metadata.iloc[index]['wav']
        audio, _ = load_wav(f'{self.path}/wavs/{wav_filename}.wav')
        S = wav_to_spectrogram(audio)
        return S, audio

    def __len__(self):
        return len(self.metadata)


def wav_collate(batch):
    spec = [item[0] for item in batch]
    audio = [item[1] for item in batch]

    spec_lengths = [len(x) for x in spec]
    audio_lengths = [len(x) for x in audio]

    max_spec = max(spec_lengths)
    max_audio = max(audio_lengths)

    spec_batch = np.stack(pad2d(x, max_spec) for x in spec)
    audio_batch = np.stack(pad1d(x, max_audio) for x in audio)

    return (torch.FloatTensor(spec_batch).permute(0, 2, 1),  # (batch, channel, time)
            torch.FloatTensor(audio_batch),
            spec_lengths, audio_lengths)


def collate_fn(batch):
    """
    Pads Variable length sequence to size of longest sequence.
    Args:
        batch:

    Returns: Padded sequences and original sizes.

    """
    text = [item[0] for item in batch]
    audio = [item[1] for item in batch]

    text_lengths = [len(x) for x in text]
    audio_lengths = [len(x) for x in audio]

    max_text = max(text_lengths)
    max_audio = max(audio_lengths)

    text_batch = np.stack(pad1d(x, max_text) for x in text)
    audio_batch = np.stack(pad2d(x, max_audio) for x in audio)

    return (torch.LongTensor(text_batch),
            torch.FloatTensor(audio_batch).permute(1, 0, 2),
            text_lengths, audio_lengths)


def pad1d(seq, max_len):
    return np.pad(seq, (0, max_len - len(seq)), mode='constant', constant_values=hp.padding_idx)


def pad2d(seq, max_len, dim=hp.num_mels, pad_value=hp.spectrogram_pad):
    padded = np.zeros((max_len, dim)) + pad_value
    padded[:len(seq), :] = seq
    return padded


class RandomBatchSampler:
    """Yields of mini-batch of indices, sequential within the batch, random between batches.
    Incomplete last batch will appear randomly with this setup.
    
    Helpful for minimizing padding while retaining randomness with variable length inputs.
    
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.


    Example:
        >>> list(RandomBatchSampler(range(10), 3))
        [[0, 1, 2], [6, 7, 8], [3, 4, 5], [9]]
    """

    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = batch_size
        self.random_batches = self.make_batches(sampler, batch_size)

    def make_batches(self, sampler, batch_size):
        indices = [i for i in sampler]
        batches = [indices[i:i+batch_size]
                   for i in range(0, len(indices), batch_size)]
        random_indices = torch.randperm(len(batches)).long()
        return [batches[i] for i in random_indices]

    def __iter__(self):
        for batch in self.random_batches:
            yield batch

    def __len__(self):
        return len(self.random_batches)


class make_data():
    """
    Make all the text and audio data with preprocessing
    
    """
    def __init__(self):
        self.eos = '~'
        self.pad = '_'
        self.chars = self.pad + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? ' + self.eos
        self.vocab = self.chars
        self.unk_idx = len(self.chars)

        self.char_to_id = {char: i for i, char in enumerate(self.chars)}
        self.id_to_char = {i: char for i, char in enumerate(self.chars)}


    def text_to_sequence(self, text):
        text += self.eos
        return [self.char_to_id.get(char, self.unk_idx) for char in text]


    def sequence_to_text(self, sequence):
        return "".join(self.id_to_char.get(i, '<unk>') for i in sequence)


    def make_text_data(self):
        text_path = '/home/ubuntu/VCTK-Corpus/txt/'
        speaker_ids = ['p225']

        for ids in speaker_ids:

            text_sorted = sorted((os.listdir(text_path+ids)))
            new_text_len = []
            for text_file in text_sorted:
                with open(text_path + ids + str('/') + text_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        new_text_len.append(len(list(line[:-1])))

                        with open("corpus_text_taco.csv", 'a') as myinput:
                            wr = csv.writer(myinput)
                            wr.writerow([self.text_to_sequence(list(line[:-1]))])

                            
    def make_audio_data(self):
        corpus_audio = []
        audio_path = '/home/ubuntu/VCTK-Corpus/wav48/'
        speaker_ids = ['p997']

        for i,ids in enumerate(speaker_ids):

            audio_sorted = sorted((os.listdir(audio_path+ids)))
            for idx, audio_file in enumerate(audio_sorted):
                read_file = librosa.core.load(audio_path + ids + str('/') + audio_file, sr=16000, mono=True)[0]

                # audio --> trim --> trim for MFCC (200) --> normalise (-1,1)  --> mu encode --> quantise --> S
                trimmed = trim_audio(read_file, 0.03, 2048)
                #Normalise Audio Data
                norm_trim = trimmed[:(len(trimmed)//200)*200]
                norm_trim = normalise_audio(norm_trim)

                causal_sample = mu_encoder_np(norm_trim)

                causal_sample = np.floor(causal_sample)
                
                s = np.abs(librosa.core.stft(y=causal_sample, n_fft=800, hop_length=200, window='hann', center=True))
                spect = librosa.feature.melspectrogram(S=s, n_mels=80, fmax=7600, fmin=125, power=2)
                log_compressed = librosa.core.amplitude_to_db(S=spect, ref=1.0, amin=5e-4, top_db=80.0)

                filename = "tacaudio/"+"{:03d}".format(i+1)+"mfcc_{0:03d}".format(idx)
                np.save(filename, log_compressed)


                with open("corpus_audio_taco.csv", 'a') as myinput:
                    wr = csv.writer(myinput)
                    wr.writerow([filename+ '.npy'])
                    
                    
def my_collate(batch):
    data = []
    data_len = []
    target = []
    target_len = []
    ids = []
    
    for item in batch:
        data.append(item[0])
        data_len.append(item[0].shape[1])
        target.append(item[1])
        target_len.append(item[1].shape[1])
        ids.append(torch.LongTensor([item[2]]).unsqueeze(0))
    max_data = max(data_len)
    max_tar = max(target_len)

    inputs = []
    targets = []
    for i in range(len(data)): 
        if data[i].size(1) < max_data:
            inputs.append(torch.cat((data[i], (torch.zeros(max_data - data[i].size(1))).type(torch.LongTensor).unsqueeze(0)),1))
        else:
            inputs.append(data[i])
            
        if target[i].size(1) < max_tar:
            targets.append(torch.cat((target[i], torch.ones(80, max_tar - target[i].size(1))*-45), 1))
        else:
            targets.append(target[i])

    return torch.stack(inputs, 0), torch.stack(targets, 0), torch.stack(ids,0)


class VCTKSets(Utils.Dataset):
    def __init__(self):
        self.text_input = list(pd.read_csv("corpus_text_taco.csv", header=None)[0].values)
        self.audio_input = list(pd.read_csv("corpus_audio_taco.csv", header=None)[0].values)
        
        self.audio_files = []
        self.id_files = []
        for i in range(len(self.audio_input)): 
            audio = torch.from_numpy(np.load(self.audio_input[i])).type(torch.FloatTensor)
            if audio.size(1) >= 600:
                del self.text_input[int(self.audio_input.index(self.audio_input[i]))]
                continue
            self.id_files.append(int(self.audio_input[i][9:12]))
            self.audio_files.append(audio)
        self.text_files = []
        for i in range(len(self.text_input)):
            text = self.text_input[i][1:-1].split(', ')
            text = [int(i) for i in text]
            self.text_files.append(torch.LongTensor([text]))
      
    def __len__(self):
        length = len(self.text_input)
        return length
    
    def __getitem__(self,index):
        return self.text_files[index], self.audio_files[index], self.id_files[index]


def trim_audio(input_data, threshold, frame):
    #Set frame length to audio length if 
    if input_data.size < frame:
        frame_length_temp = input_data.size
    else:frame_length_temp = frame
    
    energy = librosa.feature.rmse(y=input_data, frame_length=frame)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]
    # Note: indices can be an empty array, if the whole audio was silence.
    return input_data[indices[0]:indices[-1]] if indices.size else input_np[0:0]


def normalise_audio(audio):
    audio = audio - np.min(audio)
    return 2*(audio/(np.max(audio)))-1


def mu_encoder_np(input):
    QUANT = 64
    mu = QUANT-1
    x_mu = np.sign(input) * np.log1p(mu * np.abs(input)) / np.log1p(mu)
    x_mu = ((x_mu + 1) / 2 * mu + 0.5)
    return x_mu