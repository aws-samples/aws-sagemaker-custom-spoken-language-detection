"""

  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  SPDX-License-Identifier: MIT-0
 
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
import torchaudio
import random
import librosa
import pandas as pd
import numpy as np
import os
import boto3
import scipy.io.wavfile as sciwav
import io
import tempfile

class ToMono:
    """ Convert stero audio to mono (single channel) """
    
    def __init__(self, channel_dim=0):
        self.channel_dim = channel_dim
    
    def __call__(self, tensor):
        if tensor.shape[self.channel_dim] > 1:
            return torch.mean(tensor, dim=self.channel_dim, keepdim=True)
        else:
            return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + "()"
        return format_string

class RandomPitchShift:
    """ Randomly shift pitch of audio """
    
    def __init__(self, sample_rate, pitch_range=(-2, 2), channel_dim=0):
        self.sample_rate = sample_rate
        self.pitch_range = pitch_range
        self.channel_dim = channel_dim
    
    def __call__(self, tensor):
        shift = random.randint(self.pitch_range[0], self.pitch_range[1])
        
        tensor = librosa.effects.pitch_shift(tensor.numpy().squeeze(), self.sample_rate, shift)
        return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)
    
    def __repr__(self):
        format_string = self.__class__.__name__ + "(pitch_range={})".format(self.pitch_range)
        return format_string
    
class RandomCrop:
    """ Randomly crop audio file """
    
    def __init__(self, n_samples):
        self.n_samples = n_samples
        
    def __call__(self, tensor):
        if tensor.shape[-1] <= self.n_samples:
            return tensor
        else:
            start = random.randint(0, tensor.shape[-1] - self.n_samples - 1)
            return tensor[:, start:start+self.n_samples]
        
class Truncate:
    """ Truncate end of audio file """
    
    def __init__(self, n_samples):
        self.n_samples = n_samples
        
    def __call__(self, tensor):
        if tensor.shape[-1] <= self.n_samples:
            return tensor
        else:
            return tensor[:, :self.n_samples]
    
class Pad:
    """ Pad end of audio file with 0s """
    
    def __init__(self, length):
        self.length = length
        
    def __call__(self, tensor):
        if tensor.shape[-1] < self.length:
            return F.pad(tensor, (0, self.length-tensor.shape[-1]), mode='constant')
        else:
            return tensor
        
class SpecFeatures:
    """ Extract spectral features from audio file """
    
    def __init__(self, sample_rate, feature_type='mel'):
        if feature_type not in ['mel', 'mfcc']:
            raise Exception('feature_type invalid')
            
        self.feature_type = feature_type
        
        if feature_type == 'mel':
            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=1024, hop_length=256
            )
        elif feature_type == 'mfcc':
            self.transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate, n_mfcc=40
            )
    
    def __call__(self, tensor):
        features = self.transform(tensor)
        if self.feature_type == 'mfcc':
            return features[:, 1:, :]
        else:
            return features
        
class Resize:
    """ Resize tensor """
    
    def __init__(self, size):
        if type(size) == list:
            size = tuple(size)
        
        self.size = size
        
    def __call__(self, tensor):
        return F.interpolate(tensor.unsqueeze(0), self.size).squeeze(0)

class Normalize:
    """ Normalize tensor between [0, 1] """
    
    def __call__(self, tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())
        
class Standardize:
    """ Standardize tensor to have zero center and unit variance """
    
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        if self.mean is None:
            self.mean = tensor.mean()
        
        if self.std is None:
            self.std = tensor.std()
            
        return (tensor - self.mean) / self.std
        
class FrequencyMask(object):
    """
    Implements frequency masking transform from SpecAugment paper (https://arxiv.org/abs/1904.08779)
    
      Example:
        >>> transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     FrequencyMask(max_width=10, use_mean=False),
        >>> ])

    """

    def __init__(self, max_width, use_mean=False):
        self.max_width = max_width
        self.use_mean = use_mean

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) where the frequency mask is to be applied.

        Returns:
            Tensor: Transformed image with Frequency Mask.
        """
        start = random.randrange(0, tensor.shape[2])
        end = start + random.randrange(0, self.max_width)
        if self.use_mean:
            tensor[:, start:end, :] = tensor.mean()
        else:
            tensor[:, start:end, :] = tensor.min()
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + "(max_width="
        format_string += str(self.max_width) + ")"
        return format_string


class TimeMask(object):
    """
    Implements time masking transform from SpecAugment paper (https://arxiv.org/abs/1904.08779)
    
      Example:
        >>> transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     TimeMask(max_width=10, use_mean=False),
        >>> ])

    """

    def __init__(self, max_width, use_mean=False):
        self.max_width = max_width
        self.use_mean = use_mean

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) where the time mask is to be applied.

        Returns:
            Tensor: Transformed image with Time Mask.
        """
        start = random.randrange(0, tensor.shape[1])
        end = start + random.randrange(0, self.max_width)
        if self.use_mean:
            tensor[:, :, start:end] = tensor.mean()
        else:
            tensor[:, :, start:end] = tensor.min()
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + "(max_width="
        format_string += str(self.max_width) + ")"
        return format_string
    
def create_audio_transform(sample_rate, n_samples, random_pitch_shift=False, pitch_range=(-5, 5), random_crop=False, 
                           feature_type='mel', resize=None, normalize=False, standardize=False, standardize_mean=None, 
                           standardize_std=None, frequency_masking=False, time_masking=False):
    """ Create transform pipeline for data preprocessing """
    
    transform = [ToMono()]
    
    if random_pitch_shift:
        transform.append(RandomPitchShift(sample_rate, pitch_range=pitch_range))
        
    if random_crop:
        transform.append(RandomCrop(n_samples))
    else:
        transform.append(Truncate(n_samples))
        
    transform.append(Pad(n_samples))
    
    transform.append(SpecFeatures(sample_rate, feature_type=feature_type))
    
    transform.append(torchaudio.transforms.AmplitudeToDB())
    
    if resize:
        transform.append(Resize(resize))
        
    if normalize:
        transform.append(Normalize())
        
    if standardize:
        transform.append(Standardize(mean=standardize_mean, std=standardize_std))
        
    if frequency_masking:
        transform.append(FrequencyMask(15, use_mean=False))
        
    if time_masking:
        transform.append(TimeMask(15, use_mean=False))
    
    return torchvision.transforms.Compose(transform)

def sliding_window_partition(tensor, window_size, overlap_size, min_size=None, dim=-1):
    """ Partition tensor into sliding segments """
    
    n_parts = int(tensor.shape[dim] / (window_size - overlap_size))
    
    if n_parts <= 1:
        return [tensor]
    else:
        partitions = []
        for i in range(n_parts):
            start = i*(window_size - overlap_size)

            if (min_size is not None) and (tensor.shape[dim] - start >= min_size):
                partitions.append(tensor[:, start:start+window_size])
        return partitions

def load_waveform(fname, sample_rate=16000, t_start=0):
    """ Load audio file locally or from s3 """
    
    if fname[:5] == 's3://':
        s3 = boto3.resource('s3')
        
        fname = fname.split('s3://')[1]
        bucket_name = fname.split('/')[0]
        key = '/'.join(fname.split('/')[1:])
        
        bucket = s3.Bucket(bucket_name)
        obj = bucket.Object(key)

        tmp = tempfile.NamedTemporaryFile()
        with open(tmp.name, 'wb') as f:
            obj.download_fileobj(f)
            waveform, sr = torchaudio.load(tmp.name)
    else:
        waveform, sr = torchaudio.load(fname)
        
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    waveform = waveform[:, int(t_start*sample_rate):]
    return waveform

class VoxForgeDataset(Dataset):
    """ Class to load and preprocess VoxForge audio files """
    
    def __init__(
        self, manifest_path, sample_rate=16000, n_samples=64000, feature_type='mel', 
        normalize=False, standardize=False, standardize_mean=None, standardize_std=None, 
        resize=None, shuffle=True, random_pitch_shift=False, pitch_range=(-5, 5), random_crop=False, 
        frequency_masking=False, time_masking=False, partition_audio=False, partition_overlap_seconds=2, 
        partition_min_seconds=4, include_source=False, languages='all', s3_prefix=None
    ):
        
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.feature_type = feature_type 
        self.normalize = normalize
        self.standardize = standardize
        self.standardize_mean = standardize_mean
        self.standardize_std = standardize_std
        self.resize = resize
        self.random_pitch_shift = random_pitch_shift
        self.pitch_range = pitch_range
        self.random_crop = random_crop
        self.frequency_masking = frequency_masking
        self.time_masking = time_masking
        self.partition_audio = partition_audio
        self.partition_overlap_seconds = partition_overlap_seconds
        self.partition_min_seconds = partition_min_seconds
        self.include_source = include_source
        self.languages = languages
        self.s3_prefix = s3_prefix
        
        if resize and (type(resize) != tuple and type(resize) != list):
            raise Exception('resize must be tuple or list')
            
        if partition_audio and random_crop:
            raise Exception('partition_audio and random_crop are mutually exclusive.  Only one can be True')
            
        self.data_df = pd.read_csv(manifest_path)
        
        # change file paths if using sagemaker training job
        if s3_prefix:
            self.data_df['fname'] = self.data_df['fname'].apply(lambda x : os.path.join(s3_prefix, x))
        
        if self.languages != 'all':
            valid_languages = ['en', 'de', 'fr', 'ru', 'it', 'es']
            
            if (type(self.languages) != list) or (not all([(x in valid_languages) for x in self.languages])):
                raise Exception('languages must be list of valid languages')
                
            if len(self.languages) > 1:
                self.data_df = self.data_df[self.data_df['class'].isin(self.languages)]
            elif len(self.languages) == 1:
                print('One language given, setting up dataset for binary classification')
                
                lang = self.languages[0]
                self.data_df['class'] = self.data_df['class'].apply(lambda x : 1 if x == lang else 0)
            else:
                raise Exception('no languages given')
                 
        
        if partition_audio:
            self._partition_audio()
        else:
            self.data_df['t_start'] = 0
        
        if shuffle:
            self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)
            
        self.transform = create_audio_transform(
            sample_rate, n_samples, random_pitch_shift=random_pitch_shift, random_crop=random_crop, 
            pitch_range=self.pitch_range, feature_type=feature_type, resize=resize, normalize=normalize, 
            standardize=standardize, standardize_mean=standardize_mean, standardize_std=standardize_std, 
            frequency_masking=frequency_masking, time_masking=time_masking
        )
        
    def _partition_audio(self):
        t_window = self.n_samples / self.sample_rate
        class_ctr = dict(self.data_df['class'].value_counts())

        new_data = []
        for k, v in class_ctr.items():
            temp = self.data_df[self.data_df['class'] == k]

            records = temp.to_dict(orient='records')
            for x in records:
                n_parts = int(x['time'] / (t_window - self.partition_overlap_seconds))
                
                for i in range(n_parts):
                    t_start = i*(t_window - self.partition_overlap_seconds)
                    if x['time'] - t_start >= self.partition_min_seconds:
                        new_data.append({
                            'fname' : x['fname'],
                            'source' : x['source'],
                            'class' : x['class'],
                            'time' : x['time'],
                            't_start' : t_start
                        })        
        self.data_df = pd.DataFrame(new_data)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        x = dict(self.data_df.iloc[idx])
        
        waveform = load_waveform(x['fname'], sample_rate=self.sample_rate, t_start=x['t_start'])
        features = self.transform(waveform)
        
        if self.include_source:
            return features, x['class'], x['source']
        else:
            return features, x['class']