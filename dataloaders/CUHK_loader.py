# Author: Sai Krishna Prathapaneni
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch and https://github.com/dharwath/DAVEnet-pytorch
import json
import librosa
import numpy as np
import os
from PIL import Image
import pandas as pd
import scipy.signal
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import BertTokenizer

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])
# dataset_path = "/mnt/NAS/data/ruixuan/data/DLED/CUHK-PEDES/CUHK-PEDES/"
# json_path = os.path.join(dataset_path, "caption_all.json") 
class ImageCaptionDataset(Dataset):
    def __init__(self, args, dataset_json_file, type="train", text_conf = None, audio_conf=None, image_conf=None):
        """
        Dataset that manages a set of paired images and audio recordings

        :param dataset_json_file
        :param audio_conf: Dictionary containing the sample rate, window and
        the window length/stride in seconds, and normalization to perform (optional)
        :param image_transform: torchvision transform to apply to the images (optional)
        """
        self.args = args
        self.dir = dataset_json_file
        self.type = type
        if type =="train":
            self.path = os.path.join(dataset_json_file, "train_query_data.json")
        else:
            self.path = os.path.join(dataset_json_file, "test_sampled_data_with_ids_1k.json")
    
        # with open(path, 'r') as fp:
        #     data_json = json.load(path)

        self.data = pd.read_json(self.path)
        # self.image_base_path = data_json['file_path']
        # self.audio_base_path = data_json['audio_path']
       

        if not audio_conf:
            self.audio_conf = {}
        else:
            self.audio_conf = audio_conf

        if not image_conf:
            self.image_conf = {}
        else:
            self.image_conf = image_conf

        if not text_conf:
            self.text_conf ={}
        else:
            self.text_conf = text_conf

        crop_size = self.image_conf.get('crop_size', 224)
        center_crop = self.image_conf.get('center_crop', False)

        if center_crop:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        else:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.RandomResizedCrop(crop_size), transforms.ToTensor()])

        RGB_mean = self.image_conf.get('RGB_mean', [0.485, 0.456, 0.406])
        RGB_std = self.image_conf.get('RGB_std', [0.229, 0.224, 0.225])
        self.image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

        self.windows = {'hamming': scipy.signal.hamming,
        'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}
        
        #tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def _LoadAudio(self, path):
        audio_type = self.audio_conf.get('audio_type', 'melspectrogram')
        if audio_type not in ['melspectrogram', 'spectrogram']:
            raise ValueError('Invalid audio_type specified in audio_conf. Must be one of [melspectrogram, spectrogram]')
        preemph_coef = self.audio_conf.get('preemph_coef', 0.97)
        sample_rate = self.audio_conf.get('sample_rate', 16000)
        window_size = self.audio_conf.get('window_size', 0.025)
        window_stride = self.audio_conf.get('window_stride', 0.01)
        window_type = self.audio_conf.get('window_type', 'hamming')
        num_mel_bins = self.audio_conf.get('num_mel_bins', 40)
        target_length = self.audio_conf.get('target_length', 2048)
        use_raw_length = self.audio_conf.get('use_raw_length', False)
        padval = self.audio_conf.get('padval', 0)
        fmin = self.audio_conf.get('fmin', 20)
        n_fft = self.audio_conf.get('n_fft', int(sample_rate * window_size))
        win_length = int(sample_rate * window_size)
        hop_length = int(sample_rate * window_stride)

        # load audio, subtract DC, preemphasis
        y, sr = librosa.load(path, sr=sample_rate)
        if y.size == 0:
            y = np.zeros(200)
        y = y - y.mean()
        y = preemphasis(y, preemph_coef)
        # compute mel spectrogram
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length,
            window=self.windows.get(window_type, self.windows['hamming']))
        spec = np.abs(stft)**2
        if audio_type == 'melspectrogram':
            mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=num_mel_bins, fmin=fmin)
            melspec = np.dot(mel_basis, spec)
            logspec = librosa.power_to_db(melspec, ref=np.max)
        elif audio_type == 'spectrogram':
            logspec = librosa.power_to_db(spec, ref=np.max)
        n_frames = logspec.shape[1]
        if use_raw_length:
            target_length = n_frames
        p = target_length - n_frames
        if p > 0:
            logspec = np.pad(logspec, ((0,0),(0,p)), 'constant',
                constant_values=(padval,padval))
        elif p < 0:
            logspec = logspec[:,0:p]
            n_frames = target_length
        logspec = torch.FloatTensor(logspec)
        return logspec, n_frames

    def _LoadImage(self, impath):
        img = Image.open(impath).convert('RGB')
        # img = transforms.Compose(
        #         [transforms.ToTensor()])(img)
        img = self.image_resize_and_crop(img)
        #img = self.image_normalize(img) # needed while training and inference
        # else:
        #     transform = transforms.Compose(
        #         [transforms.Resize(256), transforms.ToTensor()])
        #     img = transform(img)
        #     img = self.image_normalize(img)
        return img
    
    def _LoadText(self, text):
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        
        return encoding.input_ids, encoding.attention_mask

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        datum = self.data.iloc[[index],:]
        wavpath = str(datum['audio_path'].iloc[0])
        imgpath = os.path.join(self.dir,"imgs",datum['file_path'].iloc[0])
        id = int(datum['id'].iloc[0])
        if not self.args.different_text_prompt: # using 1st caption, same text as from the audio description
            text = str(self.data.iloc[index]['caption_1'])  
            
        else:  # using 2nd caption, different text from the audio description
            text = str(self.data.iloc[index]['caption_2'])
        input_ids, attention_mask = self._LoadText(text)
        
        audio, nframes = self._LoadAudio(wavpath)
        image = self._LoadImage(imgpath)
        return image, audio, nframes, id , input_ids, attention_mask , text

    def __len__(self):
        return len(self.data)
    



if __name__ == "__main__":
    # Provide a valid path to your dataset JSON file
    dataset_path = "/mnt/NAS/data/ruixuan/data/DLED/CUHK-PEDES/CUHK-PEDES/"
   
    # dataset_path = json_padataset_path
    # Define audio and image configurations if needed
    # audio_config = {...}  # Fill in the audio configuration
    # image_config = {...}  # Fill in the image configuration

    # Instantiate the dataset
    caption_dataset = ImageCaptionDataset(dataset_path)

    # Check a sample from the dataset
    sample_index = 7  # Change this to access different samples
    sample = caption_dataset[sample_index]

    # Validate the sample
    image, audio, nframes, id= sample

    print(image.shape)
    print(audio.shape)
    print(nframes)
    print(id)
    # Check the shapes and content of image, audio, and nframes to ensure they're as expected
