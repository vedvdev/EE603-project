import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
MAP={"speech":0,"music":1}

class AudioDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.num_samples=num_samples
        self.target_sample_rate=target_sample_rate
        self.device = device
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.audio_sample_list=self._get_audio_sample_path(audio_dir)
        self.transformation = transformation.to(self.device)
        

    def __len__(self):
        return len(self.audio_sample_list)

    def __getitem__(self, index):
        audio_sample_path = self.audio_sample_list[index]
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        print(length_signal)
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _get_audio_sample_path(self,audio_dir):
        
        audio_sample_list=[]
        for file in os.listdir(audio_dir):
            path=os.path.join(audio_dir,file)
            audio_sample_list.append(path)
        return audio_sample_list
    
    def _get_audio_sample_label(self, index):
        return MAP[self.annotations.iloc[index, 3]]


if __name__ == "__main__":
    ANNOTATIONS_FILE = "D:\project\log.csv"
    AUDIO_DIR = "D:\project\data\cleaned"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 160000

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")
    device="cpu"
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = AudioDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]
    print(signal.shape)