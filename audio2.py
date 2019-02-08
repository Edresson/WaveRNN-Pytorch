import librosa
import librosa.filters
import math
import numpy as np
from scipy import signal
from hparams import hparams as hp
from scipy.io import wavfile


def load_wav(path):
    return librosa.load(path, sr=hp.sample_rate)[0]

def save_wav(wav, path):
    wav = wav * 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, hp.sample_rate, wav.astype(np.int16))

'''def melspectrogram(y):
    D = _lws_processor().stft(preemphasis(y)).T
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db
    if not hp.allow_clipping_in_normalization:
        assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    return _normalize(S)'''

def melspectrogram(y):
     # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])
 
    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sample_rate, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)


    return mel

# Fatcord's preprocessing
def quantize(x):
    """quantize audio signal

    """
    quant = (x + 1.) * (2**hp.bits - 1) / 2
    return quant.astype(np.int)


# testing
def test_everything():
    wav = np.random.randn(12000,)
    mel = melspectrogram(wav)
    quant = quantize(wav)
    print(wav.shape, mel.shape, spec.shape, quant.shape)
    print(quant.max(), quant.min(), mel.max(), mel.min())
