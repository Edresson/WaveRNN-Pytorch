import os
from model import build_model
from torch import optim
from hparams import hparams as hp
import torch
import numpy as np
import  librosa
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model):

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def evaluate_model(model):
    """evaluate model and save generated wav

    """
    
    mel = np.load('00100.npy')
    wav = model.generate(mel)
    # save wav
    wav_path = "sample-0.wav"
    librosa.output.write_wav(wav_path, wav, sr=hp.sample_rate)
        

# build model
model = build_model().to(device)

model = load_checkpoint("checkpoints/checkpoint_step000340000.pth", model)
evaluate_model(model)
