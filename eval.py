import os
from model import build_model
from torch import optim
from hparams import hparams as hp
import torch
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
    
    output_dir = '../eval/'
    mel = np.load('sample-0.npy')
    wav = model.generate(mel)
    # save wav
    wav_path = os.path.join(output_dir,"sample-0.wav")
    librosa.output.write_wav(wav_path, wav, sr=hp.sample_rate)
        

# build model
model = build_model().to(device)

model = load_checkpoint("checkpoints/", model)
evaluate_model(model)
