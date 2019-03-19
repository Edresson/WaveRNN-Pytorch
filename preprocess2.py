"""
Preprocess dataset

usage: preproess.py [options] <wav-dir>

options:
     --output-dir=<dir>      Directory where processed outputs are saved. [default: data_dir].
    -h, --help              Show help message.
"""
import os
from docopt import docopt
import numpy as np
import math, pickle, os
from audio import *
from hparams import hparams as hp
from utils import *
from tqdm import tqdm
import librosa
import codecs

def get_wav_mel(path):
    """Given path to .wav file, get the quantized wav and mel spectrogram as numpy vectors

    """
    wav = load_wav(path)
    mel = melspectrogram(wav)
    if hp.input_type == 'raw':
        return wav.astype(np.float32), mel
    elif hp.input_type == 'mulaw':
        quant = mulaw_quantize(wav, hp.mulaw_quantize_channels)
        return quant.astype(np.int), mel
    elif hp.input_type == 'bits':
        quant = quantize(wav)
        return quant.astype(np.int), mel
    else:
        raise ValueError("hp.input_type {} not recognized".format(hp.input_type))





def process_data(wav_dir, output_path, mel_path, wav_path):
    """
    given wav directory and output directory, process wav files and save quantized wav and mel
    spectrogram to output directory
    """
    # create training and testing splits
    test_wav_files = []
    train_wav_files = []
    dataset_ids = []
    print(wav_dir)
    transcript = os.path.join(wav_dir, 'texts.csv')
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()
    for i in range(len(lines)):
        line = lines[i]
        print(line)
        fname,text = line.strip().split("==")
        file_id = '{:d}'.format(i).zfill(5)
        file_name = os.path.basename(fname)
        print(file_name)
        if int(file_name.split('-')[1].replace('.wav','')) >= 5655 and int(file_name.split('-')[1].replace('.wav',''))<=5674:
            test_wav_files.append([fname,file_id])
        else:
            train_wav_files.append([fname,file_id])
            
    
    wav_files = train_wav_files
    for i, lista in enumerate(tqdm(wav_files)):
        try:
            wav_file= lista[0]
            file_id = lista[1]
            if librosa.get_duration(filename=os.path.join(wav_dir,wav_file)) > 0.67:
                # get the file id
                
                
                    wav, mel = get_wav_mel(os.path.join(wav_dir,wav_file))
                    frames=mel.shape[1]
                    # save
                    #np.save(os.path.join(mel_path,file_id+".npy"), mel)
                    mel_ob = np.load(os.path.join('mel_ob',file_id+".npy"))
                    mel_ob = mel_ob[:][:frames]
                    print(mel_ob.shape,mel.shape)
                    np.save(os.path.join(mel_path,file_id+".npy"), mel_ob)
                    # save
                    #np.save(os.path.join(mel_path,file_id+".npy"), mel)
                    np.save(os.path.join(wav_path,file_id+".npy"), wav)
                    dataset_ids.append(file_id)
                

                
            else:
                continue
        except:
            continue

    # save dataset_ids
    with open(os.path.join(output_path,'dataset_ids.pkl'), 'wb') as f:
        pickle.dump(dataset_ids, f)

    # process testing_wavs
    test_path = os.path.join(output_path,'test')
    os.makedirs(test_path, exist_ok=True)
    for i, lista in enumerate(test_wav_files):
        wav_file= lista[0]
        file_id = lista[1]
        try:
            wav, mel = get_wav_mel(os.path.join(wav_dir,wav_file))
            mel_ob = np.load(os.path.join('mel_test',file_id+".npy"))
            mel_ob = mel_ob[:][:frames]
            print(mel_ob.shape,mel.shape)
            # save test_wavs
            np.save(os.path.join(test_path,"test_{}_mel.npy".format(i)),mel_ob)
            np.save(os.path.join(test_path,"test_{}_wav.npy".format(i)),wav)
        except:
                    continue

    
    print("\npreprocessing done, total processed wav files:{}.\nProcessed files are located in:{}".format(len(wav_files), os.path.abspath(output_path)))



if __name__=="__main__":
    args = docopt(__doc__)
    wav_dir = args["<wav-dir>"]
    output_dir = args["--output-dir"]

    # create paths
    output_path = os.path.join(output_dir,"")
    mel_path = os.path.join(output_dir,"mel")
    wav_path = os.path.join(output_dir,"wav")

    # create dirs
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(mel_path, exist_ok=True)
    os.makedirs(wav_path, exist_ok=True)

    # process data
    process_data(wav_dir, output_path, mel_path, wav_path)



def test_get_wav_mel():
    wav, mel = get_wav_mel('sample.wav')
    print(wav.shape, mel.shape)
    print(wav)
