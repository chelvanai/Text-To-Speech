import sys

sys.path.append('waveglow/')

import numpy as np
import torch
from hparams import create_hparams
from scipy.io.wavfile import write

from train import load_model
from text import text_to_sequence
from denoiser import Denoiser

hparams = create_hparams()
hparams.sampling_rate = 22050

# Load tacotron2
checkpoint_path = "./models/tacotron2_statedict.pt"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

print(model)

# Load waveglow
waveglow_path = './models/waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

print(waveglow)

# text = "The frog hopped at her heals back to her chair."
text = "Really I am speaking!"

sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()

mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666).float()

audio = audio.cpu().numpy()[0]
audio = audio / np.abs(audio).max()
print(audio.shape)
write('audio.wav', hparams.sampling_rate, audio)
