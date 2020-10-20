import sys
sys.path.append('./waveglow')

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser

import torch
import numpy as np
from scipy.io.wavfile import write
import IPython.display as ipd


hparams = create_hparams()
hparams.sampling_rate = 22050

checkpoint_path = "./models/tacotron2_statedict.pt"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

waveglow_path = './models/waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

data = "Once upon a time there was a dear little girl who was loved by every one who looked at her, but most of all by her grandmother, and there was nothing that she would not have given to the child. Once she gave her a little cap of red velvet, which suited her so well that she would never wear anything else. So she was always called Little Red Riding Hood.One day her mother said to her, Come, Little Red Riding Hood, here is a piece of cake and a bottle of wine. Take them to your grandmother, she is ill and weak, and they will do her good."

sen = data.split(".")
len(sen)

# only first sentence TTS
text = sen[0] + "."
print(text)
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()
mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

audio = audio.cpu().numpy()[0]
audio = audio / np.abs(audio).max()

write('speech.wav',22050,audio)

