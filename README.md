# Tacotron2 + WaveGlow + GST
Tacotron2 + WaveGLow based Text to speech with gloable style tokens 


**Tacotron2 model architecture**

![Tacotron2 model](images/Tacotron2.png)

_The network is made of an encoder and a decoder with attention. The encoder uses a
character sequence and turns it into a feature representation. The input characters are
described using a 512-dimensional character embedding, which is given through a
stack of 3 convolutional layers. The convolutional layers improve in modeling longterm context. The output of the final layer is then fed to a unique bi-directional
LSTM, producing encoded features. The encoder output is consumed by an attention
network which compiles it into a fixed-length context vector.

The decoder is an autoregressive recurrent neural network that predicts a Mel
spectrogram from encoded sequences. The prediction of the earlier time step is passed
over 2 fully connected layers or pre-net. The pre-net output and attention context
vector are concatenated and passed into a stack of 2 unidirectional LSTM layers. The
connection of LSTM output and the attention context is measured through a linear
transform to predict the victim spectrogram frame. Ultimately, the predicted Melspectrogram is fed to a 5-layer convolutional Postnet which predicts a residual
connection to add to the prediction to increase the overall regeneration_

**Encoder part of Tacotron2 model**

![Encoder of Tacotron2](images/p1.png)

**Decoder with location sesitive attention**

![Encoder of Tacotron2](images/p2.png)

**Enhance mel-spectrogram with postnet**

 ![Encoder of Tacotron2](images/p3.png)
 
     def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm


**We first get speech dataset with annotated text file after we load the text, audio mel spectorgram in data loader**

**After in tacotron2 architecture, Encoder will extract the text data. after the decoder extract the feature of the mel spectogram with attention, and mapping theme. in this way the model learn the text to speech knowledge**

**Finnaly another CNN module enhance the decode output with help of postnet(CNN)**


    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)


1. Embedding layer encode the sentence
2. The embedding vector fed into **Encoder** module and generate future vector of the senetece.
3. The encoded context vector fed into decoder with mel spectgram. so the decoder run through attention layers and mapping the encoder context with mel spectrogram.
4. After the mel outputs fed into postnet, it will enhance the output.
5. finnaly it will update the weight parameter

**in this way many epoch it will map and learn the text to mel spectrogram relations.** 
 
